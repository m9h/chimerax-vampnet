"""BioEmu (Microsoft Research) ensemble generation on Modal.

Wraps the `bioemu` package's `bioemu.sample` CLI to generate a
conformational ensemble for a single protein sequence. Output is a
PDB topology + XTC trajectory; the adapter packs them into a single
`.npz` that the chimerax-vampnet bundle's `vampnet load_ensemble ...
source bioemu` consumes natively.

  modal run md/bioemu_modal.py::sample \\
      --sequence "ACELPECQ..." --name notch1_NEC \\
      --n-samples 200 --out notch1_NEC_bioemu.npz

Per-sequence cost on A100-80GB (~per BioEmu README): 4-150 min per
1000 samples depending on sequence length. Notch1 NEC at 174 residues
expected ~$5-10 for 1000 samples vs >$1000 for the equivalent us-MD.

Notes:
  - BioEmu only supports MONOMERS. For the Notch1 holo NRR+Fab, you'd
    run the NRR sequence alone (no Fab conditioning).
  - The bioemu package pip-installs cleanly and auto-downloads
    checkpoints from huggingface.co/microsoft/bioemu (default v1.1).
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-bioemu"

image = (
    # NGC PyTorch base for tuned CUDA + cuDNN + torch.
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.04-py3",
                                add_python=None)
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "bioemu[cuda]",  # the headline install
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "einops",
        # numpy/scipy/pandas/tqdm already in NGC pytorch image
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A100-80GB", timeout=4 * 3600)
def sample_remote(sequence: str, name: str, n_samples: int = 200,
                   model_version: str = "bioemu-v1.1") -> bytes:
    """Run BioEmu inference on a single sequence. Returns the packed
    ensemble (all-atom + CA-only) as .npz bytes."""
    import io
    import subprocess
    import sys
    import tempfile

    import numpy as np
    import mdtraj as md

    print(f"[bioemu] sequence length {len(sequence)}, n_samples={n_samples}")
    if len(sequence) > 600:
        print(f"[bioemu] WARNING: seq len {len(sequence)} > 600; will be slow")

    with tempfile.TemporaryDirectory() as tmpd:
        out_dir = Path(tmpd) / "out"
        out_dir.mkdir()

        cmd = [
            sys.executable, "-m", "bioemu.sample",
            "--sequence", sequence,
            "--num_samples", str(n_samples),
            "--output_dir", str(out_dir),
            "--model_name", model_version,
        ]
        print(f"[bioemu] {' '.join(cmd[:6])} ... (sequence truncated for log)")
        sys.stdout.flush()
        subprocess.run(cmd, check=True)

        # BioEmu writes samples.pdb (topology) + samples.xtc (trajectory).
        pdb = out_dir / "samples.pdb"
        xtc = out_dir / "samples.xtc"
        if not (pdb.exists() and xtc.exists()):
            # Some versions write under a subdir; try to find them.
            pdbs = list(out_dir.rglob("*.pdb"))
            xtcs = list(out_dir.rglob("*.xtc"))
            if pdbs and xtcs:
                pdb, xtc = pdbs[0], xtcs[0]
            else:
                raise RuntimeError(
                    f"could not find samples.pdb/xtc in {out_dir}; "
                    f"found pdbs={pdbs} xtcs={xtcs}")
        traj = md.load(str(xtc), top=str(pdb))
        coords_all = (traj.xyz * 10.0).astype(np.float32)  # nm -> A
        ca_indices = [a.index for a in traj.topology.atoms if a.name == "CA"]
        coords_ca = coords_all[:, ca_indices, :]
        print(f"[bioemu] generated {coords_all.shape[0]} frames; "
              f"all-atom {coords_all.shape}, CA-only {coords_ca.shape}")

        buf = io.BytesIO()
        np.savez_compressed(buf, coords=coords_all, coords_ca=coords_ca,
                            seqres=np.array(sequence))
        return buf.getvalue()


@app.local_entrypoint()
def sample(sequence: str, name: str = "protein", n_samples: int = 200,
           model_version: str = "bioemu-v1.1", out: str = ""):
    """Generate a BioEmu ensemble and save locally."""
    print(f"[local] requesting {n_samples} samples of seq len {len(sequence)} "
          f"({model_version})")
    data = sample_remote.remote(sequence, name, n_samples=n_samples,
                                  model_version=model_version)
    out_path = Path(out) if out else Path(f"{name}_bioemu.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load with: vampnet load_ensemble {name}_bioemu "
          f"{out_path} format bioemu")
