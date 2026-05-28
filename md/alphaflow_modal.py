"""AlphaFlow ensemble generation on Modal.

Wraps Bowen Jing et al.'s AlphaFlow (github.com/bjing2016/alphaflow,
ICML 2024) to produce N conformational samples from a single protein
sequence. Output is saved as an AlphaFlow-shape .npz that the bundle's
`vampnet load_ensemble path source alphaflow` consumes natively.

Inference settings follow the "alphaflow-md-base" recipe -- the
checkpoint trained on MD-like ensemble distributions, which is the
right comparison for chignolin/notch1 MD analyses.

  modal run alphaflow_modal.py::sample --sequence "YYDPETGTWY" --n 200 --out chignolin_af.npz

Notes:
  - AlphaFlow MD checkpoint is ~3 GB; the Modal image caches it across
    runs in /root/.cache/huggingface so repeat invocations don't re-download.
  - 200 samples on an H100 takes ~10 minutes for a small protein and
    burns ~$1.5 of Modal credit.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-alphaflow"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        "numpy<2",
        "scipy",
        "biopython",
        "huggingface_hub",
        "einops",
        "tqdm",
        "pyyaml",
        "ml-collections",
        "absl-py",
        "dm-tree",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        # AlphaFlow has its own OpenFold-derived stack baked in.
        "cd /opt && git clone https://github.com/bjing2016/alphaflow.git",
        "cd /opt/alphaflow && pip install -e . --no-deps",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="H100", timeout=3600)
def sample_remote(sequence: str, n_samples: int = 200,
                   checkpoint: str = "alphaflow-md-base") -> bytes:
    """Run AlphaFlow inference. Returns the .npz file as bytes."""
    import subprocess
    import tempfile

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    print(f"[af] downloading checkpoint {checkpoint}")
    ckpt_path = hf_hub_download(repo_id="bjing2016/alphaflow",
                                 filename=f"{checkpoint}.pt",
                                 cache_dir="/root/.cache/huggingface")
    print(f"[af] checkpoint at {ckpt_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Write a single-protein FASTA the inference script expects.
        fasta = tmp / "query.fasta"
        fasta.write_text(f">query\n{sequence}\n")
        out_dir = tmp / "out"
        out_dir.mkdir()
        cmd = [
            "python", "/opt/alphaflow/predict.py",
            "--input_fasta", str(fasta),
            "--samples", str(n_samples),
            "--weights", ckpt_path,
            "--outpdb", str(out_dir),
            "--mode", "alphafold",   # MD-conditioned
        ]
        print(f"[af] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # AlphaFlow writes one PDB per sample; aggregate into (N, A, 3).
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        sample_paths = sorted(out_dir.glob("*.pdb"))
        if not sample_paths:
            raise RuntimeError("AlphaFlow produced no PDB outputs")
        all_coords = []
        for p in sample_paths:
            s = parser.get_structure(p.stem, str(p))
            atoms = [a for a in s.get_atoms()]
            coords = np.array([a.get_coord() for a in atoms], dtype="float32")
            all_coords.append(coords)
        coords = np.stack(all_coords, axis=0)
        print(f"[af] ensemble shape: {coords.shape}")

        out_npz = tmp / "ensemble.npz"
        np.savez_compressed(out_npz, coords=coords)
        return out_npz.read_bytes()


@app.local_entrypoint()
def sample(sequence: str, n: int = 200, out: str = "alphaflow_ensemble.npz",
           checkpoint: str = "alphaflow-md-base"):
    """Generate an AlphaFlow ensemble and save to a local .npz."""
    print(f"[local] generating {n} samples for seq len {len(sequence)}")
    data = sample_remote.remote(sequence, n, checkpoint)
    Path(out).write_bytes(data)
    print(f"[local] wrote {out} ({len(data)/(1<<20):.1f} MB)")
