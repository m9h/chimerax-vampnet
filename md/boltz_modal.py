"""Boltz-2 ensemble generation on Modal.

Self-contained environment recipe — each Modal adapter in md/ builds
its own image rather than sharing a base, so dep collisions stay
isolated to the tool that needs them.

  Base image:   modal.Image.debian_slim(python_version=3.11)
                (intentionally NOT NGC — NGC PyTorch ships a scipy
                build that clashes with Boltz's pulled-in version;
                see "Tested" below for the 4-iteration NGC trail)
  Pip extras:   torch==2.4.1+cu124, boltz[cuda], biopython, mdtraj,
                huggingface_hub, einops, numpy<2, scipy, pyyaml
  Checkpoint:   Auto-downloaded by `boltz predict` from
                huggingface.co/boltz-community (boltz2_conf + boltz2_aff)
  GPU pin:      A100-80GB
  Tested:       2026-06-03 — 5-sample smoke ✓ (shape (5, 174, 3) CA)
                2026-06-03 — 200-sample run in progress

The v0.4 NGC attempts (4 iterations) all failed on the
scipy `_spropack` / `_promote` ABI cascade. The debian_slim path
trades the NGC PyTorch build's tuned kernels for a clean ABI surface
and "just works" — Boltz's bundled CUDA kernels are self-contained
enough that we don't lose the perf the NGC base would have given us.

Wraps `boltz predict` to generate N diffusion samples of a single
protein sequence and pack them as an .npz the chimerax-vampnet
bundle's `vampnet load_ensemble ... source bioemu` consumes (Boltz-2
output is shape-compatible with BioEmu's, so we route through the
same loader for now; a dedicated `boltz` loader path is a v0.5 nicety).

  modal run md/boltz_modal.py::sample \\
      --sequence "ACELPECQ..." --name notch1_NEC \\
      --n-samples 25 --out notch1_NEC_boltz.npz

Per-sequence cost on A100-80GB: a few minutes per 25 samples
(Boltz-2's diffusion sampler is much faster than AF3-style sampling).
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-boltz"

image = (
    # v0.5: switched off NGC PyTorch (whose bundled scipy/scikit-learn
    # ABIs collide with Boltz's pip-pulled versions) to a clean
    # debian_slim + pip toolchain. Per the plan at
    # ~/.claude/plans/twinkly-whistling-bonbon.md.
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "boltz[cuda]",
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "einops",
        "numpy<2",
        "scipy",
        "pyyaml",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A100-80GB", timeout=2 * 3600)
def sample_remote(sequence: str, name: str, n_samples: int = 25,
                   use_msa_server: bool = False) -> bytes:
    """Run Boltz-2 inference on a single sequence. Returns packed
    ensemble (all-atom + CA-only) as .npz bytes."""
    import io
    import subprocess
    import sys
    import tempfile
    import yaml

    import numpy as np
    import mdtraj as md

    print(f"[boltz] sequence length {len(sequence)}, n_samples={n_samples}, "
          f"use_msa_server={use_msa_server}")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        in_yaml = tmp / f"{name}.yaml"
        # Single-protein YAML per the Boltz docs. We skip msa entirely
        # for offline single-sequence inference (Boltz tolerates this
        # for monomeric sequence with a small performance hit).
        spec = {"sequences": [{"protein": {"id": "A",
                                            "sequence": sequence}}]}
        if not use_msa_server:
            spec["sequences"][0]["protein"]["msa"] = "empty"
        in_yaml.write_text(yaml.safe_dump(spec))

        out_dir = tmp / "out"
        out_dir.mkdir()

        cmd = [
            "boltz", "predict", str(in_yaml),
            "--out_dir", str(out_dir),
            "--diffusion_samples", str(n_samples),
            "--output_format", "pdb",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        print(f"[boltz] {' '.join(cmd)}")
        sys.stdout.flush()
        subprocess.run(cmd, check=True)

        # Boltz writes predictions/<name>/<name>_model_*.pdb
        pred_dir = out_dir / "predictions" / name
        pdb_files = sorted(pred_dir.glob(f"{name}_model_*.pdb"))
        if not pdb_files:
            # Fallback: try any PDB under out_dir.
            pdb_files = sorted(out_dir.rglob("*.pdb"))
        if not pdb_files:
            raise RuntimeError(f"Boltz-2 wrote no PDB outputs to {out_dir}")
        print(f"[boltz] {len(pdb_files)} models produced")

        # Stack each per-model PDB into a (N, A, 3) array.
        trajs = [md.load_pdb(str(p)) for p in pdb_files]
        n_atoms = trajs[0].n_atoms
        if not all(t.n_atoms == n_atoms for t in trajs):
            raise RuntimeError("Boltz models have inconsistent atom counts")
        coords_all = np.stack(
            [(t.xyz[0] * 10.0).astype(np.float32) for t in trajs],
            axis=0,
        )  # (N, A, 3) in Angstroms
        ca_indices = [a.index for a in trajs[0].topology.atoms
                       if a.name == "CA"]
        coords_ca = coords_all[:, ca_indices, :]
        print(f"[boltz] all-atom {coords_all.shape}, CA-only {coords_ca.shape}")

        buf = io.BytesIO()
        np.savez_compressed(buf, coords=coords_all, coords_ca=coords_ca,
                            seqres=np.array(sequence))
        return buf.getvalue()


@app.local_entrypoint()
def sample(sequence: str, name: str = "protein", n_samples: int = 25,
           use_msa_server: bool = False, out: str = ""):
    """Generate a Boltz-2 ensemble and save locally."""
    print(f"[local] requesting {n_samples} samples of seq len {len(sequence)}")
    data = sample_remote.remote(sequence, name, n_samples=n_samples,
                                  use_msa_server=use_msa_server)
    out_path = Path(out) if out else Path(f"{name}_boltz.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load with: vampnet load_ensemble {name}_boltz "
          f"{out_path} format bioemu  # (boltz loader = v0.5)")
