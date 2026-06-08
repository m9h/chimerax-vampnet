"""Prose (transferable normalising flow for peptides) on Modal.

Paper: Tan, Hassan, Klein, Syed, Beaini, Bronstein, Tong, Neklyudov 2025,
"Amortized Sampling with Transferable Normalizing Flows" (arXiv:2508.18175).
Code at https://github.com/transferable-samplers/transferable-samplers (MIT).
Presented at Starkly Speaking 2025-09-08.

Why it earns a slot in the v0.9+ multisource pipeline:

  Prose is a 280 M-param all-atom normalising flow trained on ManyPeptidesMD
  (21,700 peptide sequences × 200 ns each = 4.3 ms of MD). Once trained, it
  draws zero-shot, *uncorrelated* proposal samples for arbitrary peptide
  systems — the "transferability" claim that MarS-FM also makes but Prose
  arrives at via a different objective (likelihood-based, importance-sampling-
  refinable) rather than flow-matching. A second-opinion sampler in the
  transferable-flow class is exactly what the v0.7+v0.8 H3 analysis needs to
  separate "this finding is a Prose-class artefact" from "this finding holds
  across the transferable-flow family".

  Applicability note: Prose was trained on peptides of 2-8 residues; the
  authors demonstrate transferability *across* peptide length within that
  range, not extrapolation to 174-residue systems like Notch1 NEC. The v0.9
  use of this adapter is therefore the dipeptide/tetrapeptide benchmark axis,
  not direct application to the v0.7/v0.8 systems. Treat any output on >8-mer
  inputs as exploratory, not a calibrated sampler.

  Base image:   modal.Image.debian_slim(python_version=3.11)
  Pip extras:   torch (cu124), the transferable-samplers repo from git,
                the TarFlow architecture deps, biopython, mdtraj, numpy<2.
  Checkpoint:   Pretrained Prose weights — released on HF or GitHub releases
                per the README; verify the exact URI on first invocation.
  GPU pin:      A100-80GB (280 M params; should fit comfortably).
  Tested:       UNTESTED as of 2026-06-07 — scaffold only. Three known
                unknowns:
                  1. The pip-installable package name (likely the repo name
                     `transferable-samplers` or a snake_case variant).
                  2. The exact import path for the Prose model class
                     (`from transferable_samplers.prose import Prose`?).
                  3. Whether sampling supports an explicit temperature
                     argument (the abstract mentions transferability across
                     temperature, but the sampling API may default to 300 K).

  modal run md/prose_modal.py::sample \\
      --sequence "AAAAA" --name penta_ala \\
      --n-samples 200 --out penta_ala_prose.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-prose"
REPO_URL = "git+https://github.com/transferable-samplers/transferable-samplers.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"transferable-samplers @ {REPO_URL}",  # TODO: verify package name
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "numpy<2",
        "einops",
        "gemmi",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A100-80GB", timeout=2 * 3600)
def sample_remote(sequence: str, name: str, n_samples: int = 200,
                  temperature_K: float = 300.0,
                  finetune_steps: int = 0) -> bytes:
    """Draw n iid Prose samples for a peptide.

    If finetune_steps > 0, run the importance-sampling-based finetune the
    paper proposes (improves accuracy on unseen tetrapeptides per Sec. 5);
    set to 0 for pure zero-shot transfer.
    """
    import io

    import numpy as np
    import torch

    # First-invocation TODOs (mirroring ESMFold2 v0.7.5→v0.8 pattern):
    #   - Verify import path matches the repo's __init__.py.
    #   - Verify the .from_pretrained signature; may need an explicit URI
    #     instead of an HF identifier.
    from transferable_samplers.prose import Prose  # noqa: F401 — verify

    if len(sequence) > 8:
        print(f"[prose] WARNING: sequence length {len(sequence)} > 8; "
              "outside training distribution. Treat output as exploratory.")

    print(f"[prose] sequence={sequence} ({len(sequence)} residues), "
          f"n_samples={n_samples}, T={temperature_K} K, "
          f"finetune_steps={finetune_steps}")

    model = Prose.from_pretrained("transferable-samplers/prose-280M")
    model = model.cuda().eval()

    if finetune_steps > 0:
        # Importance-sampling-based finetune per Sec 5 of the paper.
        model.finetune(sequence, n_steps=finetune_steps, temperature=temperature_K)

    coords_all = model.sample(sequence, n_samples=n_samples,
                              temperature=temperature_K)
    coords_all = np.asarray(coords_all, dtype=np.float32)
    print(f"[prose] coords shape: {coords_all.shape}")

    # Prose is all-atom; extract CA via the model's atom-name list.
    # TODO on first run: pull the canonical atom-name list from
    # model.atom_names or model.topology(sequence) — the placeholder
    # heuristic is correct for backbone-N-CA-C-O ordering only.
    atom_names = getattr(model, "atom_names", None)
    if atom_names is not None:
        ca_mask = np.array([n == "CA" for n in atom_names], dtype=bool)
    else:
        n_atoms = coords_all.shape[1]
        ca_stride = max(1, n_atoms // max(1, len(sequence)))
        ca_mask = np.zeros(n_atoms, dtype=bool)
        ca_mask[np.arange(1, n_atoms, ca_stride)[: len(sequence)]] = True
    coords_ca = coords_all[:, ca_mask, :]

    chain_id_per_res = np.zeros(len(sequence), dtype=np.int64)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=coords_all,
        coords_ca=coords_ca,
        seqres=np.array(sequence),
        chain_id=chain_id_per_res,
        plddt=np.full(n_samples, np.nan, dtype=np.float32),
        iptm=np.full(n_samples, np.nan, dtype=np.float32),
        temperature_K=np.array(temperature_K, dtype=np.float32),
    )
    return buf.getvalue()


@app.local_entrypoint()
def sample(sequence: str, name: str = "peptide", n_samples: int = 200,
           temperature: float = 300.0, finetune_steps: int = 0, out: str = ""):
    """Generate a Prose ensemble and save locally."""
    print(f"[local] requesting {n_samples} Prose samples for {name} "
          f"(seq len {len(sequence)}, T={temperature} K)")
    data = sample_remote.remote(sequence, name, n_samples=n_samples,
                                 temperature_K=temperature,
                                 finetune_steps=finetune_steps)
    out_path = Path(out) if out else Path(f"{name}_prose.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load with: vampnet load_ensemble {name}_prose "
          f"{out_path} format alphaflow")
