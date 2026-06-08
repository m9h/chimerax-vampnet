"""Plainer EDM (energy-based diffusion w/ Fokker-Planck regulariser) on Modal.

Paper: Plainer, Wu, Klein, Günnemann, Noé 2026, "Consistent Sampling and
Simulation: Molecular Dynamics with Energy-Based Diffusion Models" (NeurIPS
2025; arXiv:2506.17139). Code at https://github.com/noegroup/ScoreMD (MIT).
Presented at Starkly Speaking 2026-01-12.

Why it earns a slot in the v0.9+ multisource pipeline:

  Plainer's contribution is *consistency* — diffusion samplers trained on
  equilibrium distributions usually have score errors at small t that violate
  the Fokker-Planck equation, so the model's energy interpretation drifts
  from the trained distribution. ScoreMD adds an FP-regularisation term that
  makes the same network usable as both (a) a Boltzmann sampler and (b) a
  Langevin-MD-style integrator. This is exactly the failure mode our v0.7+v0.8
  H3 analysis exposed in BioEmu/AlphaFlow/ESMFold2 (modal-state collapse,
  ESMFold2 Rg std=0.02 Å). ScoreMD is the cleanest published candidate for a
  generative source whose ensemble is *physically* consistent with its energy.

  The released checkpoint is a transferable Boltzmann emulator for *dipeptides*,
  trained on a corpus of dipeptide MD trajectories. Scaling beyond dipeptides
  is an open research direction; the v0.9 use of this adapter is the dipeptide
  fold-class benchmark suite (alanine dipeptide chignolin-fragments etc.) as a
  sanity check on the H3 "generative collapse" finding before claiming
  generality.

  Base image:   modal.Image.debian_slim(python_version=3.11)
                ScoreMD is pure PyTorch + JAX notebooks; we use the PyTorch
                path here. Cleanish ABI vs NGC; same rationale as boltz/esmfold2.
  Pip extras:   torch (cu124), the ScoreMD repo from git, JAX (CPU is fine
                for the PyTorch inference path), biopython, mdtraj, numpy<2.
  Checkpoint:   Auto-pulled from the ScoreMD release (dipeptide Boltzmann
                emulator); the exact HF/Zenodo path is TBD on first run —
                inspect the repo's `scripts/sample.py` for the canonical name.
  GPU pin:      A100-40GB (the released model is ~50M params, small).
  Tested:       UNTESTED as of 2026-06-07 — scaffold only. Two known
                unknowns before first invocation:
                  1. The repo's package name (likely `scoremd` or `noegroup_scoremd`)
                     and its exact sample API.
                  2. Whether the released checkpoint expects dipeptide-only
                     input or accepts arbitrary peptides; the README will say.
                Follow the ESMFold2 v0.7.5→v0.8 fix pattern: first invocation
                will likely reveal an import or API mismatch; patch and re-run.

  modal run md/edm_plainer_modal.py::sample \\
      --sequence "AA" --name ala_dipeptide \\
      --n-samples 200 --out ala_dipeptide_edm.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-edm-plainer"
REPO_URL = "git+https://github.com/noegroup/ScoreMD.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"scoremd @ {REPO_URL}",  # TODO: verify package name on first install
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "numpy<2",
        "einops",
        # gemmi: parsing fallback (ScoreMD likely emits PDB/mmCIF too)
        "gemmi",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A100-40GB", timeout=2 * 3600)
def sample_remote(sequence: str, name: str, n_samples: int = 200,
                  mode: str = "sample", n_steps: int = 1000) -> bytes:
    """Run Plainer EDM on a peptide. mode="sample" draws iid Boltzmann
    samples; mode="simulate" runs a Langevin-MD-like trajectory of n_steps.

    Returns .npz bytes with the same schema as md/esmfold2_modal.py
    (coords / coords_ca / seqres / chain_id / plddt placeholder).
    """
    import io

    import numpy as np
    import torch

    # First-invocation TODO: the exact import path is repo-version-dependent.
    # The README's sample.py is the source of truth. Adjust on first run.
    from scoremd import EDM, sample_boltzmann, simulate_md  # noqa: F401 — verify

    print(f"[edm-plainer] sequence={sequence} ({len(sequence)} residues), "
          f"mode={mode}, n_samples={n_samples}, n_steps={n_steps}")

    model = EDM.from_pretrained("noegroup/ScoreMD-dipeptides")  # TODO: verify HF path
    model = model.cuda().eval()

    if mode == "sample":
        coords_all = sample_boltzmann(model, sequence, n_samples=n_samples)
    elif mode == "simulate":
        coords_all = simulate_md(model, sequence, n_steps=n_steps,
                                  save_every=max(1, n_steps // n_samples))
    else:
        raise ValueError(f"mode must be 'sample' or 'simulate', got {mode}")

    # Expect coords_all shape (n_samples, n_atoms, 3) in Angstroms.
    coords_all = np.asarray(coords_all, dtype=np.float32)
    print(f"[edm-plainer] coords shape: {coords_all.shape}")

    # Heuristic CA mask — the released dipeptide model uses heavy atoms only,
    # so we may need to infer CA indices from the topology. ScoreMD's repo
    # exposes a topology helper; plumb that in on first run.
    # As a placeholder, take every 4th heavy atom (ALA dipeptide topology hack).
    # The real adapter will use scoremd.topology(sequence).ca_indices().
    n_atoms = coords_all.shape[1]
    ca_stride = max(1, n_atoms // max(1, len(sequence)))
    ca_idx = np.arange(0, n_atoms, ca_stride)[: len(sequence)]
    coords_ca = coords_all[:, ca_idx, :]

    chain_id_per_res = np.zeros(len(sequence), dtype=np.int64)
    plddt = np.full(n_samples, np.nan, dtype=np.float32)
    iptm = np.full(n_samples, np.nan, dtype=np.float32)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=coords_all,
        coords_ca=coords_ca,
        seqres=np.array(sequence),
        chain_id=chain_id_per_res,
        plddt=plddt,
        iptm=iptm,
        mode=np.array(mode),
    )
    return buf.getvalue()


@app.local_entrypoint()
def sample(sequence: str, name: str = "peptide", n_samples: int = 200,
           mode: str = "sample", n_steps: int = 1000, out: str = ""):
    """Generate an EDM-Plainer ensemble and save locally.

    mode="sample"    -> n iid Boltzmann samples (the H3 use case)
    mode="simulate"  -> a single MD-like trajectory of n_steps Langevin steps;
                        coords saved every n_steps/n_samples (the consistency
                        demo — should statistically match mode="sample")
    """
    print(f"[local] requesting {n_samples} EDM samples for {name} "
          f"(seq len {len(sequence)}, mode={mode})")
    data = sample_remote.remote(sequence, name, n_samples=n_samples,
                                 mode=mode, n_steps=n_steps)
    out_path = Path(out) if out else Path(f"{name}_edm.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load with: vampnet load_ensemble {name}_edm "
          f"{out_path} format alphaflow")  # reuse alphaflow loader for now
