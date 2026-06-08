"""OM-TPS (Onsager-Machlup transition path sampling) on Modal.

Paper: Raja, Šípka, Psenka, Kreiman, Pavelka, Krishnapriyan 2025,
"Action-Minimization Meets Generative Modeling: Efficient Transition Path
Sampling with the Onsager-Machlup Functional" (ICML 2025; arXiv:2504.18506).
Code at https://github.com/ASK-Berkeley/OM-TPS (MIT). Presented at Starkly
Speaking 2025-06-09.

Why it earns a slot in the v0.9+ multisource pipeline:

  OM-TPS repurposes *any* pretrained score-based generative model (diffusion
  or flow matching, including AF3-class structure predictors and MLFFs like
  MACE-OFF) for transition path sampling. The score function induces a
  stochastic dynamics, and probable transition paths are obtained by
  minimising the Onsager-Machlup action functional on the resulting SDE.
  Critically, this is *zero-shot* and *task-agnostic* — no TPS-specific
  training, no collective variables, no chemistry intuition.

  Direct relevance to this project:
    - v0.7 W3d Phase 2 (Notch1 AF-state-1 seeded MD bursts) attempted to
      generate paths connecting MD state 0 to AF state 1 via steered MD;
      OM-TPS gives a cleaner path-sampling primitive that uses the same
      AlphaFlow/ESMFold2 score we already have on the volume.
    - β2AR active↔inactive transition is the canonical TPS target — OM-TPS
      with a MACE-OFF or UMA score (see md/uma_modal.py) is the principled
      way to build that ensemble.

  Output schema is different from other generative adapters: instead of
  (n_samples, n_atoms, 3) iid frames, OM-TPS emits (n_paths, n_steps_per_path,
  n_atoms, 3) trajectories. The H3 pipeline ingests these by flattening to
  (n_paths * n_steps, n_atoms, 3) with a path-id tag for stratification.

  Base image:   modal.Image.debian_slim(python_version=3.11)
  Pip extras:   torch (cu124), OM-TPS from git, the backbone sampler the
                user picks (MACE-OFF / AlphaFlow / ESMFold2 — the adapter
                forwards the choice to OM-TPS via --backbone).
  GPU pin:      A100-80GB.
  Tested:       UNTESTED as of 2026-06-07 — scaffold only. The biggest
                first-run unknown is the OM-TPS API for plugging in an
                external pretrained sampler: check the repo's
                `om_tps/backbones/` (or analogous) for the registry pattern.

  modal run md/om_tps_modal.py::sample \\
      --start-pdb md/3i08_apo.pdb \\
      --end-pdb md/3l95_holo.pdb \\
      --name notch1_apo_to_holo \\
      --n-paths 16 --n-steps 100 --backbone mace-off \\
      --out notch1_apo_to_holo_omtps.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-om-tps"
REPO_URL = "git+https://github.com/ASK-Berkeley/OM-TPS.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"om-tps @ {REPO_URL}",  # TODO: verify package name on first run
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "numpy<2",
        "einops",
        "gemmi",
        # Backbone sampler deps — install all up front to avoid per-run
        # image rebuilds. MACE-OFF is the default backbone in the paper.
        "mace-torch",  # MACE foundation model
        "ase",  # atomic-simulation environment glue
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A100-80GB", timeout=4 * 3600)
def sample_remote(start_pdb_bytes: bytes, end_pdb_bytes: bytes, name: str,
                  n_paths: int = 16, n_steps: int = 100,
                  backbone: str = "mace-off",
                  optimiser_steps: int = 1000) -> bytes:
    """Sample n_paths transition trajectories from start_pdb to end_pdb,
    each of n_steps frames, by minimising the Onsager-Machlup action under
    the dynamics induced by the chosen backbone score model.

    backbone: one of
      - "mace-off"      : MACE-OFF foundation MLFF (paper default)
      - "esmfold2"      : use the user's ESMFold2 checkpoint as a score
                          (per md/esmfold2_modal.py); requires the
                          checkpoint to be in the volume.
      - "alphaflow"     : ditto for AlphaFlow.

    Output schema:
      coords          : (n_paths, n_steps, n_atoms, 3)  Å, all-atom
      coords_ca       : (n_paths, n_steps, n_ca, 3)     Å, CA-only
      seqres          : str
      chain_id        : (n_ca,)
      path_action     : (n_paths,) final OM action per path (lower is better)
      converged       : (n_paths,) bool
    """
    import io

    import numpy as np
    import torch

    # First-invocation TODO: the exact OM-TPS API for a generic backbone is
    # `om_tps.sample(start, end, score_fn, n_paths, n_steps)` per the
    # repo's examples; verify the keyword names on first run.
    from om_tps import sample_paths  # noqa: F401 — verify

    print(f"[om-tps] name={name}, backbone={backbone}, n_paths={n_paths}, "
          f"n_steps={n_steps}, optimiser_steps={optimiser_steps}")

    start_pdb = Path("/tmp/start.pdb")
    end_pdb = Path("/tmp/end.pdb")
    start_pdb.write_bytes(start_pdb_bytes)
    end_pdb.write_bytes(end_pdb_bytes)

    # Load the backbone score model. Each branch returns a callable
    # `score_fn(x_t, t) -> ∇log p_t(x_t)` that OM-TPS expects.
    if backbone == "mace-off":
        from mace.calculators import mace_off
        score_fn = mace_off(model="medium", device="cuda")  # API per mace-torch README
    elif backbone == "esmfold2":
        # Reuse the ESMFold2 image's import path; the score wrapper unwraps
        # the diffusion model into a (x_t, t) -> score callable.
        from om_tps.backbones.esmfold2 import esmfold2_score
        score_fn = esmfold2_score(checkpoint="biohub/ESMFold2", device="cuda")
    elif backbone == "alphaflow":
        from om_tps.backbones.alphaflow import alphaflow_score
        score_fn = alphaflow_score(device="cuda")
    else:
        raise ValueError(f"unknown backbone: {backbone}")

    result = sample_paths(
        start_pdb=str(start_pdb), end_pdb=str(end_pdb),
        score_fn=score_fn, n_paths=n_paths, n_steps=n_steps,
        optimiser_steps=optimiser_steps,
    )
    # Expected result fields per the repo's examples:
    #   result.coords:   (n_paths, n_steps, n_atoms, 3)
    #   result.ca_mask:  (n_atoms,) bool
    #   result.actions:  (n_paths,) final OM actions
    #   result.converged:(n_paths,) bool
    coords = np.asarray(result.coords, dtype=np.float32)
    ca_mask = np.asarray(result.ca_mask, dtype=bool)
    coords_ca = coords[:, :, ca_mask, :]
    print(f"[om-tps] coords {coords.shape}, coords_ca {coords_ca.shape}, "
          f"actions mean={float(result.actions.mean()):.3f}, "
          f"converged {int(result.converged.sum())}/{len(result.converged)}")

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=coords,
        coords_ca=coords_ca,
        seqres=np.array(getattr(result, "seqres", "")),
        chain_id=np.asarray(getattr(result, "chain_id", np.zeros(int(ca_mask.sum()))),
                            dtype=np.int64),
        path_action=np.asarray(result.actions, dtype=np.float32),
        converged=np.asarray(result.converged, dtype=bool),
        backbone=np.array(backbone),
    )
    return buf.getvalue()


@app.local_entrypoint()
def sample(start_pdb: str, end_pdb: str, name: str = "transition",
           n_paths: int = 16, n_steps: int = 100,
           backbone: str = "mace-off", optimiser_steps: int = 1000,
           out: str = ""):
    """Run OM-TPS between two structural endpoints.

    Example (v0.9 Notch1 W3d Phase 2 redo):
      modal run md/om_tps_modal.py::sample \\
          --start-pdb md/notch1_apo_md_state0.pdb \\
          --end-pdb md/notch1_af_state1.pdb \\
          --name notch1_md0_to_af1 \\
          --n-paths 16 --n-steps 60 --backbone alphaflow
    """
    start_bytes = Path(start_pdb).read_bytes()
    end_bytes = Path(end_pdb).read_bytes()
    print(f"[local] OM-TPS {n_paths} paths × {n_steps} steps "
          f"({backbone}), {name}: {start_pdb} → {end_pdb}")
    data = sample_remote.remote(start_bytes, end_bytes, name,
                                 n_paths=n_paths, n_steps=n_steps,
                                 backbone=backbone,
                                 optimiser_steps=optimiser_steps)
    out_path = Path(out) if out else Path(f"{name}_omtps.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print("[local] H3 ingest: flatten (n_paths, n_steps, ...) to "
          "(n_paths*n_steps, ...) and stratify by path_id; see "
          "md/multisource_h3.py for the loader pattern.")
