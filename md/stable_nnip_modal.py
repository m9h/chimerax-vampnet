"""StABlE (Stability-Aware Boltzmann Estimator) NNIP training on Modal.

Paper: Raja, Amin, Pedregosa, Krishnapriyan 2024-2025, "Stability-Aware
Training of Machine Learning Force Fields with Differentiable Boltzmann
Estimators" (TMLR 2025; arXiv:2402.13984). Code at
https://github.com/ASK-Berkeley/StABlE-Training (MIT). Presented at Starkly
Speaking 2024-04-08.

Why it earns a slot in the v0.9+ pipeline as the *PINN-flavoured* extra:

  StABlE is the cleanest published exemplar of a physics-informed *training*
  objective for ML force fields. The standard NNIP loss is forces+energies
  from QM reference data — purely data-driven, agnostic to whether the
  resulting potential gives a *stable* MD trajectory or recovers a target
  Boltzmann observable. StABlE alternates supervised epochs with on-the-fly
  MD rollouts, detects unstable regions via a differentiable Boltzmann
  estimator, and refines those regions against a *physical observable*
  (RDF, IR spectrum, an experimentally-measured equilibrium quantity). The
  Boltzmann estimator term is the physics-informed surrogate that makes
  StABlE a PINN-style approach to NNIP design.

  Direct role in this project:
    - Refine UMA (md/uma_modal.py) on the specific biomolecular systems we
      care about — Notch1 NRR, Hsp90 NTD, β2AR membrane — using each
      system's classical-MD reference RDF or radius-of-gyration distribution
      as the StABlE observable. The pretrained universal UMA may be
      pathologically unstable for some of these (especially the β2AR
      membrane case where lipid-protein clashes already broke classical
      prep at NVT); a StABlE-refined system-specific UMA could be the path
      out.
    - Compare to MACE-OFF + StABlE as a baseline; the paper benchmarks both.

  This adapter is *not* an inference adapter — it's a training-orchestration
  adapter. Output is a refined model checkpoint (uploaded to the Modal
  volume), not an ensemble or trajectory. Downstream consumption is via
  md/uma_modal.py (point its model_name at the StABlE-refined checkpoint).

  Base image:   modal.Image.debian_slim(python_version=3.11)
  Pip extras:   torch (cu124), StABlE-Training repo, the base NNIP the user
                picks (MACE / NequIP / UMA fine-tuning), ase, biopython.
  GPU pin:      A100-80GB for training; 4× H100 if the user enables
                multi-GPU via Modal's `gpu=modal.gpu.H100(count=4)`.
  Tested:       UNTESTED as of 2026-06-07 — scaffold only. First-run
                unknowns:
                  1. The repo's CLI/entry-point names — the README example
                     `python train.py --config configs/water.yml` suggests
                     a config-file pattern; the Modal wrapper passes a JSON
                     config through the file system.
                  2. Whether the StABlE pipeline supports HF-hosted base
                     models directly (UMA) or requires a state_dict file.

  modal run md/stable_nnip_modal.py::train \\
      --base-model mace-off-medium \\
      --reference-traj /vol/prepared/notch1_apo_v3/replica_0/traj.dcd \\
      --observable rdf --observable-target /vol/refs/notch1_apo_rdf.npy \\
      --epochs 200 --rollout-ps 50 \\
      --out-checkpoint notch1_stable_mace.pt
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-stable-nnip"
REPO_URL = "git+https://github.com/ASK-Berkeley/StABlE-Training.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"stable-training @ {REPO_URL}",  # TODO: verify package name
        "ase>=3.23",
        "mace-torch",     # MACE base model option
        "nequip",         # NequIP base model option
        "fairchem-core>=2.0",  # UMA base model option (shares image deps)
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "numpy<2",
        "einops",
        "matplotlib",  # diagnostic plots
    )
)

VOLUME_NAME = "chimerax-vampnet-md"
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_MOUNT = "/vol"


@app.function(gpu="A100-80GB", timeout=24 * 3600, volumes={VOL_MOUNT: vol})
def train_remote(base_model: str, ref_traj_bytes: bytes,
                  ref_topology_bytes: bytes, observable: str,
                  observable_target_bytes: bytes,
                  epochs: int = 200, rollout_ps: float = 50.0,
                  temperature_K: float = 310.0,
                  out_checkpoint: str = "stable_refined.pt") -> dict:
    """Run StABlE training and return diagnostics + uploaded checkpoint path.

    base_model: one of
      - "mace-off-{small,medium,large}"
      - "nequip-pretrained"           (loads the public NequIP checkpoint)
      - "uma-{small,medium}"          (UMA-LoRA fine-tune path)

    observable: one of
      - "rdf"                  : radial distribution function
      - "rg-distribution"      : radius-of-gyration histogram
      - "ca-ca-distance"       : selected CA-CA distance distribution
      - "secondary-structure"  : DSSP-assigned helix/sheet fraction
      All targets are 1D histograms / 1D scalars passed in via
      observable_target_bytes (numpy .npy).
    """
    import io
    import json
    import time

    import numpy as np
    import torch

    # First-invocation TODO: verify entry point in the repo's __main__.
    # The README example uses `python train.py --config X`; we may need to
    # import from the package directly, or shell out to the CLI.
    from stable_training import StABlETrainer  # noqa: F401 — verify

    workdir = Path("/tmp/stable_run")
    workdir.mkdir(parents=True, exist_ok=True)
    ref_path = workdir / "ref_traj.dcd"
    ref_path.write_bytes(ref_traj_bytes)
    top_path = workdir / "topology.pdb"
    top_path.write_bytes(ref_topology_bytes)
    target_path = workdir / "observable_target.npy"
    target_path.write_bytes(observable_target_bytes)

    print(f"[stable] base_model={base_model}, observable={observable}, "
          f"epochs={epochs}, rollout_ps={rollout_ps}, T={temperature_K} K")

    trainer = StABlETrainer(
        base_model=base_model,
        ref_trajectory=str(ref_path),
        ref_topology=str(top_path),
        observable=observable,
        observable_target=str(target_path),
        temperature_K=temperature_K,
        rollout_ps=rollout_ps,
        device="cuda",
    )
    t0 = time.time()
    history = trainer.train(n_epochs=epochs)
    elapsed = time.time() - t0
    print(f"[stable] training complete in {elapsed/60:.1f} min")

    # Save checkpoint to the persistent volume so md/uma_modal.py and
    # md/om_tps_modal.py can pick it up.
    out_dir = Path(VOL_MOUNT) / "stable_refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_checkpoint
    trainer.save(str(out_path))
    vol.commit()
    print(f"[stable] wrote refined checkpoint to {out_path}")

    # history is expected to contain: 'epoch', 'force_loss', 'energy_loss',
    # 'observable_loss', 'instability_rate' per epoch.
    summary = {
        "checkpoint_path": str(out_path),
        "elapsed_seconds": float(elapsed),
        "epochs": int(epochs),
        "final_force_loss": float(history["force_loss"][-1]),
        "final_observable_loss": float(history["observable_loss"][-1]),
        "final_instability_rate": float(history["instability_rate"][-1]),
        "base_model": base_model,
        "observable": observable,
    }
    print(f"[stable] summary: {json.dumps(summary, indent=2)}")
    return summary


@app.local_entrypoint()
def train(base_model: str = "mace-off-medium",
          reference_traj: str = "",
          reference_topology: str = "",
          observable: str = "rdf",
          observable_target: str = "",
          epochs: int = 200, rollout_ps: float = 50.0,
          temperature: float = 310.0,
          out_checkpoint: str = "stable_refined.pt"):
    """Refine an NNIP with StABlE on a system-specific observable.

    Workflow example for refining UMA on Notch1 NRR apo MD:
      modal run md/stable_nnip_modal.py::train \\
          --base-model uma-medium \\
          --reference-traj /data/.../notch1_apo_v3/replica_0/traj.dcd \\
          --reference-topology /data/.../notch1_apo_v3/equilibrated.pdb \\
          --observable rg-distribution \\
          --observable-target /data/.../notch1_apo_rg_hist.npy \\
          --epochs 300 --rollout-ps 100
    """
    ref_bytes = Path(reference_traj).read_bytes()
    top_bytes = Path(reference_topology).read_bytes()
    target_bytes = Path(observable_target).read_bytes()
    print(f"[local] StABlE train: base={base_model}, "
          f"obs={observable}, epochs={epochs}, rollout={rollout_ps} ps")
    summary = train_remote.remote(
        base_model, ref_bytes, top_bytes,
        observable, target_bytes,
        epochs=epochs, rollout_ps=rollout_ps,
        temperature_K=temperature,
        out_checkpoint=out_checkpoint,
    )
    print(f"[local] done: {summary}")
    print(f"[local] downstream: point md/uma_modal.py::produce at "
          f"checkpoint /vol/stable_refined/{out_checkpoint}")
