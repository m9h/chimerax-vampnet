"""Timewarp (time-coarsened normalising-flow MCMC proposal) on Modal.

Paper: Klein, Foong, Fjelde, Mlodozeniec, Brockschmidt, Nowozin, Noé, Tomioka
2023, "Timewarp: Transferable Acceleration of Molecular Dynamics by Learning
Time-Coarsened Dynamics" (NeurIPS 2023; arXiv:2302.01170). Code at
https://github.com/microsoft/timewarp (MIT). HF data + checkpoints at
https://huggingface.co/datasets/microsoft/timewarp. Presented at
Starkly Speaking 2023-10-25.

Why it earns a slot in the v0.9+ multisource pipeline:

  Timewarp learns a normalising flow that proposes moves of 10^5-10^6 fs
  per step, used as an MCMC proposal targeting the exact Boltzmann
  distribution. Unlike BioEmu/Prose/AlphaFlow, Timewarp is *trajectory-
  aware* — learns time-dependence, not just an equilibrium emulator. Only
  one of our generative candidates that can in principle estimate
  dynamical quantities (transition rates, residence times).

  Applicability: released checkpoint trained on 2-4-residue peptides per
  the paper, demonstrates transferability *within* that range. v0.9 use
  case is dipeptide/tetrapeptide benchmarking, not direct application to
  Notch1/Hsp90/β2AR. UniSim (transferable-samplers/UniSim) is the same
  group's larger-system successor.

Image strategy (rev 2 after the rev-1 pip-install scaffold failure):

  Timewarp's `timewarp-environment.yml` pins a 2021-era stack:
    python==3.8.10, pytorch==1.9.0+cu111, openmm==7.7, ase==3.22.1,
    mdtraj==1.9.7, pdbfixer==1.8.1, biopython==1.79, bgflow (git), deeptime.
  NGC PyTorch 26.05 ships torch 2.8 — wrong torch entirely. So we use a
  CUDA 11.1 base + micromamba (matches the project pattern from
  md/modal_md.py) + minimal conda env (drop the Azure ML, pymol, psi4,
  ambertools, nglview, dev tools from upstream — pure inference doesn't
  need them).

  Base image:   nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04 + micromamba
  Pip extras:   torch==1.9.0+cu111, biopython, mdtraj, openmm, pdbfixer,
                ase, bgflow (git), deeptime, einops, omegaconf, tqdm.
  Repo:         microsoft/timewarp @ pinned SHA (2024-08-21 main).
  Checkpoint:   microsoft/timewarp dataset on HF — cached in the Modal
                volume to avoid repeat downloads.
  GPU pin:      A100-40GB (sub-100M params).
  Reproducibility note: every git install is pinned to a SHA per the
  project's adapter-perf-reproducibility memory.

Status:
  Rev 1 (2026-06-08): pip-install-as-package scaffold — image-build failed
                       (upstream isn't pip-installable).
  Rev 2 (2026-06-08): this — micromamba + minimal conda env + git clone.
                       UNTESTED; first smoke verifies image build only.

  modal run md/timewarp_modal.py::smoke
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-timewarp"
TIMEWARP_SHA = "211fed1d4e345c04929e95dfcca21a5facbb2357"  # main as of 2024-08-21
HF_DATASET = "microsoft/timewarp"

image = (
    # add_python="3.11" is for Modal's own bootstrap layer; the actual
    # Timewarp workload runs under the micromamba env's Python 3.8.10
    # (which conda creates from the conda-forge channel — independent of
    # what Modal ships). Modal v2026 dropped standalone-3.8 support.
    modal.Image.from_registry(
        "nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04", add_python="3.11"
    )
    .apt_install("bzip2", "ca-certificates", "curl", "git", "build-essential",
                 "libxrender1", "libxext6")  # mdtraj/openmm runtime libs
    .run_commands(
        "mkdir -p /opt/conda/bin",
        "curl -sLo /tmp/mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest",
        "tar -xvjf /tmp/mm.tar.bz2 -C /opt/conda bin/micromamba",
        "rm /tmp/mm.tar.bz2",
    )
    .env({
        "PATH": "/opt/conda/bin:/opt/conda/envs/tw/bin:/usr/bin:/bin",
        "MAMBA_ROOT_PREFIX": "/opt/conda",
        # Match upstream's CUDA-11.1 binding.
        "CONDA_OVERRIDE_CUDA": "11.1",
        # The timewarp repo has __init__.py at its root but no setup.py,
        # so we clone it as /opt/timewarp and put /opt on PYTHONPATH —
        # `import timewarp` then resolves to the cloned repo.
        "PYTHONPATH": "/opt",
    })
    .run_commands(
        # Minimal conda env: only what evaluate.py actually needs.
        # Drop azureml/pymol/psi4/ambertools/nglview/dev-tools from upstream
        # environment.yml — those are training-time + Azure infra cruft.
        "/opt/conda/bin/micromamba create -y -n tw -c pytorch -c conda-forge "
        "python=3.8.10 "
        "'pytorch=1.9.0=py3.8_cuda11.1_cudnn8.0.5_0' "
        "cudatoolkit=11.1 "
        # MKL pinning: pytorch 1.9 binaries reference iJIT_NotifyEvent from
        # libittnotify (old Intel ITT layer); MKL 2022+ removed it. Pin to
        # the 2021.4.0 era that matches pytorch 1.9's binary linkage.
        "'mkl=2021.4.0' "
        "openmm=7.7 mdtraj=1.9.7 pdbfixer=1.8.1 biopython=1.79 "
        "ase=3.22.1 numpy=1.21 "
        "&& /opt/conda/bin/micromamba clean -a -y",
    )
    .run_commands(
        # Pip-installable extras that aren't on conda-forge with the right pins.
        "/opt/conda/envs/tw/bin/pip install --no-cache-dir "
        "'protobuf~=3.19.0' tensorboard einops omegaconf tqdm pyyaml docopt "
        "psutil cached-property multimethod gitpython monty "
        "'setuptools==59.5.0' deeptime "
        f"git+https://github.com/noegroup/bgflow.git "
        "huggingface_hub",
    )
    .run_commands(
        # Timewarp itself has no setup.py — clone as /opt/timewarp and
        # rely on PYTHONPATH=/opt (set in .env above) for `import timewarp`.
        f"git clone https://github.com/microsoft/timewarp.git /opt/timewarp && "
        f"cd /opt/timewarp && git checkout {TIMEWARP_SHA}",
    )
    .run_commands(
        # Sanity check the install at image-build time (fail loudly here, not
        # at runtime). Confirms timewarp + bgflow + torch are all importable.
        "/opt/conda/envs/tw/bin/python -c "
        "'import torch; print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available()); "
        "import timewarp; print(\"timewarp ok\"); "
        "import bgflow; print(\"bgflow ok\"); "
        "import openmm; print(\"openmm\", openmm.version.full_version)'",
    )
)

VOLUME_NAME = "chimerax-vampnet-md"
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_MOUNT = "/vol"

# Reuse the project-wide HF token for dataset+checkpoint downloads (the
# microsoft/timewarp dataset is currently public, but the pattern + token
# costs nothing extra and future-proofs if it ever gets gated).
HF_SECRET = modal.Secret.from_name("huggingface-secret")

PYTHON = "/opt/conda/envs/tw/bin/python"


@app.function(gpu="A100-40GB", timeout=600, volumes={VOL_MOUNT: vol},
              secrets=[HF_SECRET])
def smoke_remote() -> dict:
    """Pipeline smoke: confirms image build, conda env activation, and
    that timewarp + torch + bgflow + openmm all import + CUDA is visible.
    No model load yet — that's rev 3."""
    import subprocess
    import sys
    print(f"[timewarp-smoke] python={PYTHON}")
    r = subprocess.run([PYTHON, "-c",
        "import torch, timewarp, bgflow, openmm, mdtraj, ase, deeptime; "
        "print('torch', torch.__version__); "
        "print('cuda available:', torch.cuda.is_available()); "
        "print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        "print('timewarp', getattr(timewarp, '__version__', '?')); "
        "print('bgflow', getattr(bgflow, '__version__', '?')); "
        "print('openmm', openmm.version.full_version); "
        "print('mdtraj', mdtraj.version.version); "
        "print('deeptime', deeptime.__version__);"
    ], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr, file=sys.stderr)
    return {
        "ok": r.returncode == 0,
        "stdout": r.stdout,
        "stderr": r.stderr[-500:] if r.stderr else "",
        "timewarp_sha": TIMEWARP_SHA,
    }


@app.local_entrypoint()
def smoke():
    """Smoke-test the Timewarp image only (no model load yet)."""
    print("[local] launching Timewarp image smoke")
    result = smoke_remote.remote()
    print(f"[local] smoke result: {result}")


@app.function(gpu="A100-40GB", timeout=4 * 3600, volumes={VOL_MOUNT: vol},
              secrets=[HF_SECRET])
def sample_remote(protein: str = "alanine-dipeptide", n_samples: int = 1000,
                   n_proposal_steps: int = 1000) -> bytes:
    """[Rev 3 TODO — currently raises NotImplementedError]

    Planned: download microsoft/timewarp/AD-3 dataset + checkpoint via
    huggingface_hub, run evaluate.py with --mh, parse outputs to .npz
    in the shared {coords, coords_ca, seqres, chain_id, plddt, iptm,
    accept_rate, autocorr_time, temperature_K} schema.
    """
    raise NotImplementedError(
        "sample_remote pending: rev 3 (after smoke validates image build). "
        "Need to wire HF download of AD-3 + checkpoint, then shell out to "
        "evaluate.py from the cloned timewarp repo."
    )


@app.local_entrypoint()
def sample(protein: str = "alanine-dipeptide", n_samples: int = 1000,
           n_proposal_steps: int = 1000, out: str = ""):
    """[Rev 3 TODO] Placeholder; use `smoke` for now."""
    print(f"[local] sample for {protein} not yet implemented (rev 3)")
    data = sample_remote.remote(protein, n_samples, n_proposal_steps)
    out_path = Path(out) if out else Path(f"{protein}_timewarp.npz")
    out_path.write_bytes(data)
