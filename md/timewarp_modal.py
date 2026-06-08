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
    # devel (not runtime) variant: DeepSpeed's setup.py unconditionally
    # probes /usr/local/cuda/bin/nvcc to read TORCH_CUDA_ARCH_LIST; the
    # runtime image has no nvcc and DS_BUILD_OPS=0 doesn't gate the probe.
    modal.Image.from_registry(
        "nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04", add_python="3.11"
    )
    .apt_install("bzip2", "ca-certificates", "curl", "git", "build-essential",
                 "libxrender1", "libxext6")  # mdtraj/openmm runtime libs
    # System-Python 3.11 deps: Modal's @app.function body runs under this
    # Python, NOT the conda env. huggingface_hub here is for the HF download
    # orchestration; the conda env has its own copy for the inline sampler.
    .pip_install("huggingface_hub")
    .run_commands(
        "mkdir -p /opt/conda/bin",
        "curl -sLo /tmp/mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest",
        "tar -xvjf /tmp/mm.tar.bz2 -C /opt/conda bin/micromamba",
        "rm /tmp/mm.tar.bz2",
    )
    .env({
        # NOTE: don't prepend /opt/conda/envs/tw/bin to PATH — the conda
        # env's Python 3.8 would shadow Modal's bootstrap Python 3.11
        # (Modal requires ≥3.10). The conda-env workload is invoked
        # explicitly via PYTHON = "/opt/conda/envs/tw/bin/python".
        # /usr/local/bin first so Modal's add_python="3.11" binary is
        # the canonical `python` in PATH (Modal needs to detect it).
        "PATH": "/usr/local/bin:/opt/conda/bin:/usr/bin:/bin",
        "MAMBA_ROOT_PREFIX": "/opt/conda",
        # Match upstream's CUDA-11.1 binding.
        "CONDA_OVERRIDE_CUDA": "11.1",
        # The timewarp repo has __init__.py at its root but no setup.py,
        # so we clone it as /opt/timewarp. PYTHONPATH includes both /opt
        # (so `import timewarp` resolves to the cloned package) AND
        # /opt/timewarp (so its sibling subpackages — `import simulation`,
        # `import utilities` — resolve to the in-repo directories).
        "PYTHONPATH": "/opt:/opt/timewarp",
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
        # DS_BUILD_OPS=0 / DS_SKIP_CUDA_CHECK=1: DeepSpeed's setup.py probes
        # nvcc to determine compute capabilities; the CUDA *runtime* image
        # (vs *devel*) has no nvcc. We only need DeepSpeed's import-time
        # presence (timewarp.utils.training_utils does `from deepspeed
        # import DeepSpeedEngine`), not its JIT-compiled CUDA ops.
        "DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 "
        "/opt/conda/envs/tw/bin/pip install --no-cache-dir "
        "'protobuf~=3.19.0' tensorboard einops omegaconf tqdm pyyaml docopt "
        "psutil cached-property multimethod gitpython monty lmdb "
        "'setuptools==59.5.0' deeptime "
        # deepspeed 0.10.0 (mid-2023) is contemporary with the timewarp paper
        # and avoids the 0.19.1 Muon optimizer bug that NameErrors on py3.8.
        # We don't actually USE deepspeed (it's a training-only import); the
        # version pin is just to satisfy the `from deepspeed import
        # DeepSpeedEngine` in timewarp.utils.training_utils.
        "'deepspeed==0.10.0' "
        f"git+https://github.com/noegroup/bgflow.git "
        "huggingface_hub",
    )
    .run_commands(
        # Timewarp itself has no setup.py — clone as /opt/timewarp and
        # rely on PYTHONPATH=/opt (set in .env above) for `import timewarp`.
        f"git clone https://github.com/microsoft/timewarp.git /opt/timewarp && "
        f"cd /opt/timewarp && git checkout {TIMEWARP_SHA}",
    )
    # `utilities` is a Microsoft-internal sibling pkg that timewarp imports
    # extensively (utilities.logger, utilities.model_utils, utilities.common,
    # utilities.delayed_reporter, ...). Not published anywhere. Rather than
    # chase the per-submodule whack-a-mole, the inline sampler script
    # installs a sys.meta_path finder at boot that returns dynamic stubs
    # for any `utilities.*` import — see _SAMPLER_SCRIPT below.
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
def sample_remote(protein: str = "ad2", n_samples: int = 200,
                   n_proposal_steps: int = 100,
                   model_size: str = "2aa") -> bytes:
    """Rev 9: HF-download checkpoint + AD-3 data, run MH sampling via the
    timewarp internal API (load_model + sample_from_trajectory), pack
    coords into the project npz schema.

    model_size: "2aa" (alanine dipeptide, 426 MB checkpoint) or
                "4aa" (tetrapeptides, 4.76 GB).
    """
    import io
    import subprocess
    import sys
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir = Path(VOL_MOUNT) / "timewarp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch the checkpoint + config (one .pt + one .yaml). hf_hub_download
    #    caches per-file; safe to call every run.
    print(f"[timewarp] downloading {model_size}_best_model.pt + config")
    ckpt = hf_hub_download("microsoft/timewarp", f"{model_size}_best_model.pt",
                            repo_type="dataset", cache_dir=str(cache_dir))
    cfg = hf_hub_download("microsoft/timewarp", f"{model_size}_config.yaml",
                           repo_type="dataset", cache_dir=str(cache_dir))
    # 2) Fetch the AD-3 data folder. Layout: AD-3/{test,train}/{pdb_name}
    #    -traj-state0.pdb + -traj-arrays.npz. For the 2aa model the only
    #    AD-3 entry is `ad2` (alanine dipeptide). 4AA-large/test has the
    #    held-out tetrapeptides.
    data_subdir = "AD-3" if model_size == "2aa" else "4AA-large"
    split = "test"  # always evaluate against the held-out split
    print(f"[timewarp] downloading {data_subdir}/{split} data")
    data_root = snapshot_download("microsoft/timewarp", repo_type="dataset",
                                   cache_dir=str(cache_dir),
                                   allow_patterns=[f"{data_subdir}/{split}/*"])
    data_dir = Path(data_root) / data_subdir / split
    print(f"[timewarp] checkpoint={ckpt}\n           data_dir={data_dir}")

    # 3) Drop a small inline Python script into /tmp and run it under the
    #    conda env's Python 3.8. evaluate.py / sample.py only emit figures;
    #    we use the underlying API (load_model + sample_from_trajectory)
    #    to capture the raw trajectory.
    script = Path("/tmp/_timewarp_sample.py")
    script.write_text(_SAMPLER_SCRIPT)
    out_npz = Path("/tmp/timewarp_out.npz")

    cmd = [PYTHON, str(script),
            "--savefile", ckpt,
            "--config", cfg,
            "--data-dir", str(data_dir),
            "--protein", protein,
            "--num-samples", str(n_samples),
            "--num-proposal-steps", str(n_proposal_steps),
            "--out-npz", str(out_npz)]
    print(f"[timewarp] {' '.join(cmd)}")
    sys.stdout.flush()
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(
            f"timewarp sampling failed (exit {r.returncode}); "
            f"stderr tail: {r.stderr[-2000:]}")

    if not out_npz.exists():
        raise RuntimeError("sampler script reported success but no npz written")
    vol.commit()
    return out_npz.read_bytes()


# The inline sampling script runs under the conda env's Python 3.8 + torch
# 1.9 + the cloned timewarp repo on PYTHONPATH. Lives here as a string so
# the adapter file is self-contained — no auxiliary file to track / commit /
# version-skew with the Modal image.
_SAMPLER_SCRIPT = '''
"""Inline timewarp MCMC sampler — see md/timewarp_modal.py for context."""
import argparse, sys, time, types
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import torch


# ---- missing-pkg stubs ---------------------------------------------------
# timewarp imports several sibling pkgs that are MS-internal or heavy
# (`utilities.*`, `visualise.*`, `pymol2`). The inference path doesn\\'t
# actually call them — they\\'re module-level imports only. A sys.meta_path
# finder intercepts ALL of these and returns minimal real implementations
# for known inference-time call paths, stub classes otherwise.
_STUB_PREFIXES = ("utilities", "visualise", "pymol2", "lmdb")


def _make_stub_class(name):
    """Return a no-op class usable in type annotations (Optional[X]), calls,
    as a base class, AND with generic subscripting (Foo[X], for code that
    treats utilities.* stubs as typing.Generic subclasses)."""
    return type(name, (), {
        "__init__": lambda self, *a, **kw: None,
        "__call__": lambda self, *a, **kw: None,
        "__getattr__": lambda self, n: (lambda *a, **kw: None),
        # Foo[Bar] returns Foo so the stub can serve as a generic base.
        "__class_getitem__": classmethod(lambda cls, item: cls),
    })


class _StubFinder:
    def find_module(self, fullname, path=None):
        root = fullname.split(".", 1)[0]
        if root in _STUB_PREFIXES:
            return self
        return None
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []  # mark as package
        mod.__loader__ = self
        # Known-needed real implementations:
        if fullname == "utilities.model_utils":
            mod.unflatten_state_dict = lambda sd, *a, **kw: sd
            # Real load_model — timewarp's load_model delegates to here.
            # The checkpoint is a torch.save dict; the model_constructor
            # callback builds the architecture from data["training_config"].
            def _load_model(path, model_constructor):
                data = torch.load(path, map_location="cpu")
                model = model_constructor(data)
                sd_key = ("model_state_dict" if "model_state_dict" in data
                          else "module" if "module" in data
                          else "state_dict" if "state_dict" in data else None)
                if sd_key:
                    model.load_state_dict(data[sd_key], strict=False)
                return model
            mod.load_model = _load_model
            mod.load_checkpoint_in_subdir = lambda p: torch.load(
                str(p), map_location="cpu")
        elif fullname == "utilities.common":
            mod.StrPath = str
        elif fullname == "utilities.logger":
            mod.TrainingLogger = _make_stub_class("TrainingLogger")
        elif fullname == "utilities.cache":
            class NullaryClosure:
                """Holds a 0-arg deferred call; invoke via __call__."""
                __slots__ = ("fn", "args", "kwargs")
                def __init__(self, fn, args, kwargs):
                    self.fn, self.args, self.kwargs = fn, args, kwargs
                @classmethod
                def create(cls, fn, *args, **kwargs):
                    return cls(fn, args, kwargs)
                def __call__(self):
                    return self.fn(*self.args, **self.kwargs)
                invoke = __call__  # some Cache impls call .invoke()
            class Cache:
                """No-caching identity: every load_or_produce invokes the closure."""
                def __init__(self, *a, **kw): pass
                def load_or_produce(self, closure, *a, **kw):
                    return closure()
                def empty_like(self):
                    return type(self)()
            mod.NullaryClosure = NullaryClosure
            mod.Cache = Cache
        # Catch-all: any unknown attribute (incl. CamelCase classes used in
        # type annotations) returns a real stub class, not a sub-module —
        # so `from utilities.logger import TensorBoardLogger` works without
        # the finder being re-triggered as a submodule lookup.
        mod.__getattr__ = lambda n, _name=fullname: _make_stub_class(  # type: ignore[attr-defined]
            f"{_name}.{n}"
        )
        sys.modules[fullname] = mod
        return mod

sys.meta_path.insert(0, _StubFinder())
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savefile", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--num-proposal-steps", type=int, default=100)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    print(f"[script] loading model from {args.savefile}", flush=True)
    from timewarp.utils.training_utils import load_model
    from timewarp.utils.evaluation_utils import sample_with_model
    from timewarp.datasets import RawMolDynDataset
    from timewarp.dataloader import moldyn_dense_collate_fn
    from timewarp.utils.openmm import OpenmmPotentialEnergyTorch
    from timewarp.utils.openmm.openmm_bridge import (
        device_to_platform_and_properties)
    from simulation.md import (
        get_simulation_environment, get_simulation_environment_integrator)
    from openmm import unit as u

    device = torch.device("cuda")
    # load_model returns a LossWrapper(module=actual_model) for consistent
    # return type — unwrap before sampling. unwrap_loss_wrapper handles
    # both wrapped + already-bare cases.
    from timewarp.losses import unwrap_loss_wrapper
    model = unwrap_loss_wrapper(load_model(path=args.savefile)).to(device).eval()
    print(f"[script] model loaded: {type(model).__name__}", flush=True)

    # The dataset API expects a step_width (number of MD steps between
    # the conditioning frame and the predicted frame). For the released
    # 2aa model the convention is 1000 (1 ns at 1 fs); confirm from cfg
    # if a future model uses a different stride.
    step_width = 1000
    dataset = RawMolDynDataset(data_dir=args.data_dir, step_width=step_width)
    it = dataset.make_iterator([args.protein])
    batch = moldyn_dense_collate_fn([next(it)])
    # DenseMolDynBatch is a (possibly frozen) dataclass — use fields() +
    # dataclasses.replace to move tensors to CUDA without assuming dict.
    import dataclasses
    _gpu_fields = {f.name: (getattr(batch, f.name).to(device)
                            if hasattr(getattr(batch, f.name), "to")
                            else getattr(batch, f.name))
                    for f in dataclasses.fields(batch)}
    batch = dataclasses.replace(batch, **_gpu_fields)

    # Build the OpenMM energy + masses required by sample_with_model
    # for MH acceptance ratios. evaluate.py uses parameters="alanine-
    # dipeptide" for all 2aa runs (it\\'s the force-field name, not the
    # protein name).
    parameters = "alanine-dipeptide"
    state0_path = next(Path(args.data_dir).glob(f"{args.protein}*state0.pdb"))
    simulation = get_simulation_environment(str(state0_path), parameters)
    integrator = get_simulation_environment_integrator(parameters)
    system = simulation.system
    platform, platform_properties = device_to_platform_and_properties(
        device, num_threads=1)
    openmm_potential_energy_torch = OpenmmPotentialEnergyTorch(
        system, integrator,
        platform_name=platform,
        platform_properties=platform_properties,
    )
    num_atoms = system.getNumParticles()
    masses = torch.tensor(
        [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(num_atoms)]
    ).to(device)
    print(f"[script] openmm energy + {num_atoms} masses ready", flush=True)

    print(f"[script] running MH: {args.num_samples} samples × "
          f"{args.num_proposal_steps} proposals/iter", flush=True)
    t0 = time.time()
    with torch.no_grad():
        sampled_coords, _, _, chain_stats = sample_with_model(
            batch, model, device,
            openmm_potential_energy_torch, masses,
            args.num_samples,
            True,  # mh
            random_velocs=False, resample_velocs=False,
            initialize_randomly=False, sim=None,
            openmm_on_current=False, openmm_on_proposal=False,
            num_openmm_steps=0,
            num_proposal_steps=args.num_proposal_steps,
            adaptive_parallelism=False,
        )
    elapsed = time.time() - t0
    print(f"[script] sampling done in {elapsed:.1f} s", flush=True)

    coords = np.asarray(sampled_coords, dtype=np.float32)
    accept = float(np.mean(np.asarray(chain_stats.acceptance)))
    print(f"[script] coords shape {coords.shape}, accept_rate {accept:.3f}",
          flush=True)

    # CA mask via topology — for ala-dipeptide there are 22 atoms with
    # CA at index 8 (ACE-ALA-NME canonical ordering). For safety, take
    # the topology's atom list.
    from openmm.app import PDBFile
    pdb_path = next(Path(args.data_dir).glob(f"{args.protein}*state0.pdb"))
    pdb = PDBFile(str(pdb_path))
    atom_names = [a.name for a in pdb.topology.atoms()]
    ca_idx = [i for i, n in enumerate(atom_names) if n == "CA"]
    coords_ca = coords[:, ca_idx, :] if ca_idx else coords
    # Sequence from residue list
    THREE_TO_ONE = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G",
                    "HIS":"H","ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N",
                    "PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V",
                    "TRP":"W","TYR":"Y","ACE":"","NME":""}
    seqres = "".join(THREE_TO_ONE.get(r.name, "X") for r in pdb.topology.residues())

    np.savez_compressed(args.out_npz,
        coords=coords, coords_ca=coords_ca,
        seqres=np.array(seqres),
        chain_id=np.zeros(len(ca_idx) if ca_idx else coords.shape[1], dtype=np.int64),
        plddt=np.full(coords.shape[0], np.nan, dtype=np.float32),
        iptm=np.full(coords.shape[0], np.nan, dtype=np.float32),
        accept_rate=np.array(accept, dtype=np.float32),
        elapsed_seconds=np.array(elapsed, dtype=np.float32),
    )
    print(f"[script] wrote {args.out_npz}", flush=True)


if __name__ == "__main__":
    main()
'''


@app.local_entrypoint()
def sample(protein: str = "ad2", n_samples: int = 200,
           n_proposal_steps: int = 100, model_size: str = "2aa",
           out: str = ""):
    """Run Timewarp MH sampling via the internal API (no figure-only
    evaluate.py shell-out). Pulls the HF checkpoint + AD-3 data on first
    invocation; cached on the Modal volume thereafter."""
    print(f"[local] timewarp sample: {protein} ({model_size}), "
          f"{n_samples} samples × {n_proposal_steps} proposals")
    data = sample_remote.remote(protein, n_samples, n_proposal_steps,
                                 model_size=model_size)
    out_path = Path(out) if out else Path(f"{protein}_timewarp.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
