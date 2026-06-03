"""AlphaFlow / ESMFlow ensemble generation on Modal — v0.5 micromamba rewrite.

Self-contained environment recipe — each Modal adapter in md/ builds
its own image rather than sharing a base, so dep collisions stay
isolated to the tool that needs them.

  Base image:   nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
                (CUDA-11 era to match AlphaFlow's torch==1.12.1+cu113 pin)
  Env manager:  micromamba creating /opt/conda/envs/af (python=3.9)
  Pip recipe:   verbatim from upstream alphaflow README, see below
  Source repo:  github.com/bjing2016/alphaflow (cloned to /opt/alphaflow)
  Checkpoint:   bjing-mit/alphaflow → params/esmflow_md_base_202402.pt
                (downloaded at runtime via huggingface_hub)
  GPU pin:      A100-80GB
  Tested:       2026-06-03 — smoke in progress (v0.5 first attempt
                  after 9 failed NGC iterations in v0.4)

Environment recipe (verbatim from upstream alphaflow README):
  - python=3.9
  - numpy==1.21.2 pandas==1.5.3
  - torch==1.12.1+cu113 (legacy PyTorch wheel index)
  - biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0
    scipy==1.7.1 absl-py einops
  - pytorch_lightning==2.0.4 fair-esm mdtraj==1.9.9 wandb
  - openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037

The v0.4 attempt (9 iterations) failed on NGC PyTorch's pre-bundled
deps colliding with AlphaFlow's 2024-pinned conda env. v0.5 swaps to
a bare CUDA-11.8 base + micromamba reproducing the upstream conda
env verbatim, isolated from any other tool's deps.

Usage:
  modal run md/alphaflow_modal.py::sample --sequence "ACELPECQ..." \\
      --name notch1_NEC --n 5 --out notch1_NEC_af_smoke.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-alphaflow"

# Self-contained image: CUDA 11.8 base (contemporaneous with AlphaFlow's
# torch 1.12.1+cu113 pin) + micromamba reproducing the upstream conda
# env spec verbatim. No NGC PyTorch in this stack.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "wget", "curl", "bzip2",
                  "ca-certificates", "zlib1g-dev")
    .run_commands(
        # Install micromamba (same pattern as md/modal_md.py).
        "mkdir -p /opt/conda/bin",
        "wget -qO /tmp/mm.tar.bz2 "
        "https://micro.mamba.pm/api/micromamba/linux-64/latest",
        "tar -xvjf /tmp/mm.tar.bz2 -C /opt/conda bin/micromamba",
        "rm /tmp/mm.tar.bz2",
    )
    .env({
        # System Python (3.11 via add_python) MUST come before the af
        # env so Modal's runtime can import the modal package. The af
        # env's python is invoked explicitly via /opt/conda/envs/af/bin/python
        # in subprocesses — it does not need to be on PATH.
        "PATH": "/usr/local/bin:/opt/conda/bin:/usr/local/cuda/bin:"
                "/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
        "MAMBA_ROOT_PREFIX": "/opt/conda",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    })
    .run_commands(
        # Create the AlphaFlow env per the upstream README. mdtraj is
        # installed via pip (not conda-forge) so it links against
        # numpy 1.21.2; the conda-forge mdtraj wheel had been
        # compiled against newer numpy ABI which broke at runtime.
        # zlib1g-dev (apt-installed above) lets the pip mdtraj wheel
        # build its xtc/trr/tng extensions.
        "/opt/conda/bin/micromamba create -y -n af -c conda-forge "
        "python=3.9 pip && /opt/conda/bin/micromamba clean -a -y",
        # AlphaFlow numpy/pandas first.
        "/opt/conda/envs/af/bin/pip install --no-cache-dir "
        "'numpy==1.21.2' 'pandas==1.5.3'",
        # torch pin: install via the legacy +cu113 wheel.
        "/opt/conda/envs/af/bin/pip install --no-cache-dir "
        "torch==1.12.1+cu113 "
        "--extra-index-url https://download.pytorch.org/whl/cu113",
        # Remaining alphaflow deps. Use --no-deps on pytorch_lightning
        # to keep the resolver from upgrading torch to 2.x.
        "/opt/conda/envs/af/bin/pip install --no-cache-dir "
        "'biopython==1.79' 'dm-tree==0.1.6' 'modelcif==0.7' "
        "'ml-collections==0.1.0' 'scipy==1.7.1' absl-py einops",
        # pytorch_lightning + fair-esm + wandb installed individually
        # with --no-deps for pytorch_lightning so torch stays at 1.12.1.
        "/opt/conda/envs/af/bin/pip install --no-cache-dir --no-deps "
        "'pytorch_lightning==2.0.4' 'lightning-utilities>=0.7.0' "
        "torchmetrics fsspec 'PyYAML>=5.4' tqdm packaging "
        "'typing_extensions>=4.0.0'",
        "/opt/conda/envs/af/bin/pip install --no-cache-dir fair-esm wandb "
        "'mdtraj==1.9.9'",
        # OpenFold pinned commit + CUDA-kernel build. With CUDA 11.8
        # available via the base image, the build should succeed; the
        # fallback --no-deps install keeps the package importable even
        # if its build step fails (alphaflow's predict.py imports it).
        "/opt/conda/envs/af/bin/pip install --no-cache-dir "
        "'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037' "
        "|| /opt/conda/envs/af/bin/pip install --no-cache-dir --no-deps "
        "'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037'",
        # Clone alphaflow.
        "cd /opt && git clone https://github.com/bjing2016/alphaflow.git",
        # Patch predict.py: upstream HEAD added a torch.load(
        # weights_only=False) call that requires torch >=1.13, but
        # we're pinned to 1.12.1+cu113 (which AlphaFlow's own README
        # specifies). Drop the kwarg — the pre-2.5-era default
        # behaviour is what we want.
        "sed -i 's/, weights_only=False//' /opt/alphaflow/predict.py",
        # Useful pip-installable but not in their conda env (we use it
        # for npz packaging; openfold/alphaflow themselves don't need it).
        "/opt/conda/envs/af/bin/pip install --no-cache-dir huggingface_hub",
        # The Modal runtime runs sample_remote() in the system Python
        # 3.11 (added via add_python=3.11), which needs its own numpy
        # and mdtraj for the post-process (load PDBs from the af
        # subprocess output + pack into npz). Lightweight installs.
        "/usr/local/bin/pip install --no-cache-dir numpy mdtraj",
    )
)

VOLUME_NAME = "chimerax-vampnet-md"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A100-80GB", timeout=2 * 3600,
               volumes={"/vol": vol},
               retries=0)
def sample_remote(sequence: str, name: str = "query", n_samples: int = 200,
                   checkpoint: str = "esmflow_md_base_202402",
                   vol_output: str | None = None) -> bytes:
    """Run AlphaFlow / ESMFlow-MD inference. Returns packed ensemble (.npz bytes)
    AND writes them to the shared volume at /vol/<vol_output> if given,
    so a re-launched local client can recover the result even after a
    local-side SIGTERM."""
    import io
    import os
    import subprocess
    import sys
    import tempfile

    import numpy as np

    # All subprocesses must use the env's python.
    py = "/opt/conda/envs/af/bin/python"

    # Download the checkpoint via the env's huggingface_hub.
    print(f"[af] downloading checkpoint params/{checkpoint}.pt from bjing-mit/alphaflow")
    download_script = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id='bjing-mit/alphaflow', "
        f"filename='params/{checkpoint}.pt', cache_dir='/root/.cache/huggingface'))"
    )
    r = subprocess.run([py, "-c", download_script],
                       check=True, capture_output=True, text=True)
    ckpt_path = r.stdout.strip().splitlines()[-1]
    print(f"[af] checkpoint at {ckpt_path}")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        csv_path = tmp / "input.csv"
        csv_path.write_text(f"name,seqres\n{name},{sequence}\n")
        out_dir = tmp / "out"
        out_dir.mkdir()
        cmd = [
            py, "predict.py",
            "--mode", "esmfold",
            "--input_csv", str(csv_path),
            "--samples", str(n_samples),
            "--weights", ckpt_path,
            "--outpdb", str(out_dir),
        ]
        print(f"[af] {' '.join(cmd)}")
        sys.stdout.flush()
        subprocess.run(cmd, check=True, cwd="/opt/alphaflow")

        # AlphaFlow writes one multi-MODEL PDB per protein name.
        import mdtraj as md
        sample_paths = sorted(out_dir.glob("*.pdb"))
        if not sample_paths:
            raise RuntimeError("AlphaFlow produced no PDB outputs")
        pdb_path = sample_paths[0]
        traj = md.load(str(pdb_path))
        coords_all = (traj.xyz * 10.0).astype(np.float32)
        ca_indices = [a.index for a in traj.topology.atoms if a.name == "CA"]
        coords_ca = coords_all[:, ca_indices, :]
        print(f"[af] all-atom {coords_all.shape}, CA-only {coords_ca.shape}")

        buf = io.BytesIO()
        np.savez_compressed(buf, coords=coords_all, coords_ca=coords_ca,
                            seqres=np.array(sequence))
        data = buf.getvalue()

        # Also write to the shared Modal volume if the caller gave a
        # destination name there. This survives a local-side SIGTERM:
        # the volume entry can be pulled by a fresh `modal volume get`.
        if vol_output:
            vol_path = Path("/vol") / vol_output
            vol_path.parent.mkdir(parents=True, exist_ok=True)
            vol_path.write_bytes(data)
            try:
                vol.commit()
            except Exception as exc:
                print(f"[af] volume commit failed: {exc}")
            print(f"[af] wrote {vol_path} ({len(data)/(1<<20):.1f} MB) to volume")

        return data


@app.local_entrypoint()
def sample(sequence: str, name: str = "query", n: int = 200,
           out: str = "alphaflow_ensemble.npz",
           checkpoint: str = "esmflow_md_base_202402"):
    """Generate an ESMFlow-MD ensemble and save to a local .npz."""
    print(f"[local] generating {n} samples for seq len {len(sequence)} "
          f"(checkpoint {checkpoint})")
    vol_dest = f"alphaflow_outputs/{Path(out).name}"
    data = sample_remote.remote(sequence, name, n, checkpoint, vol_dest)
    Path(out).write_bytes(data)
    print(f"[local] wrote {out} ({len(data)/(1<<20):.1f} MB)")
