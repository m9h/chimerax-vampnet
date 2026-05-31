"""Modal-hosted OpenMM MD for the chimerax-vampnet Tier-1/Tier-2 trajectories.

Why Modal: a single GB10 takes 4 days for 1 us of chignolin and 5+ days
for 3x100 ns of Notch1 NRR. On Modal we fan out N replicas onto H100s
or A100s in parallel and land everything inside ~16 hr wall clock.

Layout:
  modal Volume "chimerax-vampnet-md" holds:
    inputs/<system>.pdb              (uploaded once via `modal volume put`)
    prepared/<system>/system.xml     (output of prep)
    prepared/<system>/state.xml
    prepared/<system>/equilibrated.pdb
    prepared/<system>/replica_<r>/   (output of produce)

Entrypoints:
  modal run modal_md.py::prep --system chignolin --pdb /local/5awl.pdb
  modal run modal_md.py::produce --system chignolin --replica 0 --ns 250
  modal run modal_md.py::chignolin_full   # fan out the chignolin Tier-1 run
  modal run modal_md.py::notch1_apo       # fan out 3 replicas of Notch1 NRR apo
  modal run modal_md.py::notch1_holo      # fan out 3 replicas of Notch1 NRR holo
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-md"
VOLUME_NAME = "chimerax-vampnet-md"

# OpenMM CUDA image (x86_64 — Modal's default platform).
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-runtime-ubuntu24.04", add_python="3.13")
    .apt_install("bzip2", "ca-certificates", "curl")
    .run_commands(
        # Install micromamba.
        "mkdir -p /opt/conda/bin",
        "curl -sLo /tmp/mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest",
        "tar -xvjf /tmp/mm.tar.bz2 -C /opt/conda bin/micromamba",
        "rm /tmp/mm.tar.bz2",
        "/opt/conda/bin/micromamba --version",
    )
    .env({"PATH": "/opt/conda/bin:/opt/conda/envs/md/bin:/usr/bin",
          "CONDA_OVERRIDE_CUDA": "12.9",
          "MAMBA_ROOT_PREFIX": "/opt/conda"})
    .run_commands(
        # CUDA-enabled openmm + openmmtools + pdbfixer in a dedicated env.
        "/opt/conda/bin/micromamba create -y -n md -c conda-forge "
        "python=3.13 'openmm=8.5.1' openmmtools pdbfixer numpy scipy "
        "'cuda-version>=12.9,<13' && /opt/conda/bin/micromamba clean -a -y",
        "/opt/conda/envs/md/bin/python -c 'import openmm; from openmm import Platform; "
        "print(\"OpenMM\", openmm.version.full_version); "
        "[print(\"  \", Platform.getPlatform(i).getName()) for i in range(Platform.getNumPlatforms())]'",
    )
    # Bundle our local prep.py/produce.py so the remote function can import them.
    .add_local_dir(str(Path(__file__).parent), "/workspace", copy=True)
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOL_MOUNT = "/vol"
PYTHON = "/opt/conda/envs/md/bin/python"


@app.function(
    gpu="H100",
    timeout=4 * 3600,
    volumes={VOL_MOUNT: vol},
)
def prep_remote(system: str, pdb_bytes: bytes, padding_nm: float = 1.0,
                 temperature_K: float = 310.0, ionic_strength: float = 0.15,
                 dt_fs: float = 4.0, hmr_amu: float = 4.0,
                 com_restrain: str | None = None,
                 com_restrain_k: float = 100.0):
    """Run prep.py on Modal. Writes prepared/<system>/ on the volume.

    pdb_bytes: raw bytes of the input PDB. Pushed by the launcher so we
    don't depend on the user having pre-populated the volume.
    com_restrain: "CHAIN_A:CHAIN_B" string forwarded to prep.py's
    --com-restrain (e.g. "A:B" for Notch1 NRR NEC<->NTM).
    """
    import subprocess
    import sys

    in_dir = Path(VOL_MOUNT) / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(VOL_MOUNT) / "prepared" / system
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = in_dir / f"{system}.pdb"
    pdb_path.write_bytes(pdb_bytes)

    cmd = [PYTHON, "/workspace/prep.py", str(pdb_path), str(out_dir),
           "--padding-nm", str(padding_nm),
           "--ionic-strength", str(ionic_strength),
           "--temperature", str(temperature_K),
           "--dt-fs", str(dt_fs),
           "--hmr-amu", str(hmr_amu)]
    if com_restrain:
        cmd += ["--com-restrain", com_restrain,
                "--com-restrain-k", str(com_restrain_k)]
    print(f"[modal.prep] {' '.join(cmd)}")
    sys.stdout.flush()
    r = subprocess.run(cmd, check=True)
    vol.commit()
    return {"system": system, "out_dir": str(out_dir), "exit_code": r.returncode}


@app.function(
    gpu="A100-80GB",
    timeout=24 * 3600,
    volumes={VOL_MOUNT: vol},
)
def produce_remote(system: str, replica: int, ns: float = 100.0,
                    dcd_interval_ps: float = 10.0, dt_fs: float = 4.0):
    """Run produce.py for one replica on a single GPU.

    Steps and DCD stride are computed from the chosen timestep (dt_fs).
    The integrator dt itself comes from prepared/<system>/integrator.xml,
    written by prep at the same dt; if you change dt_fs here without
    re-prepping you'll break the energy conservation budget.
    """
    import subprocess
    import sys

    out_dir = Path(VOL_MOUNT) / "prepared" / system
    if not (out_dir / "system.xml").exists():
        raise RuntimeError(f"no prepared system at {out_dir}; run prep first")

    steps_per_ps = 1000.0 / dt_fs       # 250 at 4 fs, 500 at 2 fs
    steps = int(ns * 1000 * steps_per_ps)
    dcd_interval = int(dcd_interval_ps * steps_per_ps)
    report_interval = max(1000, dcd_interval // 5)

    cmd = [PYTHON, "/workspace/produce.py", str(out_dir),
           "--replica", str(replica),
           "--steps", str(steps),
           "--report-interval", str(report_interval),
           "--dcd-interval", str(dcd_interval)]
    print(f"[modal.produce] {' '.join(cmd)}")
    sys.stdout.flush()
    r = subprocess.run(cmd, check=True)
    vol.commit()
    return {"system": system, "replica": replica, "ns": ns,
            "out_dir": str(out_dir / f"replica_{replica}")}


@app.local_entrypoint()
def prep(system: str, pdb: str, padding_nm: float = 1.0,
         temperature: float = 310.0, ionic_strength: float = 0.15,
         dt_fs: float = 4.0, hmr_amu: float = 4.0,
         com_restrain: str = "", com_restrain_k: float = 100.0):
    """Upload a local PDB to Modal and run prep on it.

    com_restrain: "CHAIN_A:CHAIN_B" passed to prep.py's --com-restrain.
    Example: com_restrain="A:B" for Notch1 NRR NEC<->NTM COM-distance
    restraint (chain IDs of the equilibrated PDB, after PDBFixer
    renumbers them).
    """
    pdb_bytes = Path(pdb).read_bytes()
    print(f"[local] uploading {len(pdb_bytes)} bytes for system={system}"
          + (f"  com_restrain={com_restrain}@k={com_restrain_k}" if com_restrain else ""))
    result = prep_remote.remote(system, pdb_bytes,
                                 padding_nm=padding_nm,
                                 temperature_K=temperature,
                                 ionic_strength=ionic_strength,
                                 dt_fs=dt_fs, hmr_amu=hmr_amu,
                                 com_restrain=com_restrain or None,
                                 com_restrain_k=com_restrain_k)
    print(f"[local] prep done: {result}")


@app.local_entrypoint()
def produce(system: str, replica: int = 0, ns: float = 100.0,
             dcd_interval_ps: float = 10.0, dt_fs: float = 4.0):
    """Run one production replica synchronously and wait for completion."""
    result = produce_remote.remote(system, replica, ns=ns,
                                    dcd_interval_ps=dcd_interval_ps,
                                    dt_fs=dt_fs)
    print(f"[local] produce done: {result}")


@app.local_entrypoint()
def fanout(system: str, replicas: int = 3, ns: float = 100.0,
            dcd_interval_ps: float = 10.0, dt_fs: float = 4.0,
            start_replica: int = 0):
    """Launch N replicas in parallel via spawn(); fire-and-forget.

    Replica indices are `start_replica` through `start_replica + replicas - 1`,
    so e.g. start_replica=1, replicas=2 runs indices 1 and 2 (useful when
    replica 0 already exists from a prior run).

    Print function-call IDs so the user can poll status independently
    (no local CLI connection needed afterward).
    """
    handles = []
    for r in range(start_replica, start_replica + replicas):
        h = produce_remote.spawn(system, r, ns=ns,
                                  dcd_interval_ps=dcd_interval_ps,
                                  dt_fs=dt_fs)
        print(f"[modal] {system} replica {r}: function_call_id={h.object_id}")
        handles.append(h.object_id)
    print(f"[modal] {replicas} replicas of {system} spawned "
          f"(indices {start_replica}..{start_replica + replicas - 1})")
    return handles


@app.local_entrypoint()
def chignolin_full(ns: float = 1000.0):
    """Chignolin Tier-1: single 1 us replica on an H100. ~16 hr wall, ~$120."""
    return fanout("chignolin", replicas=1, ns=ns, dcd_interval_ps=10.0)


@app.local_entrypoint()
def notch1_apo(ns: float = 100.0, replicas: int = 3):
    """3x100 ns Notch1 NRR apo (PDB 3I08) on H100s in parallel."""
    return fanout("notch1_apo", replicas=replicas, ns=ns, dcd_interval_ps=20.0)


@app.local_entrypoint()
def notch1_holo(ns: float = 100.0, replicas: int = 3):
    """3x100 ns Notch1 NRR holo (PDB 3L95) on H100s in parallel."""
    return fanout("notch1_holo", replicas=replicas, ns=ns, dcd_interval_ps=20.0)


@app.local_entrypoint()
def status():
    """List recent function calls on this app."""
    # Modal CLI has `modal app list` and `modal call list` for this.
    print("Use `modal app list` and `modal call logs <function_call_id>` to poll status.")
    print("Or `modal volume get chimerax-vampnet-md prepared/<system>/replica_<r>/` to pull a trajectory.")
