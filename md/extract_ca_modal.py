"""Modal-side CA extractor for MD trajectories on the chimerax-vampnet-md
volume.

For each replica of a given system, stream the DCD on the Modal
volume, downsample to a target ps stride, extract Cα atoms only,
and write a compact `ca_traj.npz` next to the DCD. The CA-only
npz files are then small enough to pull locally for analysis.

  modal run md/extract_ca_modal.py::run --system hsp90_ntd_apo_v1
  modal run md/extract_ca_modal.py::run --system hsp90_ntd_holo_v1

Output schema (per replica):
  ca_traj.npz with keys:
    - coords_A:        (n_frames, n_ca, 3) float32, Angstroms
    - chains:          (n_ca,) U2 chain IDs
    - resids:          (n_ca,) int residue numbers (per-chain)
    - frame_stride_ps: scalar, ps between consecutive frames
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-extract-ca"
VOLUME_NAME = "chimerax-vampnet-md"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("numpy", "mdtraj")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(timeout=2 * 3600, volumes={"/vol": vol})
def extract_remote(system: str, n_replicas: int = 3,
                    target_stride_ps: float = 20.0,
                    dcd_frame_ps: float = 10.0):
    """Extract Cα npz per replica. dcd_frame_ps is the ps per stored
    DCD frame (from the produce-time dcd_interval × dt). At 4 fs
    HMR with dcd_interval=2500 steps → 10 ps per DCD frame.
    target_stride_ps=20 then strides every 2nd frame."""
    import json
    import numpy as np
    import mdtraj as md

    base = Path("/vol/prepared") / system
    eq_pdb = base / "equilibrated.pdb"
    if not eq_pdb.exists():
        raise FileNotFoundError(f"missing topology {eq_pdb}")

    stride = max(1, int(round(target_stride_ps / dcd_frame_ps)))
    print(f"[extract] system={system} stride={stride} (every "
          f"{stride * dcd_frame_ps:.0f} ps)")

    summary = []
    for r in range(n_replicas):
        rep_dir = base / f"replica_{r}"
        dcd = rep_dir / "traj.dcd"
        if not dcd.exists():
            print(f"  [warn] replica {r}: no DCD at {dcd}; skipping")
            continue
        print(f"  [extract] replica {r}: loading {dcd} (stride {stride})")
        traj = md.load(str(dcd), top=str(eq_pdb), stride=stride)
        ca_idx = traj.topology.select("name CA")
        coords_A = (traj.xyz[:, ca_idx, :] * 10.0).astype("float32")
        chains = np.array([
            traj.topology.atom(int(i)).residue.chain.chain_id for i in ca_idx
        ], dtype="U2")
        resids = np.array([
            traj.topology.atom(int(i)).residue.resSeq for i in ca_idx
        ], dtype=np.int32)
        out = rep_dir / "ca_traj.npz"
        np.savez_compressed(
            out,
            coords_A=coords_A,
            chains=chains,
            resids=resids,
            frame_stride_ps=np.float32(stride * dcd_frame_ps),
        )
        print(f"  [extract] replica {r}: wrote {out} "
              f"({coords_A.shape} → {out.stat().st_size/(1<<20):.1f} MiB)")
        summary.append({
            "replica": r,
            "n_frames": int(coords_A.shape[0]),
            "n_ca": int(coords_A.shape[1]),
            "stride_ps": float(stride * dcd_frame_ps),
            "out_path": str(out),
        })

    vol.commit()
    return {"system": system, "n_replicas": n_replicas, "summary": summary}


@app.local_entrypoint()
def run(system: str, n_replicas: int = 3, stride_ps: float = 20.0,
         dcd_frame_ps: float = 10.0):
    """Extract CA-only npz arrays for each replica of `system`."""
    r = extract_remote.remote(system, n_replicas, stride_ps, dcd_frame_ps)
    print(f"[local] extract done: {r}")
