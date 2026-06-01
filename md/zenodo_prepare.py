"""Build the Zenodo deposit for chimerax-vampnet v0.3.

Reads every replica DCD from the chimerax-vampnet-md volume, extracts
CA-only coordinates (10x stride from 20 ps -> 200 ps), saves as .npz,
and writes a metadata.json with the prep + restraint parameters used.

  modal run md/zenodo_prepare.py::build

Output: /vol/zenodo/v0.3/<system>/<replica>/ca_traj.npz  (a few MB each)
        /vol/zenodo/v0.3/<system>/metadata.json
        /vol/zenodo/v0.3/README.md
        /vol/zenodo/v0.3/zenodo.json    (Zenodo metadata for upload)

Then locally:
  modal volume get chimerax-vampnet-md zenodo/v0.3/ /data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-zenodo"
VOLUME_NAME = "chimerax-vampnet-md"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2")
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


def _read_dcd(path, n_atoms):
    import numpy as np
    with open(path, "rb") as f:
        data = f.read()
    pos = 4 + 4
    header_ints = struct.unpack("<20i", data[pos:pos + 80])
    has_cell = header_ints[10] != 0
    pos = 0
    pos += 4 + 4 + 80 + 4
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4 + blocksize + 4
    pos += 4
    pos += 4 + 4
    frames = []
    while pos + 4 <= len(data):
        if has_cell:
            blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
            pos += 4 + blocksize + 4
        try:
            xyz = []
            for _ in range(3):
                blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
                pos += 4
                vals = np.frombuffer(data[pos:pos + blocksize], dtype="<f4")
                pos += blocksize + 4
                xyz.append(vals)
            frame = np.stack(xyz, axis=-1)
            if frame.shape[0] != n_atoms:
                break
            frames.append(frame)
        except Exception:
            break
    return np.stack(frames, axis=0).astype(np.float32)


def _ca_indices(pdb_path):
    cas, ca_chain, ca_resid = [], [], []
    all_idx = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(all_idx)
                    ca_chain.append(line[21])
                    ca_resid.append(int(line[22:26]))
                all_idx += 1
    return cas, ca_chain, ca_resid, all_idx


SYSTEM_SPECS = {
    # name on volume -> deposit metadata
    "chignolin": {
        "description": "Chignolin CLN025 mini-protein, 1 us at 340 K",
        "protocol": "Tier-1 unit test (chimerax-vampnet v0.1)",
        "input_pdb": "5awl.pdb",
        "temperature_K": 340.0,
        "dcd_interval_ps": 10.0,
        "restraint": None,
        "n_replicas": 1,  # only replica_0 is the analysis-ready one
    },
    "notch1_apo": {
        "description": "Notch1 NRR apo (3I08), 3x100 ns at 310 K, unrestrained",
        "protocol": "v0.2 baseline (no membrane-anchor restraint)",
        "input_pdb": "3i08_apo.pdb",
        "temperature_K": 310.0,
        "dcd_interval_ps": 20.0,
        "restraint": None,
        "n_replicas": 3,
    },
    "notch1_holo_diag": {
        "description": "Notch1 NRR + anti-NRR Fab (3L95 chains X+H+L), 3x100 ns at 310 K, unrestrained",
        "protocol": "v0.2 baseline (no membrane-anchor restraint)",
        "input_pdb": "3l95_holo.pdb (chains X+H+L)",
        "temperature_K": 310.0,
        "dcd_interval_ps": 20.0,
        "restraint": None,
        "n_replicas": 3,
    },
    "notch1_apo_v3": {
        "description": "Notch1 NRR apo (3I08), 3x100 ns at 310 K, with NEC-NTM COM-distance restraint",
        "protocol": "v0.3 (CustomCentroidBondForce on NEC-NTM COM, k=100 kJ/mol/nm^2)",
        "input_pdb": "3i08_apo.pdb",
        "temperature_K": 310.0,
        "dcd_interval_ps": 20.0,
        "restraint": {"type": "com_distance", "chain_a": "A_NEC",
                       "chain_b": "B_NTM", "k_kj_per_nm2": 100.0,
                       "r0_A": 3.94},
        "n_replicas": 3,
    },
    "notch1_holo_v3": {
        "description": "Notch1 NRR holo (3L95, NRR + Fab), 3x100 ns at 310 K, with NEC-NTM COM-distance restraint",
        "protocol": "v0.3 (CustomCentroidBondForce on NEC-NTM COM, k=100 kJ/mol/nm^2)",
        "input_pdb": "3l95_holo.pdb (chains X+H+L, NRR chain X split at S1 cleavage residue 1671 into X+K)",
        "temperature_K": 310.0,
        "dcd_interval_ps": 20.0,
        "restraint": {"type": "com_distance", "chain_a": "A_NEC",
                       "chain_b": "B_NTM", "k_kj_per_nm2": 100.0,
                       "r0_A": 3.98},
        "n_replicas": 3,
    },
}


@app.function(cpu=8, memory=32768, timeout=3600, volumes={"/vol": vol})
def build_remote(stride: int = 10):
    """Extract CA-only NPZs for every replica of every deposit-worthy system.
    stride=10 means 200 ps per frame in the deposited NPZs (down from 20 ps
    in the raw DCDs). 500 frames per 100 ns replica."""
    import numpy as np
    out_root = Path("/vol/zenodo/v0.3")
    out_root.mkdir(parents=True, exist_ok=True)

    for sys_name, spec in SYSTEM_SPECS.items():
        sys_dir = Path("/vol/prepared") / sys_name
        if not sys_dir.exists():
            print(f"[skip] {sys_name}: not on volume")
            continue
        eq_pdb = sys_dir / "equilibrated.pdb"
        if not eq_pdb.exists():
            print(f"[skip] {sys_name}: no equilibrated.pdb")
            continue
        cas, ca_chain, ca_resid, n_atoms = _ca_indices(eq_pdb)
        cas = np.array(cas, dtype=np.int64)
        print(f"[{sys_name}] topology: {n_atoms} atoms, {len(cas)} CAs")

        out_sys = out_root / sys_name
        out_sys.mkdir(parents=True, exist_ok=True)

        replicas_meta = []
        for r in range(spec["n_replicas"]):
            dcd = sys_dir / f"replica_{r}" / "traj.dcd"
            if not dcd.exists():
                print(f"[skip] {sys_name}/replica_{r}: no traj.dcd")
                continue
            print(f"[{sys_name}] reading replica {r} ...")
            coords = _read_dcd(str(dcd), n_atoms=n_atoms)
            n_full = coords.shape[0]
            coords = coords[::stride]
            ca = coords[:, cas, :]
            print(f"  -> {coords.shape[0]} frames (stride {stride} from {n_full}), "
                  f"CA shape {ca.shape}")
            out_rep = out_sys / f"replica_{r}"
            out_rep.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_rep / "ca_traj.npz",
                coords_A=ca.astype("float32"),
                chains=np.array(ca_chain),
                resids=np.array(ca_resid, dtype="int32"),
                frame_stride_ps=spec["dcd_interval_ps"] * stride,
            )
            replicas_meta.append({
                "replica": r,
                "n_frames": int(coords.shape[0]),
                "frame_stride_ps": spec["dcd_interval_ps"] * stride,
                "duration_ns": float(coords.shape[0]) * spec["dcd_interval_ps"] * stride / 1000.0,
                "raw_dcd_n_frames": n_full,
                "raw_dcd_frame_stride_ps": spec["dcd_interval_ps"],
            })

        meta = dict(spec)
        meta["system"] = sys_name
        meta["n_ca"] = int(len(cas))
        meta["chains"] = sorted(set(ca_chain))
        meta["replicas"] = replicas_meta
        meta["frame_stride_ps_in_deposit"] = spec["dcd_interval_ps"] * stride
        (out_sys / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"  metadata -> {out_sys/'metadata.json'}")

    vol.commit()
    return {"systems": list(SYSTEM_SPECS.keys()), "out_root": str(out_root)}


@app.local_entrypoint()
def build():
    r = build_remote.remote()
    print(f"\n[local] done: {r}")
    print(f"[local] pull with: modal volume get chimerax-vampnet-md zeno/v0.3/ "
          f"/data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/")
