"""Quick anchor diagnostic on a single Notch1 v3 replica: load the
trajectory's CA coords and report NEC-NTM COM separation stats. If the
anchor is doing its job, std and max should be drastically smaller than
the no-anchor v0.2 result (apo: 13.9 +/- 24.2 A, range 2.8-107.2)."""

from __future__ import annotations
from pathlib import Path
import modal

APP_NAME = "chimerax-vampnet-notch1-v3-quickcheck"
VOLUME_NAME = "chimerax-vampnet-md"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2")
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


def _read_dcd(path, n_atoms):
    import struct, numpy as np
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


@app.function(cpu=4, memory=16384, timeout=900, volumes={"/vol": vol})
def check_remote(system: str = "notch1_apo_v3", replica: int = 0,
                  stride: int = 5):
    import numpy as np
    base = Path("/vol/prepared") / system
    eq_pdb = base / "equilibrated.pdb"
    cas, ca_chain = [], []
    all_idx = 0
    with open(eq_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(all_idx)
                    ca_chain.append(line[21])
                all_idx += 1
    cas = np.array(cas, dtype=np.int64)
    nec_pos = np.array([i for i, c in enumerate(ca_chain) if c == "A"], dtype=np.int64)
    ntm_pos = np.array([i for i, c in enumerate(ca_chain) if c == "B"], dtype=np.int64)

    dcd = base / f"replica_{replica}" / "traj.dcd"
    coords = _read_dcd(str(dcd), n_atoms=all_idx)
    print(f"loaded {coords.shape[0]} frames, all atoms")
    coords = coords[::stride]
    ca = coords[:, cas, :]
    nec = ca[:, nec_pos, :]
    ntm = ca[:, ntm_pos, :]
    com_sep = np.linalg.norm(nec.mean(1) - ntm.mean(1), axis=-1)

    # Anchor-stability: how much the C-terminal CAs of NTM (anchored)
    # drift from their initial positions across the trajectory.
    anchor_ca_idx = ntm_pos[-5:]  # last 5 CAs of NTM
    anchor_pos = ca[:, anchor_ca_idx, :]  # (N, 5, 3)
    anchor_init = anchor_pos[0]
    anchor_displacement = np.linalg.norm(anchor_pos - anchor_init[None], axis=-1)
    # Per-frame mean displacement across 5 anchored CAs
    mean_disp = anchor_displacement.mean(1)

    result = {
        "system": system,
        "replica": replica,
        "n_frames": int(coords.shape[0]),
        "ns_simulated": float(coords.shape[0] * stride * 20.0 / 1000.0),
        "nec_n_cas": int(len(nec_pos)),
        "ntm_n_cas": int(len(ntm_pos)),
        "com_sep_mean_A": float(com_sep.mean()),
        "com_sep_std_A": float(com_sep.std()),
        "com_sep_min_A": float(com_sep.min()),
        "com_sep_max_A": float(com_sep.max()),
        "com_sep_p_gt_20A": float((com_sep > 20).mean()),
        "com_sep_p_gt_50A": float((com_sep > 50).mean()),
        "anchor_disp_mean_A": float(mean_disp.mean()),
        "anchor_disp_max_A": float(mean_disp.max()),
        "anchor_disp_p_gt_5A": float((mean_disp > 5).mean()),
    }
    print("\n=== NEC-NTM COM separation ===")
    print(f"  mean {result['com_sep_mean_A']:5.1f} +/- {result['com_sep_std_A']:5.1f} A")
    print(f"  range {result['com_sep_min_A']:5.1f} - {result['com_sep_max_A']:5.1f} A")
    print(f"  P(sep > 20A): {result['com_sep_p_gt_20A']*100:5.1f}%   "
          f"P(sep > 50A): {result['com_sep_p_gt_50A']*100:5.1f}%")
    print(f"\n=== Anchored-CA displacement from start ===")
    print(f"  mean {result['anchor_disp_mean_A']:5.1f} A   max {result['anchor_disp_max_A']:5.1f} A")
    print(f"  P(displacement > 5A): {result['anchor_disp_p_gt_5A']*100:5.1f}%")
    print(f"\nv0.2 no-anchor baseline (apo): COM sep 13.9 +/- 24.2 A, range 2.8-107.2")
    return result


@app.local_entrypoint()
def check(system: str = "notch1_apo_v3", replica: int = 0):
    r = check_remote.remote(system=system, replica=replica)
    print(f"\n[local] result: {r}")
