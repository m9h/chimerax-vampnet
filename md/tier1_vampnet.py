"""Tier-1 end-to-end test: fit a small VAMPnet on the alanine dipeptide
trajectory and verify that the learned soft-assignment partitions phi/psi
space into well-separated basins (i.e. each state has a distinct
phi/psi centroid).

Runs OUTSIDE ChimeraX -- bypasses the bundle's session machinery and
calls the same featurize_torsions / VAMPNet stack the bundle uses
internally. This is the actual Day-3/Day-4 validation.

  $ .venv/bin/python md/tier1_vampnet.py /data/datasets/chimerax-vampnet/ala 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing the bundle's featurize.py without invoking ChimeraX.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurize import featurize_torsions  # noqa: E402


def _dihedral(p0, p1, p2, p3):
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9))
    return np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))


def _read_dcd(path: Path, n_atoms: int) -> np.ndarray:
    """Minimal DCD parser for openmm-written trajectories."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    pos += 4 + 4 + 80 + 4
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4 + blocksize + 4
    pos += 4
    natom = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4 + 4
    has_cell = True  # openmm always writes cell block
    # Re-parse header to detect has_cell properly.
    pos = 4 + 4
    header_ints = struct.unpack("<20i", data[pos:pos + 80])
    pos = 0  # restart from top
    pos += 4 + 4 + 80 + 4
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4 + blocksize + 4
    pos += 4
    pos += 4 + 4
    has_cell = header_ints[10] != 0
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
            if frame.shape[0] != natom:
                break
            frames.append(frame)
        except Exception:
            break
    return np.stack(frames, axis=0).astype(np.float32)


def _parse_pdb_atoms(pdb_path: Path):
    """Return [(res_id, res_name, atom_name, atom_index), ...] in file order."""
    atoms = []
    idx = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resid = line[22:26].strip()
                resnm = line[17:20].strip()
                aname = line[12:16].strip()
                atoms.append((resid, resnm, aname, idx))
                idx += 1
    return atoms


def _find_atom(atoms, res_name, atom_name):
    for (resid, resnm, aname, i) in atoms:
        if resnm == res_name and aname == atom_name:
            return i
    raise KeyError(f"{res_name}/{atom_name} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", type=Path)
    ap.add_argument("n_states", type=int)
    ap.add_argument("--lag", type=int, default=20, help="lag time in frames (2 ps each)")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    atoms = _parse_pdb_atoms(args.traj_dir / "ala_dipeptide.pdb")
    phi_q = (_find_atom(atoms, "ACE", "C"), _find_atom(atoms, "ALA", "N"),
              _find_atom(atoms, "ALA", "CA"), _find_atom(atoms, "ALA", "C"))
    psi_q = (_find_atom(atoms, "ALA", "N"), _find_atom(atoms, "ALA", "CA"),
              _find_atom(atoms, "ALA", "C"), _find_atom(atoms, "NME", "N"))
    n_atoms_total = len(atoms)
    coords = _read_dcd(args.traj_dir / "traj.dcd", n_atoms=n_atoms_total)
    print(f"[tier1] coords {coords.shape}")

    quads = [phi_q, psi_q]
    X = featurize_torsions(coords, quads)
    print(f"[tier1] feature matrix {X.shape}  (sin/cos for {len(quads)} torsions)")

    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    n_states = args.n_states
    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 32), nn.ELU(),
        nn.Linear(32, 32), nn.ELU(),
        nn.Linear(32, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=args.lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    vampnet = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    print(f"[tier1] fitting VAMPnet  n_states={n_states}  lag={args.lag}  epochs={args.epochs}")
    model = vampnet.fit(loader, n_epochs=args.epochs).fetch_model()

    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)

    # Recover phi/psi for each frame so we can report per-state centroids.
    phi = np.degrees(_dihedral(coords[:, phi_q[0]], coords[:, phi_q[1]],
                                coords[:, phi_q[2]], coords[:, phi_q[3]]))
    psi = np.degrees(_dihedral(coords[:, psi_q[0]], coords[:, psi_q[1]],
                                coords[:, psi_q[2]], coords[:, psi_q[3]]))

    print(f"\n[tier1] per-state populations + centroids:")
    for s in range(n_states):
        mask = hard == s
        if not mask.any():
            print(f"  state {s}: EMPTY")
            continue
        # Circular mean via sin/cos averaging.
        phi_c = np.degrees(np.arctan2(np.sin(np.radians(phi[mask])).mean(),
                                       np.cos(np.radians(phi[mask])).mean()))
        psi_c = np.degrees(np.arctan2(np.sin(np.radians(psi[mask])).mean(),
                                       np.cos(np.radians(psi[mask])).mean()))
        print(f"  state {s}: pop={mask.mean()*100:5.1f}%  "
              f"phi_centroid={phi_c:7.1f} deg  psi_centroid={psi_c:7.1f} deg")

    msm = MaximumLikelihoodMSM(reversible=True, lagtime=args.lag).fit_fetch(hard)
    print(f"\n[tier1] MSM implied timescales (k={n_states-1}): "
          + ", ".join(f"{t*0.002:.2f} ns" for t in msm.timescales(k=n_states - 1)))
    print(f"[tier1] stationary distribution: "
          + ", ".join(f"{p*100:.1f}%" for p in msm.stationary_distribution))

    n_unique = len(set(hard))
    print(f"\n[tier1] FOUND {n_unique} / {n_states} distinct states")
    if n_unique == n_states:
        print("[tier1] VAMPnet recovered the requested number of metastable states ✓")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
