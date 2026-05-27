"""Tier-1 end-to-end test on the 1 us chignolin trajectory.

CLN025 is the canonical 2-state mini-protein folding benchmark. We
expect:
  - The slowest implied timescale at ~hundreds of ns at 340 K.
  - Two well-separated metastable states: folded beta-hairpin and
    unfolded extended.
  - State 'folded' has small Trp-Tyr CA distance (residue 9 to 1)
    and small radius of gyration; 'unfolded' the opposite.

  $ .venv/bin/python md/tier1_chignolin.py /data/datasets/chimerax-vampnet/chignolin_modal/chignolin/replica_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurize import featurize_ca_distances  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1_vampnet import _read_dcd  # noqa: E402


def _parse_pdb_ca(pdb_path: Path):
    """Return CA atom indices (0-based, in file order)."""
    cas = []
    idx = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(idx)
                idx += 1
    return np.array(cas, dtype=np.int64), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", type=Path,
                    help="dir containing traj.dcd, plus a sibling equilibrated.pdb")
    ap.add_argument("--n-states", type=int, default=2)
    ap.add_argument("--lag", type=int, default=100,
                    help="lag in (subsampled) frames")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="subsample frames for speed; trajectory is 10 ps/frame")
    ap.add_argument("--dcd-stride-ps", type=float, default=10.0,
                    help="DCD frame stride in ps (10 ps default for our 4 fs / 2500-step writer)")
    args = ap.parse_args()

    eq_pdb = args.traj_dir.parent / "equilibrated.pdb"
    if not eq_pdb.exists():
        eq_pdb = args.traj_dir / "equilibrated.pdb"

    ca_idx, n_atoms = _parse_pdb_ca(eq_pdb)
    print(f"[tier1.cln] PDB has {n_atoms} atoms; {len(ca_idx)} are CA")

    coords_all = _read_dcd(args.traj_dir / "traj.dcd", n_atoms=n_atoms)
    total_ns = coords_all.shape[0] * args.dcd_stride_ps / 1000.0
    print(f"[tier1.cln] trajectory: {coords_all.shape}, total {total_ns:.1f} ns @ {args.dcd_stride_ps} ps stride")

    coords = coords_all[::args.frame_stride]
    print(f"[tier1.cln] after stride={args.frame_stride}: {coords.shape}")

    # Restrict to CA atoms.
    ca_coords = coords[:, ca_idx, :]  # (N, 10, 3)

    # CA-distances feature (10 CAs -> 45 pair distances).
    X = featurize_ca_distances(ca_coords)
    print(f"[tier1.cln] features: {X.shape}")

    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    P = X.shape[1]
    n_states = args.n_states
    lobe = nn.Sequential(
        nn.Linear(P, 64), nn.ELU(),
        nn.Linear(64, 64), nn.ELU(),
        nn.Linear(64, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=args.lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    print(f"[tier1.cln] fitting VAMPnet  n_states={n_states} lag={args.lag} epochs={args.epochs}")
    vampnet = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    model = vampnet.fit(loader, n_epochs=args.epochs).fetch_model()

    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    print(f"[tier1.cln] state populations: " +
          "  ".join(f"s{s}={(hard==s).mean()*100:.1f}%" for s in range(n_states)))

    # Define physical observables to distinguish folded/unfolded.
    # 1) Trp9 - Tyr1 CA distance (the hairpin closure).
    trp_tyr = np.linalg.norm(ca_coords[:, 8] - ca_coords[:, 0], axis=-1)
    # 2) Radius of gyration over CAs.
    centroid = ca_coords.mean(axis=1, keepdims=True)
    rg = np.sqrt(((ca_coords - centroid) ** 2).sum(-1).mean(-1))
    # 3) End-to-end distance.
    e2e = np.linalg.norm(ca_coords[:, -1] - ca_coords[:, 0], axis=-1)

    print(f"\n[tier1.cln] per-state physical observables:")
    print(f"  {'state':<7}{'pop':>6}{'Trp9-Tyr1':>13}{'  Rg':>8}{'  e2e':>8}")
    for s in range(n_states):
        mask = hard == s
        if not mask.any():
            print(f"  {s:<7}EMPTY")
            continue
        print(f"  {s:<7}{mask.mean()*100:>5.1f}%"
              f"{trp_tyr[mask].mean():>10.2f} A"
              f"{rg[mask].mean():>8.2f}"
              f"{e2e[mask].mean():>8.2f}")

    msm = MaximumLikelihoodMSM(reversible=True, lagtime=args.lag).fit_fetch(hard)
    its_frames = msm.timescales(k=n_states - 1)
    ps_per_frame = args.frame_stride * args.dcd_stride_ps
    its_ns = [t * ps_per_frame / 1000.0 for t in its_frames]
    print(f"\n[tier1.cln] MSM implied timescales: " +
          ", ".join(f"{t:.1f} ns" for t in its_ns))
    print(f"[tier1.cln] stationary distribution: " +
          ", ".join(f"{p*100:.1f}%" for p in msm.stationary_distribution))

    print(f"\n[tier1.cln] PASS criteria (chignolin-CLN025 reference):")
    print(f"  - slowest timescale should be 100-500 ns at 340 K")
    print(f"  - folded state has Trp9-Tyr1 ~5-7 A; unfolded ~10-15 A")
    print(f"  - folded state has Rg ~5 A; unfolded ~7-8 A")


if __name__ == "__main__":
    main()
