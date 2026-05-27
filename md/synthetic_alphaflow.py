"""Generate a synthetic AlphaFlow-style ensemble from an existing MD
trajectory. Useful for demoing the bundle's multi-source path without
a live AlphaFlow inference run.

Method: pick N frames uniformly from the MD trajectory + add small
Gaussian noise to CA coordinates (sigma_default ~ 0.5 A). Real AlphaFlow
output has more structured variation (modes of the flow-matching prior)
but for pipeline-exercise purposes the synthetic ensemble suffices.

  $ .venv/bin/python md/synthetic_alphaflow.py \\
        /data/datasets/chimerax-vampnet/chignolin_modal/chignolin/replica_0/traj.dcd \\
        /data/datasets/chimerax-vampnet/chignolin_modal/chignolin/equilibrated.pdb \\
        /data/datasets/chimerax-vampnet/chignolin_modal/synthetic_alphaflow.npz \\
        --n-samples 200 --sigma 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1_vampnet import _read_dcd  # noqa: E402


def _parse_atom_count(pdb_path: Path) -> int:
    n = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dcd", type=Path)
    ap.add_argument("equilibrated_pdb", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="Gaussian noise stddev (A) added to coordinates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_atoms = _parse_atom_count(args.equilibrated_pdb)
    coords = _read_dcd(args.traj_dcd, n_atoms=n_atoms)
    print(f"[synth-af] loaded {coords.shape} from {args.traj_dcd}")

    rng = np.random.default_rng(args.seed)
    frame_idx = rng.choice(coords.shape[0], size=args.n_samples, replace=False)
    frame_idx.sort()
    samples = coords[frame_idx].copy()
    noise = rng.normal(0, args.sigma, size=samples.shape).astype("float32")
    samples = (samples + noise).astype("float32")

    # AlphaFlow convention: 'coords' key holds (N, A, 3).
    np.savez_compressed(args.out_npz, coords=samples,
                         _source_frames=frame_idx.astype(np.int32),
                         _sigma_A=args.sigma)
    print(f"[synth-af] wrote {samples.shape} to {args.out_npz}")
    print(f"[synth-af]   sigma={args.sigma} A, sampled {args.n_samples} frames out of {coords.shape[0]}")


if __name__ == "__main__":
    main()
