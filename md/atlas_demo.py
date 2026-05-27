"""End-to-end demo of the bundle on an ATLAS-fetched trajectory.

After running md/atlas_fetch.py to grab a protein's MD ensemble, this
script parses the GROMACS .xtc files (via mdtraj fallback or our
internal parser), computes CA-distance features, fits a VAMPnet, and
reports per-state populations + implied timescales.

  $ .venv/bin/python md/atlas_demo.py 1k5n_A
  $ .venv/bin/python md/atlas_demo.py 1k5n_A --n-states 4 --lag 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurize import featurize_ca_distances  # noqa: E402

ATLAS_ROOT = Path("/data/datasets/chimerax-vampnet/atlas")


def _parse_ca_indices(pdb_path: Path):
    """CA atom indices (0-based among all atoms)."""
    cas = []
    idx = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(idx)
                idx += 1
    return np.array(cas, dtype=np.int64), idx


def _load_xtc_via_mdtraj(xtc_path: Path, pdb_path: Path) -> np.ndarray:
    """Load an XTC trajectory and return (N, A, 3) coordinates in
    Angstroms (mdtraj returns nm)."""
    try:
        import mdtraj as md
    except ImportError as e:
        raise RuntimeError(
            "mdtraj is required to read ATLAS XTC files; install with "
            "pip install mdtraj"
        ) from e
    traj = md.load_xtc(str(xtc_path), top=str(pdb_path))
    return traj.xyz * 10.0   # nm -> A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_chain", help="ATLAS entry id, e.g. 1k5n_A")
    ap.add_argument("--root", type=Path, default=ATLAS_ROOT,
                    help="ATLAS download root (default: %(default)s)")
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--lag", type=int, default=50,
                    help="lag in frames (10 ps each)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frame-stride", type=int, default=2)
    args = ap.parse_args()

    sysdir = args.root / f"atlas_{args.pdb_chain}"
    if not sysdir.exists():
        print(f"[atlas-demo] {sysdir} not present; run "
              f"`python md/atlas_fetch.py {args.pdb_chain} {args.root}` first")
        sys.exit(2)

    with (sysdir / "metadata.json").open() as f:
        meta = json.load(f)
    print(f"[atlas-demo] {args.pdb_chain}: {meta.get('protein_name')} "
          f"({meta.get('length')} residues)")
    print(f"  alpha {meta.get('alpha%','?')}%, beta {meta.get('beta%','?')}%, "
          f"coil {meta.get('coil%','?')}%")
    print(f"  avg RMSF {meta.get('avg_RMSF','?')} A, "
          f"avg Rg {meta.get('avg_gyration','?')} A")

    ca_idx, n_atoms = _parse_ca_indices(sysdir / "reference.pdb")
    print(f"  reference structure: {n_atoms} atoms, {len(ca_idx)} CAs")

    all_ca = []
    for r in range(3):
        xtc = sysdir / f"replica_{r}" / "traj.xtc"
        if not xtc.exists():
            print(f"  [warn] missing {xtc}")
            continue
        coords = _load_xtc_via_mdtraj(xtc, sysdir / "reference.pdb")
        ca = coords[:, ca_idx, :]
        ca = ca[::args.frame_stride]
        print(f"  replica {r}: {coords.shape[0]} frames -> {ca.shape[0]} after stride")
        all_ca.append(ca)
    joint = np.concatenate(all_ca, axis=0).astype("float32")
    print(f"  combined: {joint.shape}")

    X = featurize_ca_distances(joint)
    # Subsample to <= 2000 features for speed.
    if X.shape[1] > 2000:
        rng = np.random.default_rng(0)
        keep = rng.choice(X.shape[1], 2000, replace=False)
        X = X[:, keep]
    print(f"  features: {X.shape}")

    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    P = X.shape[1]
    n_states = args.n_states
    lobe = nn.Sequential(
        nn.Linear(P, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=args.lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    print(f"  fitting VAMPnet n_states={n_states} lag={args.lag} epochs={args.epochs}")
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    model = net.fit(loader, n_epochs=args.epochs).fetch_model()

    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=args.lag).fit_fetch(hard)
    its = msm.timescales(k=n_states - 1)
    # 10 ps per ATLAS frame, frame_stride further subsamples.
    ps_per_frame = 10.0 * args.frame_stride
    its_ns = [t * ps_per_frame / 1000.0 for t in its]

    pops = [(hard == s).mean() for s in range(n_states)]
    print(f"\n[atlas-demo] state populations: " +
          ", ".join(f"{p*100:.1f}%" for p in pops))
    print(f"[atlas-demo] implied timescales:  " +
          ", ".join(f"{t:.1f} ns" for t in its_ns))


if __name__ == "__main__":
    main()
