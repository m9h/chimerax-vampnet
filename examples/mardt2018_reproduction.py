"""Reproduce the canonical VAMPnet results from Mardt, Pasquali, Wu, Noé
(Nature Comm. 2018) on alanine dipeptide and a pentapeptide.

The point: the chimerax-vampnet bundle implements the published method,
no black magic. Same inputs + similar hyperparams -> same metastable
state decomposition + similar implied timescales as the paper reports.

Trajectories provided by Frank Noé's mdshare repository (the same
public benchmarks the original paper used):

  https://markovmodel.github.io/mdshare/

  $ .venv/bin/python examples/mardt2018_reproduction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "md"))

from featurize import featurize_ca_distances, featurize_torsions  # noqa: E402

MDSHARE = Path("/data/datasets/chimerax-vampnet/mdshare")


def _load_xtc(xtc: Path, pdb: Path) -> np.ndarray:
    import mdtraj as md
    traj = md.load_xtc(str(xtc), top=str(pdb))
    return traj.xyz * 10.0  # nm -> A


def _phi_psi_indices(top):
    """Return [(C_prev, N, CA, C), (N, CA, C, N_next), ...] index quads
    for backbone phi/psi using mdtraj topology."""
    import mdtraj as md
    phi, _ = md.compute_phi(md.Trajectory(np.zeros((1, top.n_atoms, 3)), top))
    psi, _ = md.compute_psi(md.Trajectory(np.zeros((1, top.n_atoms, 3)), top))
    return list(map(tuple, phi)) + list(map(tuple, psi))


def _fit(X, n_states, lag, epochs=80):
    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 32), nn.ELU(),
        nn.Linear(32, 32), nn.ELU(),
        nn.Linear(32, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    m = net.fit(loader, n_epochs=epochs).fetch_model()
    soft = np.asarray(m.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=lag).fit_fetch(hard)
    return hard, msm


def alanine_dipeptide():
    """Mardt 2018 Fig 2: alanine dipeptide with 6 metastable states.

    Paper used 250 ns trajectories with backbone phi/psi as features.
    Reported slow timescales ~ 1500 ps and ~ 70 ps."""
    import mdtraj as md

    print("\n" + "=" * 70)
    print("Mardt et al. 2018 -- Alanine dipeptide reproduction")
    print("=" * 70)

    pdb = MDSHARE / "alanine-dipeptide-nowater.pdb"
    xtcs = [MDSHARE / f"alanine-dipeptide-{i}-250ns-nowater.xtc" for i in (0, 1, 2)]
    trajs = [md.load_xtc(str(x), top=str(pdb)) for x in xtcs]
    print(f"  3 trajectories, {trajs[0].n_atoms} atoms, "
          f"{sum(t.n_frames for t in trajs)} total frames "
          f"({sum(t.n_frames for t in trajs) * 0.001:.0f} ns @ 1 ps stride)")

    # Use mdtraj's compute_phi + compute_psi directly (cleaner than re-
    # implementing the quad index lookup).
    phi_psi_per_traj = []
    for t in trajs:
        phi = md.compute_phi(t)[1]   # (n_frames, n_phi)
        psi = md.compute_psi(t)[1]
        feats = np.concatenate([np.sin(phi), np.cos(phi),
                                np.sin(psi), np.cos(psi)], axis=1)
        phi_psi_per_traj.append(feats)
    X = np.concatenate(phi_psi_per_traj, axis=0).astype("float32")
    print(f"  features: {X.shape}  (sin/cos of phi + psi)")

    lag = 100  # 100 frames * 1 ps = 100 ps -- matches paper-scale lag
    for n_states in (4, 6):
        hard, msm = _fit(X, n_states, lag)
        its = msm.timescales(k=n_states - 1)
        its_ps = [t * 1.0 for t in its]  # 1 ps per frame
        pops = [(hard == s).mean() * 100 for s in range(n_states)]
        print(f"\n  n_states={n_states}, lag=100 ps:")
        print(f"    populations:  " + "  ".join(f"{p:5.1f}%" for p in pops))
        print(f"    timescales (ps): " + ", ".join(f"{t:.0f}" for t in its_ps))
    print(f"\n  Mardt 2018 Fig 2 reports slow timescales ~1500 ps and ~70 ps")
    print(f"  for n_states=6. Our slowest mode should be in that ballpark.")


def pentapeptide():
    """Mardt 2018 Fig 3: pentapeptide WLALL with 4 metastable states from
    25 x 500 ns implicit-solvent trajectories. Reported slow timescale
    ~ 4 ns at 300 K."""
    import mdtraj as md

    print("\n" + "=" * 70)
    print("Mardt et al. 2018 -- Pentapeptide reproduction")
    print("=" * 70)

    pdb = MDSHARE / "pentapeptide-impl-solv.pdb"
    xtcs = sorted(MDSHARE.glob("pentapeptide-*-500ns-impl-solv.xtc"))
    print(f"  {len(xtcs)} replicas")
    feats_per = []
    for x in xtcs:
        t = md.load_xtc(str(x), top=str(pdb))
        phi = md.compute_phi(t)[1]
        psi = md.compute_psi(t)[1]
        feats = np.concatenate([np.sin(phi), np.cos(phi),
                                np.sin(psi), np.cos(psi)], axis=1).astype("float32")
        feats_per.append(feats)
    X = np.concatenate(feats_per, axis=0)
    print(f"  features: {X.shape}  (sin/cos backbone torsions, {X.shape[1]//4} phi/psi pairs)")
    print(f"  total simulated time: {X.shape[0]} frames * 0.1 ps = {X.shape[0] * 0.1 / 1000:.0f} ns")

    lag = 500   # frames * 0.1 ps = 50 ps
    hard, msm = _fit(X, n_states=4, lag=lag, epochs=60)
    its = msm.timescales(k=3)
    its_ns = [t * 0.1 / 1000 for t in its]
    pops = [(hard == s).mean() * 100 for s in range(4)]
    print(f"\n  n_states=4, lag={lag * 0.1} ps:")
    print(f"    populations:  " + "  ".join(f"{p:5.1f}%" for p in pops))
    print(f"    timescales (ns): " + ", ".join(f"{t:.2f}" for t in its_ns))
    print(f"\n  Mardt 2018 Fig 3 reports a slowest mode ~4 ns at 300 K.")


def main():
    alanine_dipeptide()
    pentapeptide()


if __name__ == "__main__":
    main()
