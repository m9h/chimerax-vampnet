"""Multi-source demo: train one VAMPnet on chignolin MD + a synthetic
AlphaFlow-style ensemble, then report per-source state populations.

The point of multi-source is: an AlphaFlow / BioEmu sample takes ~seconds
to generate (vs hours of MD) but is a static prediction. If the
predictor's samples fall into the SAME metastable basins the MD finds,
you get a free conformational-ensemble upgrade. If they don't, you've
identified states the static predictor missed.

  $ .venv/bin/python md/joint_md_af_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurize import featurize_ca_distances  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1_vampnet import _read_dcd  # noqa: E402

DATA_ROOT = Path("/data/datasets/chimerax-vampnet/chignolin_modal")


def _parse_ca(pdb_path: Path):
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
    eq_pdb = DATA_ROOT / "chignolin" / "equilibrated.pdb"
    ca_idx, n_atoms = _parse_ca(eq_pdb)

    md_coords = _read_dcd(DATA_ROOT / "chignolin" / "replica_0" / "traj.dcd", n_atoms=n_atoms)
    md_ca = md_coords[:, ca_idx, :].astype("float32")
    # Subsample MD for speed (every 50 frames -> 2000 frames of 1 us).
    md_ca = md_ca[::50]
    print(f"[multi] MD: {md_ca.shape}")

    af = np.load(DATA_ROOT / "synthetic_alphaflow.npz")
    af_coords = af["coords"]
    af_ca = af_coords[:, ca_idx, :].astype("float32")
    print(f"[multi] AF synthetic: {af_ca.shape}")

    joint_ca = np.concatenate([md_ca, af_ca], axis=0)
    source = np.array(["md"] * len(md_ca) + ["alphaflow"] * len(af_ca))
    print(f"[multi] joint: {joint_ca.shape}, {(source=='md').sum()} MD + {(source=='alphaflow').sum()} AF")

    X = featurize_ca_distances(joint_ca)
    print(f"[multi] features: {X.shape}")

    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    n_states = 4
    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 64), nn.ELU(),
        nn.Linear(64, 64), nn.ELU(),
        nn.Linear(64, n_states), nn.Softmax(dim=-1),
    )
    # Use lag=20 frames; given MD is stride-50 of 10 ps frames, lag=20
    # corresponds to 10 ns.
    lag = 20
    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    print(f"[multi] fitting VAMPnet n_states={n_states} lag={lag}")
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    model = net.fit(loader, n_epochs=80).fetch_model()

    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)

    print(f"\n[multi] per-state populations split by source:")
    print(f"  {'state':<7}{'MD %':>9}{'AF %':>9}{'MD/AF ratio':>15}")
    for s in range(n_states):
        md_pop = ((hard == s) & (source == "md")).sum() / max(1, (source == "md").sum())
        af_pop = ((hard == s) & (source == "alphaflow")).sum() / max(1, (source == "alphaflow").sum())
        ratio = md_pop / af_pop if af_pop > 0 else float("inf")
        print(f"  {s:<7}{md_pop*100:>8.1f}%{af_pop*100:>8.1f}%{ratio:>14.2f}")

    print(f"\n[multi] interpretation:")
    print(f"  - states where MD and AF populations roughly agree -> AF found a real basin")
    print(f"  - states with MD-only population -> AF missed that conformation")
    print(f"  - states with AF-only population -> AF hallucinated, MD never visits")
    print(f"  (synthetic AF derived from MD, so we expect agreement; real AlphaFlow")
    print(f"   would have more independent variation)")

    # MSM on the joint trajectory (just MD frames in order for time-ordered tau).
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=lag).fit_fetch(hard[: len(md_ca)])
    its = msm.timescales(k=n_states - 1)
    # MD subsampled stride=50 -> 500 ps/frame; lag*500 ps = 10 ns
    its_ns = [t * 0.5 for t in its]   # 0.5 ns/frame at stride 50, 10 ps DCD
    print(f"\n[multi] MSM implied timescales (slowest first): " +
          ", ".join(f"{t:.1f} ns" for t in its_ns))


if __name__ == "__main__":
    main()
