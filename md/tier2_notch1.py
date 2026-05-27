"""Tier-2 analysis: apo vs holo Notch1 NRR.

Trains separate VAMPnets on each system's CA-distance ensemble and
reports:
  - VAMP-2 score (state-separation quality)
  - slowest implied timescale (rate of the dominant slow conformational
    mode)
  - per-state mean radius of gyration

Hypothesis H2 from paper.tex: the anti-NRR Fab measurably stabilizes
the auto-inhibited state. If supported, expect:
  - slower implied timescale in holo (longer-lived states)
  - tighter Rg distribution in holo (less conformational excursion)

  $ .venv/bin/python md/tier2_notch1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurize import featurize_ca_distances  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1_vampnet import _read_dcd  # noqa: E402

DATA_ROOT = Path("/data/datasets/chimerax-vampnet/notch1_modal")


def _parse_ca_indices(pdb_path: Path):
    """Return CA atom indices and total atom count."""
    cas = []
    idx = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(idx)
                idx += 1
    return np.array(cas, dtype=np.int64), idx


def _load_replicas(system_dir: Path, n_replicas: int, frame_stride: int = 5):
    """Load and concatenate per-replica DCDs, subset to CA atoms."""
    eq_pdb = system_dir / "equilibrated.pdb"
    ca_idx, n_atoms = _parse_ca_indices(eq_pdb)

    all_ca = []
    for r in range(n_replicas):
        traj = system_dir / f"replica_{r}" / "traj.dcd"
        if not traj.exists():
            print(f"  [warn] missing {traj}")
            continue
        coords = _read_dcd(traj, n_atoms=n_atoms)
        coords = coords[::frame_stride]
        ca = coords[:, ca_idx, :]
        print(f"  replica {r}: {coords.shape[0]} frames, {ca.shape[1]} CAs")
        all_ca.append(ca)
    stacked = np.concatenate(all_ca, axis=0)
    print(f"  combined: {stacked.shape}")
    return stacked


def _subsample_features(coords, max_features=2000, rng=None):
    """Compute CA-distance feature matrix; subsample pair indices if needed."""
    N, A, _ = coords.shape
    total_pairs = A * (A - 1) // 2
    if total_pairs <= max_features:
        return featurize_ca_distances(coords)
    if rng is None:
        rng = np.random.default_rng(0)
    iu, ju = np.triu_indices(A, k=1)
    keep = rng.choice(total_pairs, size=max_features, replace=False)
    pairs = list(zip(iu[keep].tolist(), ju[keep].tolist()))
    return featurize_ca_distances(coords, ca_pairs=pairs)


def _rg_per_frame(coords):
    centroid = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - centroid) ** 2).sum(-1).mean(-1))


def _fit_vampnet(X, n_states, lag, epochs):
    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    model = net.fit(loader, n_epochs=epochs).fetch_model()
    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=lag).fit_fetch(hard)
    return model, hard, msm


def _find_nrr_chains(pdb_path: Path):
    """Per-chain lists of CA *positional* indices (0-based among CAs only,
    so they index directly into a (N_frames, N_CAs, 3) array)."""
    cas_per_chain: dict = {}
    ca_pos = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    chain = line[21:22]
                    cas_per_chain.setdefault(chain, []).append(ca_pos)
                    ca_pos += 1
    return cas_per_chain


def main():
    DCD_STRIDE_PS = 20.0
    FRAME_STRIDE = 5
    PS_PER_FRAME = DCD_STRIDE_PS * FRAME_STRIDE
    LAG_FRAMES = 50
    N_STATES = 4
    EPOCHS = 60

    results = {}
    for system in ("notch1_apo", "notch1_holo"):
        system_dir = DATA_ROOT / system
        print(f"\n=== {system} ===")
        coords = _load_replicas(system_dir, n_replicas=3, frame_stride=FRAME_STRIDE)

        # Identify NRR chains. For apo: chains A (NEC, 174 CAs) + B (NTM, 60 CAs).
        # For holo: same logical chains likely renumbered; pick the two smallest
        # chains that are 50-200 CAs each — those are the NRR fragments.
        chain_map = _find_nrr_chains(system_dir / "equilibrated.pdb")
        chains = sorted(chain_map.items(), key=lambda kv: len(kv[1]))
        print(f"  chains: " + ", ".join(f"{c}({len(idx)} CAs)" for c, idx in chains))

        if system == "notch1_apo":
            nec_idx = chain_map.get("A", [])
            ntm_idx = chain_map.get("B", [])
        else:
            # Holo: find the two chains shaped like NEC + NTM (Fab heavy ~220, light ~110,
            # so NRR fragments are anything in the 50-180 range).
            small = [c for c, idx in chains if 40 <= len(idx) <= 200]
            if len(small) >= 2:
                nec_idx = chain_map[small[-1]]   # bigger of the small ones
                ntm_idx = chain_map[small[0]]    # smaller
            else:
                nec_idx = chains[0][1]
                ntm_idx = chains[1][1] if len(chains) > 1 else []
        nec_idx = np.array(nec_idx, dtype=np.int64)
        ntm_idx = np.array(ntm_idx, dtype=np.int64)
        print(f"  NRR fragments: NEC ({len(nec_idx)} CAs)  +  NTM ({len(ntm_idx)} CAs)")

        nrr_ca = coords[:, np.concatenate([nec_idx, ntm_idx]), :]
        rg_nrr = _rg_per_frame(nrr_ca)
        nec = coords[:, nec_idx, :]
        ntm = coords[:, ntm_idx, :]
        com_sep = np.linalg.norm(nec.mean(1) - ntm.mean(1), axis=-1)

        # Activation order parameter: NEC-NTM COM separation. In vivo (membrane-
        # constrained) this stays ~10-15 A; activation requires increasing it to
        # ~20+ A to expose the S2 cleavage site.
        print(f"  activation OP (NEC-NTM COM sep):")
        print(f"    mean +/- std:  {com_sep.mean():.1f} +/- {com_sep.std():.1f} A")
        print(f"    range:         {com_sep.min():.1f} - {com_sep.max():.1f} A")
        print(f"    P(sep > 20 A): {(com_sep > 20).mean()*100:.1f}%  (S2-exposing populations)")
        print(f"    P(sep > 50 A): {(com_sep > 50).mean()*100:.1f}%  (fully dissociated)")

        X = _subsample_features(nrr_ca, max_features=2000)
        print(f"  feature matrix: {X.shape}")
        model, hard, msm = _fit_vampnet(X, N_STATES, LAG_FRAMES, EPOCHS)

        try:
            its_frames = msm.timescales(k=N_STATES - 1)
            its_ns = [t * PS_PER_FRAME / 1000.0 for t in its_frames]
        except Exception as e:
            its_ns = []
            print(f"  [warn] implied timescales unavailable: {e}")
        pops = [(hard == s).mean() for s in range(N_STATES)]

        print(f"  implied timescales: " + (", ".join(f"{t:.1f} ns" for t in its_ns) if its_ns else "(insufficient inter-state transitions)"))
        print(f"  state populations:  " + ", ".join(f"{p*100:.1f}%" for p in pops))
        print(f"  per-state COM separation:")
        for s in range(N_STATES):
            mask = hard == s
            if mask.any():
                print(f"    state {s}: pop {mask.mean()*100:5.1f}%   sep {com_sep[mask].mean():5.1f} +/- {com_sep[mask].std():4.1f} A  Rg {rg_nrr[mask].mean():5.2f} A")
            else:
                print(f"    state {s}: EMPTY")

        results[system] = {
            "its_ns": its_ns,
            "com_sep_mean": float(com_sep.mean()),
            "com_sep_std": float(com_sep.std()),
            "com_sep_p20": float((com_sep > 20).mean()),
            "com_sep_p50": float((com_sep > 50).mean()),
            "n_frames": int(coords.shape[0]),
        }

    print("\n=== H2 hypothesis test (apo vs holo) ===")
    apo = results["notch1_apo"]
    holo = results["notch1_holo"]
    print(f"  NEC-NTM COM separation (the activation-direction order parameter):")
    print(f"     apo:  mean {apo['com_sep_mean']:5.1f} A, std {apo['com_sep_std']:5.1f}, P(>20A)={apo['com_sep_p20']*100:5.1f}%, P(>50A)={apo['com_sep_p50']*100:5.1f}%")
    print(f"     holo: mean {holo['com_sep_mean']:5.1f} A, std {holo['com_sep_std']:5.1f}, P(>20A)={holo['com_sep_p20']*100:5.1f}%, P(>50A)={holo['com_sep_p50']*100:5.1f}%")
    print()
    if apo['com_sep_std'] > holo['com_sep_std']:
        ratio = apo['com_sep_std'] / max(holo['com_sep_std'], 1e-6)
        print(f"  H2 SUPPORTED: apo NEC-NTM separation fluctuates {ratio:.1f}x more than holo")
        print(f"  -> antibody stabilizes the auto-inhibited (closed) state")
    else:
        print(f"  H2 NOT supported at this sampling depth (or COM sep is similar in both)")
    if apo['com_sep_p20'] > holo['com_sep_p20'] and apo['com_sep_p20'] > 0:
        print(f"  Activation-direction excursions (sep > 20 A): apo {apo['com_sep_p20']*100:.1f}%, holo {holo['com_sep_p20']*100:.1f}%")


if __name__ == "__main__":
    main()
