"""H3 multi-source joint VAMPnet analysis on Notch1 NEC.

Combines apo NEC ensembles from up to five sources (any subset that
is present on disk):
  1. Classical MD (v0.3 COM-restrained, NEC subset of full NRR)
  2. MarS-FM (notch1_apo_NEC200_marsfm.npz)
  3. BioEmu (notch1_NEC_bioemu200.npz)
  4. Boltz-2 (notch1_NEC_boltz200.npz)
  5. AlphaFlow / ESMFlow-MD (notch1_NEC_af200.npz)

Fits a single VAMPnet on the union of all frames, then reports per-
state source breakdown — the direct test of H3 ("which states does
each source uniquely access?").

  $ .venv/bin/python md/notch1_h3_multisource.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "md"))

# Apo NEC = first 174 CAs of the v0.3 MD NRR (chain A, NEC).
NEC_CA_RANGE = slice(0, 174)


def _load_md_apo_nec_ca():
    """Load apo NEC CAs from the v0.3 MD trajectory by pulling from
    the Modal volume's cached DCDs. Falls back to a stub if the local
    cache is empty; in that case, run md/notch1_h2_modal.py::analyze
    first which downloads + writes the CA arrays to disk."""
    # For convenience, load from the local CA-only deposit if present.
    local_npz = Path(__file__).parent / "notch1_apo_v3_ca.npz"
    if local_npz.exists():
        return np.load(local_npz)["coords"][:, NEC_CA_RANGE, :]
    # Otherwise pull from the Zenodo-staging dir.
    zen = Path("/data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/"
               "v0.3/notch1_apo_v3")
    if zen.exists():
        rep0 = np.load(zen / "replica_0/ca_traj.npz")["coords_A"]
        rep1 = np.load(zen / "replica_1/ca_traj.npz")["coords_A"]
        rep2 = np.load(zen / "replica_2/ca_traj.npz")["coords_A"]
        ca = np.concatenate([rep0, rep1, rep2], axis=0)
        return ca[:, NEC_CA_RANGE, :]
    raise FileNotFoundError(
        "No local MD apo NEC source. Either run "
        "md/notch1_h2_modal.py::analyze to cache MD CAs locally, "
        "or run the Zenodo staging pipeline.")


def _load_source(name, path, key):
    """Load CA coords from a source npz. Returns (n, 174, 3)."""
    p = Path(__file__).parent / path
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if key not in d.files:
        return None
    ca = d[key]
    if ca.shape[1] != 174:
        print(f"  [warn] {name}: expected 174 CAs, got {ca.shape[1]}; skipping")
        return None
    return ca.astype(np.float32)


def _featurize(coords, max_pairs=500, rng_seed=0):
    """CA-CA distance features, z-scored + clipped."""
    N, A, _ = coords.shape
    iu, ju = np.triu_indices(A, k=1)
    total = len(iu)
    rng = np.random.default_rng(rng_seed)
    if total > max_pairs:
        sel = rng.choice(total, size=max_pairs, replace=False)
        pair_idx = np.stack([iu[sel], ju[sel]], axis=1)
    else:
        pair_idx = np.stack([iu, ju], axis=1)
    a = coords[:, pair_idx[:, 0], :]
    b = coords[:, pair_idx[:, 1], :]
    raw = np.sqrt(((a - b) ** 2).sum(-1)).astype(np.float32)
    mu = raw.mean(0, keepdims=True)
    sigma = raw.std(0, keepdims=True) + 1e-3
    z = ((raw - mu) / sigma).clip(-5.0, 5.0)
    return z


def _fit_vampnet(X, n_states, lag, epochs=60):
    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    torch.manual_seed(0)
    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True,
                                          drop_last=True)
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu", epsilon=1e-3)
    model = net.fit(loader, n_epochs=epochs).fetch_model()
    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    return hard


def main():
    sources = {}
    print("=" * 70)
    print("H3 multi-source joint VAMPnet on Notch1 NEC (apo, 174 CAs)")
    print("=" * 70)

    # MD
    try:
        md_ca = _load_md_apo_nec_ca()
        sources["MD"] = md_ca
        print(f"  MD:         {md_ca.shape[0]} frames")
    except Exception as e:
        print(f"  MD: not available ({e})")

    # MarS-FM
    mars = _load_source("MarS-FM", "../notch1_apo_NEC200_marsfm.npz", "coords_ca")
    if mars is None:
        mars = _load_source("MarS-FM", "notch1_apo_NEC200_marsfm.npz", "coords_ca")
    if mars is not None:
        sources["MarS-FM"] = mars
        print(f"  MarS-FM:    {mars.shape[0]} frames")

    # BioEmu
    bioemu = _load_source("BioEmu", "../notch1_NEC_bioemu200.npz", "coords_ca")
    if bioemu is None:
        bioemu = _load_source("BioEmu", "notch1_NEC_bioemu200.npz", "coords_ca")
    if bioemu is not None:
        sources["BioEmu"] = bioemu
        print(f"  BioEmu:     {bioemu.shape[0]} frames")

    # Boltz-2
    boltz = _load_source("Boltz-2", "../notch1_NEC_boltz200.npz", "coords_ca")
    if boltz is None:
        boltz = _load_source("Boltz-2", "notch1_NEC_boltz200.npz", "coords_ca")
    if boltz is not None:
        sources["Boltz-2"] = boltz
        print(f"  Boltz-2:    {boltz.shape[0]} frames")

    # AlphaFlow / ESMFlow-MD
    af = _load_source("AlphaFlow", "../notch1_NEC_af200.npz", "coords_ca")
    if af is None:
        af = _load_source("AlphaFlow", "notch1_NEC_af200.npz", "coords_ca")
    if af is not None:
        sources["AlphaFlow"] = af
        print(f"  AlphaFlow:  {af.shape[0]} frames")

    if len(sources) < 2:
        print(f"\nNeed >= 2 sources for joint analysis; have {len(sources)}. Aborting.")
        return

    # Concatenate; track source origin per frame.
    src_names = list(sources.keys())
    all_coords = np.concatenate([sources[s] for s in src_names], axis=0)
    src_idx = np.concatenate([
        np.full(sources[s].shape[0], i, dtype=np.int64)
        for i, s in enumerate(src_names)
    ])
    print(f"\nJoint ensemble: {all_coords.shape[0]} frames "
          f"({len(src_names)} sources)")

    # Featurize + fit VAMPnet
    n_states = 4
    lag = 20  # arbitrary; generative samples have no time structure
    X = _featurize(all_coords)
    print(f"Feature matrix: {X.shape}")
    print("Fitting joint VAMPnet (k=4 states, ~1 min on CPU)...")
    hard = _fit_vampnet(X, n_states=n_states, lag=lag)

    # Per-state source breakdown
    print("\n" + "=" * 70)
    print("Per-state source breakdown")
    print("=" * 70)
    header = f"{'state':<8s}{'pop':>8s}  " + "  ".join(
        f"{s:>10s}" for s in src_names
    )
    print(header)
    print("-" * len(header))
    state_breakdown = []
    for s in range(n_states):
        mask = hard == s
        pop = mask.mean() * 100
        per_src = []
        for i, sname in enumerate(src_names):
            n_in_state_from_src = ((src_idx == i) & mask).sum()
            n_from_src = (src_idx == i).sum()
            frac_of_src_in_state = n_in_state_from_src / max(n_from_src, 1)
            per_src.append(frac_of_src_in_state * 100)
        row = f"state {s:>2d} {pop:>7.1f}% " + "  ".join(
            f"{p:>9.1f}%" for p in per_src
        )
        print(row)
        state_breakdown.append({
            "state": s,
            "population_pct": float(pop),
            "frac_of_src_in_state": {sname: float(p) for sname, p in zip(src_names, per_src)},
        })

    # H3 verdict: which sources reach which states uniquely?
    print("\n" + "=" * 70)
    print("H3 verdict — source-specific state coverage")
    print("=" * 70)
    for s in range(n_states):
        mask = hard == s
        if not mask.any():
            continue
        srcs_in_state = sorted({src_names[i] for i in range(len(src_names))
                                 if ((src_idx == i) & mask).sum() > 0})
        only_from = []
        for i, sname in enumerate(src_names):
            in_state = ((src_idx == i) & mask).sum()
            from_src = (src_idx == i).sum()
            in_other_states = sum(((src_idx == j) & mask).sum()
                                   for j in range(len(src_names)) if j != i)
            if in_state > 0 and in_other_states == 0:
                only_from.append(sname)
        tag = f" (UNIQUE to {only_from[0]})" if len(only_from) == 1 else ""
        print(f"  state {s}: reached by {len(srcs_in_state)} source(s) "
              f"{srcs_in_state}{tag}")

    out = Path(__file__).parent / "notch1_h3_multisource_results.json"
    out.write_text(json.dumps({
        "sources_loaded": src_names,
        "n_frames_per_source": {s: int(sources[s].shape[0]) for s in src_names},
        "n_states": n_states,
        "state_breakdown": state_breakdown,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
