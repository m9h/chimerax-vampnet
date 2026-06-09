"""Protein-agnostic H3 multi-source joint VAMPnet analysis (v0.7).

Refactored from `md/notch1_h3_multisource.py` (which had Notch1-
specific paths and chain-A 174-CA slicing baked in). v0.7's Hsp90
NTD pilot needs the same analysis on a different protein, so the
hardcoded Notch1 bits are now per-system config dicts:

  SYSTEMS = {
      "notch1_apo_v3": {
          "md_loader":   ...,
          "md_ca_range": slice(0, 174),
          "source_npz_pattern": {
              "MarS-FM":   "notch1_apo_NEC200_marsfm.npz",
              "BioEmu":    "notch1_NEC_bioemu200.npz",
              ...
          },
      },
      "hsp90_ntd": { ... },
  }

  $ .venv/bin/python md/multisource_h3.py --system hsp90_ntd
  $ .venv/bin/python md/multisource_h3.py --system notch1_apo_v3  # back-compat

The Notch1 v0.5 5-source result is fully reproducible via the
notch1_apo_v3 system entry. The Hsp90 NTD entry will work as soon
as the v0.7 generative ensembles land on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------
# Per-system config: each entry says how to load the MD reference
# ensemble and where to find the per-source generative npz files.
# --------------------------------------------------------------------

ZEN_V3 = Path("/data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/"
                "v0.3/notch1_apo_v3")


def _load_notch1_apo_v3_md():
    """3 × 100 ns Notch1 apo NRR, chain A NEC subset (first 174 CAs)."""
    if not ZEN_V3.exists():
        raise FileNotFoundError(f"Notch1 v0.3 deposit missing at {ZEN_V3}")
    cas = []
    for r in range(3):
        d = np.load(ZEN_V3 / f"replica_{r}/ca_traj.npz")
        coords = d["coords_A"]
        chains = d["chains"].astype("U2")
        cas.append(coords[:, chains == "A", :])
    return np.concatenate(cas, axis=0)


def _load_hsp90_ntd_apo_v1_md():
    """3 × 300 ns Hsp90 NTD apo MD (v0.7 W1). Pulls CA arrays from
    /data/datasets/chimerax-vampnet/notch1_modal/hsp90_ntd_apo_v1/...
    once W1 lands; for now raises FileNotFoundError until then."""
    # Convention: same as the v0.3 deposit layout — replica_0/ca_traj.npz
    # written by an analyze step that runs after the Modal MD lands.
    base = Path("/data/datasets/chimerax-vampnet/hsp90_v1/hsp90_ntd_apo_v1")
    if not base.exists():
        raise FileNotFoundError(
            f"Hsp90 NTD apo MD CAs not yet on disk at {base}. "
            "Run the v0.7 W1 analyze step first.")
    cas = []
    for r in range(3):
        rep = base / f"replica_{r}/ca_traj.npz"
        if not rep.exists():
            break
        d = np.load(rep)
        cas.append(d["coords_A"] if "coords_A" in d.files else d[d.files[0]])
    if not cas:
        raise FileNotFoundError(f"no Hsp90 apo replicas at {base}")
    return np.concatenate(cas, axis=0)


SYSTEMS = {
    "notch1_apo_v3": {
        "md_loader": _load_notch1_apo_v3_md,
        "md_ca_range": slice(0, 174),
        "expected_n_ca": 174,
        "source_npz": {
            "MarS-FM":   "notch1_apo_NEC200_marsfm.npz",
            "BioEmu":    "notch1_NEC_bioemu200.npz",
            "Boltz-2":   "notch1_NEC_boltz200.npz",
            "AlphaFlow": "notch1_NEC_af200.npz",
            # ESMFold2 (v0.7.x): must be the SAME single-chain NEC
            # selection as the other sources (174 CAs) to share this
            # VAMPnet feature space — generate with
            #   modal run md/esmfold2_modal.py::sample \
            #     --sequence <NEC seq> --name notch1_NEC \
            #     --n-samples 200 --out notch1_NEC_esmfold2200.npz
            # The multi-chain NEC+NTM ESMFold2 run is a SEPARATE deposit
            # (different CA count) for the complex H2/H3 variant, not this
            # 174-CA NEC comparison.
            "ESMFold2":  "notch1_NEC_esmfold2200.npz",
            # AF3 added in v0.7 W4a when weights arrive.
        },
    },
    "hsp90_ntd": {
        "md_loader": _load_hsp90_ntd_apo_v1_md,
        # Hsp90 NTD MD has 207 CAs (1YER residues 17-223), no chain
        # slicing needed (the MD is NTD-only).
        "md_ca_range": slice(None),
        "expected_n_ca": 207,
        "source_npz": {
            "MarS-FM":   "hsp90_ntd_marsfm200.npz",
            "BioEmu":    "hsp90_ntd_bioemu200.npz",
            "Boltz-2":   "hsp90_ntd_boltz200.npz",
            "AlphaFlow": "hsp90_ntd_af200.npz",
            # ESMFold2 (v0.7.x): single-chain NTD, 207 CAs to match. Run
            #   modal run md/esmfold2_modal.py::sample \
            #     --sequence <Hsp90 NTD 17-223 seq> --name hsp90_ntd \
            #     --n-samples 200 --out hsp90_ntd_esmfold2200.npz
            "ESMFold2":  "hsp90_ntd_esmfold2200.npz",
        },
    },
    "b2ar_2rh1": {
        # v0.8 W3: β2AR 2RH1 inactive (282 residues, 7TM GPCR). No MD
        # reference yet — membrane MD prep NaN'd at NVT (v0.8 W1 deferred).
        # The MD-less generative-only analysis still tests the cross-
        # sampler H3 finding on a third fold class (Class A GPCR).
        "md_loader": lambda: (_ for _ in ()).throw(
            FileNotFoundError("β2AR membrane MD deferred — see v0.8 W1")),
        "md_ca_range": slice(None),
        "expected_n_ca": 282,
        "source_npz": {
            "MarS-FM":   "b2ar_2rh1_marsfm200.npz",
            "BioEmu":    "b2ar_2rh1_bioemu200.npz",
            "Boltz-2":   "b2ar_2rh1_boltz200.npz",
            "AlphaFlow": "b2ar_2rh1_af200.npz",
            "ESMFold2":  "b2ar_2rh1_esmfold2200.npz",
        },
    },
}


def _load_source(path: str, key: str, expected_n_ca: int):
    """Load CA coords from a source npz. Returns (n, expected_n_ca, 3)
    or None if absent / mismatched."""
    p = ROOT / path
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if key not in d.files:
        return None
    ca = d[key]
    if ca.shape[1] != expected_n_ca:
        print(f"  [warn] {path}: expected {expected_n_ca} CAs, got "
              f"{ca.shape[1]}; skipping")
        return None
    return ca.astype(np.float32)


# --------------------------------------------------------------------
# Featurization + VAMPnet (unchanged from notch1_h3_multisource.py).
# --------------------------------------------------------------------

def _featurize(coords, max_pairs=500, rng_seed=0):
    N, A, _ = coords.shape
    iu, ju = np.triu_indices(A, k=1)
    rng = np.random.default_rng(rng_seed)
    sel = rng.choice(len(iu), size=min(max_pairs, len(iu)), replace=False)
    pair_idx = np.stack([iu[sel], ju[sel]], axis=1)
    a = coords[:, pair_idx[:, 0], :]
    b = coords[:, pair_idx[:, 1], :]
    raw = np.sqrt(((a - b) ** 2).sum(-1)).astype(np.float32)
    mu = raw.mean(0, keepdims=True)
    sigma = raw.std(0, keepdims=True) + 1e-3
    return ((raw - mu) / sigma).clip(-5.0, 5.0)


def _fit_vampnet(X, n_states=4, lag=20, epochs=60):
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
    ds = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True,
                                          drop_last=True)
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu", epsilon=1e-3)
    model = net.fit(loader, n_epochs=epochs).fetch_model()
    soft = np.asarray(model.transform(X.astype("float32")))
    return soft.argmax(axis=-1)


# --------------------------------------------------------------------
# Main driver.
# --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="notch1_apo_v3",
                    choices=list(SYSTEMS.keys()),
                    help="protein system to analyze")
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--lag", type=int, default=20)
    args = p.parse_args()

    cfg = SYSTEMS[args.system]
    expected_n_ca = cfg["expected_n_ca"]

    print("=" * 70)
    print(f"H3 multi-source joint VAMPnet — system: {args.system} "
          f"({expected_n_ca} CAs)")
    print("=" * 70)

    sources = {}
    try:
        md_ca = cfg["md_loader"]()
        if cfg["md_ca_range"] != slice(None):
            md_ca = md_ca[:, cfg["md_ca_range"], :]
        sources["MD"] = md_ca
        print(f"  MD:         {md_ca.shape[0]} frames, {md_ca.shape[1]} CAs")
    except FileNotFoundError as e:
        print(f"  MD: not available ({e})")

    for sname, path in cfg["source_npz"].items():
        ca = _load_source(path, "coords_ca", expected_n_ca)
        if ca is None:
            print(f"  {sname}: not available (looked at {path})")
            continue
        sources[sname] = ca
        print(f"  {sname}: {ca.shape[0]} frames")

    if len(sources) < 2:
        print(f"\n[abort] need >= 2 sources for joint analysis; have "
              f"{len(sources)}")
        return

    src_names = list(sources.keys())
    all_coords = np.concatenate([sources[s] for s in src_names], axis=0)
    src_idx = np.concatenate([np.full(sources[s].shape[0], i, dtype=np.int64)
                                for i, s in enumerate(src_names)])
    print(f"\nJoint ensemble: {all_coords.shape[0]} frames "
          f"({len(src_names)} sources)")

    X = _featurize(all_coords)
    print(f"Feature matrix: {X.shape}")
    print(f"Fitting joint VAMPnet (k={args.n_states} states, "
          f"lag={args.lag}, ~1 min on CPU)…")
    hard = _fit_vampnet(X, n_states=args.n_states, lag=args.lag)

    print("\n" + "=" * 70)
    print("Per-state source breakdown")
    print("=" * 70)
    header = (f"{'state':<8s}{'pop':>8s}  "
              + "  ".join(f"{s:>10s}" for s in src_names))
    print(header)
    print("-" * len(header))
    state_breakdown = []
    for s in range(args.n_states):
        mask = hard == s
        pop = mask.mean() * 100
        per_src = []
        for i, sname in enumerate(src_names):
            n_in = int(((src_idx == i) & mask).sum())
            n_src = int((src_idx == i).sum())
            per_src.append(n_in / max(n_src, 1) * 100)
        print(f"state {s:>2d} {pop:>7.1f}% "
              + "  ".join(f"{p:>9.1f}%" for p in per_src))
        state_breakdown.append({
            "state": s,
            "population_pct": float(pop),
            "frac_of_src_in_state": {
                sname: float(p) for sname, p in zip(src_names, per_src)
            },
        })

    print("\n" + "=" * 70)
    print("H3 verdict — source-specific state coverage")
    print("=" * 70)
    for s in range(args.n_states):
        mask = hard == s
        if not mask.any():
            continue
        srcs = sorted({src_names[i] for i in range(len(src_names))
                        if ((src_idx == i) & mask).sum() > 0})
        only_from = []
        for i, sname in enumerate(src_names):
            in_s = int(((src_idx == i) & mask).sum())
            in_other = sum(int(((src_idx == j) & mask).sum())
                            for j in range(len(src_names)) if j != i)
            if in_s > 0 and in_other == 0:
                only_from.append(sname)
        tag = f" (UNIQUE to {only_from[0]})" if len(only_from) == 1 else ""
        print(f"  state {s}: reached by {len(srcs)} source(s) {srcs}{tag}")

    out = ROOT / "md" / f"{args.system}_h3_multisource_results.json"
    out.write_text(json.dumps({
        "system": args.system,
        "sources_loaded": src_names,
        "n_frames_per_source": {s: int(sources[s].shape[0]) for s in src_names},
        "n_states": args.n_states,
        "lag": args.lag,
        "state_breakdown": state_breakdown,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
