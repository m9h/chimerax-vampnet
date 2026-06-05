"""Hsp90 NTD cryptic pocket analysis — per (state, source) opening
distributions on the v0.7 6-source ensemble (apo MD + holo MD + 4
generative).

CA-based pocket-opening proxies (cheap, uses existing CA arrays):

  1. Lid-loop COM distance to pocket-floor COM
     - Lid loop: residues 100-115 (local CA indices 83-98)
     - Pocket floor: residues 50-60 (local CA indices 33-43, contains
       the conserved Asn51 binding residue)
     - High distance → lid lifted → pocket open

  2. Cα(Asn51) — Cα(Thr109) gate distance
     - Asn51 is on the β-sheet floor; Thr109 is on Helix 2 / lid loop
     - Direct distance between the two CAs spanning the pocket
       entrance. Larger = more open.

  3. Lid-loop Rg (replicated from v0.7 W3 biology for completeness)

Pocket-lining residues used (1YER numbering → local CA index):
  Asn51  → 34 (binding residue, β-sheet floor)
  Lys58  → 41 (binding)
  Ile96  → 79 (β-sheet)
  Met98  → 81 (β-sheet, gates pocket)
  Asp102 → 85 (Helix 2 N-cap)
  Thr109 → 92 (Helix 2 / lid loop)
  Thr115 → 98 (lid loop end)
  Leu122 → 105 (post-lid)
  Phe138 → 121 (binding)
  Thr184 → 167 (binding, far β-sheet)

Outputs:
  md/figures/hsp90_cryptic_pocket.png — 4-panel scatter (3 pocket
    metrics + lid-Rg) per (state × source)
  md/hsp90_cryptic_pocket_results.json — machine-readable per-state
    per-source distributions
  md/hsp90_cryptic_pocket_results.md — biological interpretation

Run:
  $ .venv/bin/python md/hsp90_cryptic_pocket.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "md" / "figures"
sys.path.insert(0, str(ROOT / "md"))

# Local CA indices in the 207-CA (1YER 17-223) frame.
LID_LOOP = slice(83, 99)        # residues 100-115 (Helix 2 / lid)
POCKET_FLOOR = slice(33, 44)    # residues 50-60 (β-sheet floor)
IDX_ASN51 = 34
IDX_THR109 = 92


def _load_source_ca(path):
    """Load CA coords from a generative npz."""
    p = ROOT / path
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "coords_ca" not in d.files:
        return None
    return d["coords_ca"].astype(np.float32)


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


def _radius_of_gyration(coords):
    com = coords.mean(axis=1, keepdims=True)
    d2 = ((coords - com) ** 2).sum(-1)
    return np.sqrt(d2.mean(axis=1))


def _com_distance(coords, a_slice, b_slice):
    return np.linalg.norm(
        coords[:, a_slice, :].mean(axis=1) - coords[:, b_slice, :].mean(axis=1),
        axis=-1,
    )


def main():
    print("=" * 70)
    print("Hsp90 NTD cryptic pocket analysis (CA-based proxies)")
    print("=" * 70)

    apo_base = Path(
        "/data/datasets/chimerax-vampnet/hsp90_v1/hsp90_ntd_apo_v1"
    )
    holo_base = Path(
        "/data/datasets/chimerax-vampnet/hsp90_v1/hsp90_ntd_holo_v1"
    )

    # Load MD (apo, holo: holo sliced to common 207 CAs).
    apo_md = np.concatenate(
        [np.load(apo_base / f"replica_{r}/ca_traj.npz")["coords_A"]
          for r in range(3)], axis=0,
    )
    holo_md = np.concatenate(
        [np.load(holo_base / f"replica_{r}/ca_traj.npz")["coords_A"][:, 6:, :]
          for r in range(3)], axis=0,
    )

    sources = {"MD_apo": apo_md, "MD_holo": holo_md}
    for tag, path in [("MarS-FM", "hsp90_ntd_marsfm200.npz"),
                       ("BioEmu", "hsp90_ntd_bioemu200.npz"),
                       ("Boltz-2", "hsp90_ntd_boltz200.npz"),
                       ("AlphaFlow", "hsp90_ntd_af200.npz")]:
        ca = _load_source_ca(path)
        if ca is not None:
            sources[tag] = ca
            print(f"  {tag}: {ca.shape[0]} frames")
    print(f"  MD_apo:  {apo_md.shape[0]} frames")
    print(f"  MD_holo: {holo_md.shape[0]} frames")

    src_names = list(sources.keys())
    all_coords = np.concatenate([sources[s] for s in src_names], axis=0)
    src_idx = np.concatenate([
        np.full(sources[s].shape[0], i, dtype=np.int64)
        for i, s in enumerate(src_names)
    ])

    # Compute per-frame pocket metrics.
    print("Computing pocket metrics…")
    feat_lid_floor = _com_distance(all_coords, LID_LOOP, POCKET_FLOOR)
    feat_n51_t109 = np.linalg.norm(
        all_coords[:, IDX_ASN51, :] - all_coords[:, IDX_THR109, :], axis=-1,
    )
    feat_lid_rg = _radius_of_gyration(all_coords[:, LID_LOOP, :])
    feat_floor_lid_e2e = np.linalg.norm(
        all_coords[:, IDX_ASN51, :] - all_coords[:, 98, :], axis=-1,
    )

    # 6-source VAMPnet state assignments (same as joint apo+holo H3).
    print("Fitting 6-source VAMPnet for state coloring…")
    X = _featurize(all_coords)
    hard = _fit_vampnet(X)

    features = {
        "lid-floor_COM_A":  feat_lid_floor,
        "Asn51-Thr109_A":   feat_n51_t109,
        "lid-Rg_A":         feat_lid_rg,
        "Asn51-Thr115_A":   feat_floor_lid_e2e,
    }

    # Per-state per-source means + std.
    print("\n" + "=" * 70)
    print("Per-(state × source) pocket-opening metrics (Å)")
    print("=" * 70)
    cols = list(features.keys())
    print("| state | source | n | " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * (len(cols) + 3)) + "|")
    breakdown = []
    for s in range(4):
        for i, sname in enumerate(src_names):
            mask = (hard == s) & (src_idx == i)
            n_in = int(mask.sum())
            row = {"state": s, "source": sname, "n_frames": n_in}
            if n_in == 0:
                print(f"| {s} | {sname} | 0 | " + " | ".join(["—"] * len(cols)) + " |")
            else:
                cells = []
                for fname, fvals in features.items():
                    m, sd = fvals[mask].mean(), fvals[mask].std()
                    row[fname + "_mean"] = float(m)
                    row[fname + "_std"] = float(sd)
                    cells.append(f"{m:.1f}±{sd:.1f}")
                print(f"| {s} | {sname} | {n_in} | " + " | ".join(cells) + " |")
            breakdown.append(row)

    # JSON dump.
    json_path = ROOT / "md" / "hsp90_cryptic_pocket_results.json"
    json_path.write_text(json.dumps({
        "sources": src_names,
        "n_frames_per_source": {s: int(sources[s].shape[0]) for s in src_names},
        "pocket_metrics": breakdown,
        "definitions": {
            "LID_LOOP_local_idx": [LID_LOOP.start, LID_LOOP.stop],
            "POCKET_FLOOR_local_idx": [POCKET_FLOOR.start, POCKET_FLOOR.stop],
            "IDX_ASN51_local": IDX_ASN51,
            "IDX_THR109_local": IDX_THR109,
        },
    }, indent=2))
    print(f"\nwrote {json_path}")

    # Figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for ax, (fname, fvals) in zip(axes.flatten(), features.items()):
            for i, sname in enumerate(src_names):
                for s in range(4):
                    mask = (src_idx == i) & (hard == s)
                    if mask.sum() < 3:
                        continue
                    x = s + 0.12 * (i - 2.5)
                    ax.scatter(
                        [x] * mask.sum(), fvals[mask],
                        alpha=0.3, s=5, c=colors[i],
                    )
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"S{s}" for s in range(4)])
            ax.set_ylabel(f"{fname} (Å)")
            ax.set_title(fname)
            ax.grid(alpha=0.3)
        handles = [plt.scatter([], [], c=colors[i], label=sname, s=20)
                    for i, sname in enumerate(src_names)]
        fig.legend(handles=handles, loc="lower center", ncol=len(src_names),
                    bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
        fig.suptitle("Hsp90 NTD cryptic pocket opening (CA proxies, v0.7)",
                      fontsize=14)
        fig.tight_layout()
        fig_path = FIG_DIR / "hsp90_cryptic_pocket.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"wrote {fig_path}")
    except ImportError:
        print("[warn] matplotlib unavailable")


if __name__ == "__main__":
    main()
