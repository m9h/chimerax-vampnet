"""Hsp90 NTD H3 biological interpretation — per-state per-source
structural features. Preliminary 4-source version (MD pending).

Mirrors `md/notch1_h3_biology.py` but for the v0.7 Hsp90 pilot.
Without MD, this is a "where does each generative sampler live"
analysis on the 207-CA NTD relative to the 1YER apo crystal.

Features computed per (state, source) bucket:
  1. Rg(NTD)               — radius of gyration (207 CAs)
  2. End-to-end distance   — CA(1) to CA(207)
  3. ATP-pocket COM-COM    — central beta-sheet (50-70) to
                              lid loop (100-120) COM distance.
                              Open vs closed lid is the canonical
                              cryptic-pocket signature.
  4. Lid-loop spread       — Rg of lid loop alone (residues 100-120)
  5. RMSD vs 1YER apo Cα   — divergence from crystal apo

When MD lands (W1), re-run with the MD source added.

Run:
  $ .venv/bin/python md/hsp90_ntd_h3_biology.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "md" / "figures"

# Hsp90α NTD residue mapping (1YER residues 17-223, sliced to local
# indices 0-206 in the generative ensemble outputs).
# These are coarse anchors taken from the standard literature mapping
# of the NTD ATP-binding pocket; refine if needed.
DOMAIN_CENTRAL_BSHEET = slice(34, 54)  # ~residues 51-71 (β-sheet core
                                         # forming back wall of pocket)
DOMAIN_LID_LOOP       = slice(83, 103)  # ~residues 100-120 (the lid)
DOMAIN_HELIX2         = slice(99, 125)  # helix 2 covering pocket entrance


def _load_source(path):
    p = ROOT / path
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "coords_ca" not in d.files:
        return None
    return d["coords_ca"].astype(np.float32)


def _radius_of_gyration(coords):
    com = coords.mean(axis=1, keepdims=True)
    d2 = ((coords - com) ** 2).sum(-1)
    return np.sqrt(d2.mean(axis=1))


def _end_to_end(coords):
    return np.linalg.norm(coords[:, 0, :] - coords[:, -1, :], axis=-1)


def _domain_com(coords, a, b):
    return np.linalg.norm(
        coords[:, a, :].mean(axis=1) - coords[:, b, :].mean(axis=1),
        axis=-1,
    )


def _kabsch_rmsd(a, b):
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    cov = a_c.T @ b_c
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    a_rot = a_c @ rot
    return float(np.sqrt(((a_rot - b_c) ** 2).sum(-1).mean()))


def _rmsd_to_ref(coords, ref):
    return np.array([_kabsch_rmsd(coords[i], ref) for i in range(coords.shape[0])])


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


def _load_1yer_apo_ca():
    """1YER apo crystal CA coords for residues 17-223 (207 CAs).
    Loaded fresh from /tmp/hsp90_pdbs/1yer.pdb."""
    import mdtraj as md
    t = md.load("/tmp/hsp90_pdbs/1yer.pdb")
    ca = t.topology.select("name CA")
    return (t.xyz[0, ca, :] * 10.0).astype(np.float32)


def main():
    print("=" * 70)
    print("Hsp90 NTD H3 biology — preliminary 4-source (MD pending)")
    print("=" * 70)

    sources = {}
    for tag, path in [("MarS-FM",   "hsp90_ntd_marsfm200.npz"),
                       ("BioEmu",    "hsp90_ntd_bioemu200.npz"),
                       ("Boltz-2",   "hsp90_ntd_boltz200.npz"),
                       ("AlphaFlow", "hsp90_ntd_af200.npz")]:
        ca = _load_source(path)
        if ca is None:
            print(f"  [warn] {tag}: missing {path}")
            continue
        sources[tag] = ca
        print(f"  {tag}: {ca.shape[0]} frames, {ca.shape[1]} CAs")

    src_names = list(sources.keys())
    all_coords = np.concatenate([sources[s] for s in src_names], axis=0)
    src_idx = np.concatenate([np.full(sources[s].shape[0], i, dtype=np.int64)
                                for i, s in enumerate(src_names)])

    X = _featurize(all_coords)
    print("\nFitting 4-source joint VAMPnet…")
    hard = _fit_vampnet(X)

    print("Loading 1YER apo crystal reference…")
    try:
        ref_1yer = _load_1yer_apo_ca()
        if ref_1yer.shape[0] != all_coords.shape[1]:
            print(f"  [warn] 1YER CA count {ref_1yer.shape[0]} != "
                  f"ensemble {all_coords.shape[1]}; truncating to common.")
            n_common = min(ref_1yer.shape[0], all_coords.shape[1])
            ref_1yer = ref_1yer[:n_common]
            all_coords_align = all_coords[:, :n_common, :]
        else:
            all_coords_align = all_coords
        feat_rmsd = _rmsd_to_ref(all_coords_align, ref_1yer)
    except FileNotFoundError as e:
        print(f"  [warn] 1YER not available: {e}; RMSD = NaN")
        feat_rmsd = np.full(all_coords.shape[0], np.nan)

    feat_rg     = _radius_of_gyration(all_coords)
    feat_e2e    = _end_to_end(all_coords)
    feat_atp    = _domain_com(all_coords, DOMAIN_CENTRAL_BSHEET,
                                DOMAIN_LID_LOOP)
    feat_lid_rg = _radius_of_gyration(all_coords[:, DOMAIN_LID_LOOP, :])

    features = {
        "Rg_A":          feat_rg,
        "end_to_end_A":  feat_e2e,
        "ATP-pocket_COM_A": feat_atp,
        "lid-loop_Rg_A": feat_lid_rg,
        "RMSD_to_1YER_A": feat_rmsd,
    }

    print("\n" + "=" * 70)
    print("Per-state per-source feature means (Å)")
    print("=" * 70)
    cols = list(features.keys())
    header = "| state | source | n | " + " | ".join(c.replace("_A", "") for c in cols) + " |"
    print(header)
    print("|" + "|".join(["---"] * (len(cols) + 3)) + "|")
    breakdown = []
    for s in range(4):
        for i, sname in enumerate(src_names):
            mask = (hard == s) & (src_idx == i)
            n_in = int(mask.sum())
            row = {"state": s, "source": sname, "n_frames": n_in}
            if n_in > 0:
                cells = []
                for f in cols:
                    m, sd = features[f][mask].mean(), features[f][mask].std()
                    row[f + "_mean"] = float(m)
                    row[f + "_std"] = float(sd)
                    cells.append(f"{m:.1f} ± {sd:.1f}")
                print(f"| {s} | {sname} | {n_in} | " + " | ".join(cells) + " |")
            else:
                print(f"| {s} | {sname} | 0 | " + " | ".join(["—"] * len(cols)) + " |")
            breakdown.append(row)

    # Save JSON.
    json_path = ROOT / "md" / "hsp90_ntd_h3_biology_results.json"
    json_path.write_text(json.dumps({
        "sources": src_names,
        "n_frames_per_source": {s: int(sources[s].shape[0]) for s in src_names},
        "feature_breakdown": breakdown,
        "domain_slices": {
            "CENTRAL_BSHEET": [DOMAIN_CENTRAL_BSHEET.start, DOMAIN_CENTRAL_BSHEET.stop],
            "LID_LOOP":       [DOMAIN_LID_LOOP.start, DOMAIN_LID_LOOP.stop],
            "HELIX2":         [DOMAIN_HELIX2.start, DOMAIN_HELIX2.stop],
        },
    }, indent=2))
    print(f"\nWrote {json_path}")

    # 6-panel figure (matching the Notch1 style).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        panels = list(features.items())
        for ax, (fname, fvals) in zip(axes.flatten(), panels):
            for i, sname in enumerate(src_names):
                src_mask = src_idx == i
                for s in range(4):
                    state_mask = hard == s
                    mask = src_mask & state_mask
                    if mask.sum() < 3:
                        continue
                    ax.scatter([s + 0.15 * i - 0.3] * mask.sum(),
                                fvals[mask],
                                alpha=0.3, s=6, c=colors[i])
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"S{s}" for s in range(4)])
            ax.set_ylabel(fname + " (Å)")
            ax.grid(alpha=0.3)
        # Legend in unused panel.
        axes[1, 2].clear()
        for i, sname in enumerate(src_names):
            axes[1, 2].scatter([], [], c=colors[i], label=sname)
        axes[1, 2].legend(loc="center", fontsize=12)
        axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
        axes[1, 2].set_title("Source colour key")
        fig.suptitle("Hsp90 NTD — per-state per-source features "
                      "(4-source generative-only, v0.7 preliminary)", fontsize=14)
        fig.tight_layout()
        fig_path = FIG_DIR / "hsp90_ntd_h3_biology_preliminary.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"Wrote {fig_path}")
    except ImportError as e:
        print(f"[warn] matplotlib unavailable: {e}")


if __name__ == "__main__":
    main()
