"""2D landscape projection of the v0.5 5-source ensemble.

Two complementary 2D views:

  (a) MD-only on (NEC-NTM COM, LNR-A → HD-N) -- this is the only
      view where the H2 CV (chain-A vs chain-B COM distance) is
      defined, because the generative sources are NEC-only. Colour
      points by VAMPnet state assignment so we can see whether
      states 0 and 3 are distinguishable in this 2D feature plane.

  (b) All 5 sources on (LNR-A → HD-N, Rg) -- these are CVs that
      every source can report. Colour by source to show the
      v0.6 H3 biology finding in 2D: where in feature space
      each sampler family lives.

Output: md/figures/notch1_h3_2d_landscape.png (two-panel figure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "md" / "figures"
ZEN_ROOT = Path("/data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/"
                 "v0.3/notch1_apo_v3")
sys.path.insert(0, str(ROOT / "md"))

DOMAIN_LNR_A = slice(0, 35)
DOMAIN_HD_N = slice(120, 174)


def _load_md_nec_ntm():
    nec_list, ntm_list = [], []
    for r in range(3):
        d = np.load(ZEN_ROOT / f"replica_{r}/ca_traj.npz")
        chains = d["chains"].astype("U2")
        coords = d["coords_A"]
        nec_list.append(coords[:, chains == "A", :])
        ntm_list.append(coords[:, chains == "B", :])
    return np.concatenate(nec_list, 0), np.concatenate(ntm_list, 0)


def _load_source(path):
    p = ROOT / path
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["coords_ca"].astype(np.float32) if "coords_ca" in d.files else None


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


def _domain_com(coords, a_slice, b_slice):
    com_a = coords[:, a_slice, :].mean(axis=1)
    com_b = coords[:, b_slice, :].mean(axis=1)
    return np.linalg.norm(com_a - com_b, axis=-1)


def _nec_ntm_com(nec, ntm):
    return np.linalg.norm(nec.mean(axis=1) - ntm.mean(axis=1), axis=-1)


def main():
    print("loading sources…")
    nec_md, ntm_md = _load_md_nec_ntm()
    sources = {"MD": nec_md}
    for tag, path in [("MarS-FM", "notch1_apo_NEC200_marsfm.npz"),
                       ("BioEmu", "notch1_NEC_bioemu200.npz"),
                       ("Boltz-2", "notch1_NEC_boltz200.npz"),
                       ("AlphaFlow", "notch1_NEC_af200.npz")]:
        s = _load_source(path)
        if s is not None:
            sources[tag] = s
    src_names = list(sources.keys())
    all_coords = np.concatenate([sources[s] for s in src_names], axis=0)
    src_idx = np.concatenate([np.full(sources[s].shape[0], i, dtype=np.int64)
                                for i, s in enumerate(src_names)])

    print("fitting joint VAMPnet for state assignment colouring…")
    X = _featurize(all_coords)
    hard = _fit_vampnet(X)

    # Compute the 2D CVs.
    lnra_hd = _domain_com(all_coords, DOMAIN_LNR_A, DOMAIN_HD_N)  # Angstroms
    rg = _radius_of_gyration(all_coords)                              # Angstroms

    # MD-only 2D: (NEC-NTM COM, LNR-A → HD-N) coloured by state.
    md_mask = src_idx == src_names.index("MD")
    nec_ntm_md = _nec_ntm_com(nec_md, ntm_md)
    lnra_hd_md = lnra_hd[md_mask]
    hard_md = hard[md_mask]

    # All-source 2D: (LNR-A → HD-N, Rg) coloured by source.
    print("plotting…")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel (a): MD-only by state.
    ax = axes[0]
    state_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for s in range(4):
        m = hard_md == s
        if not m.any():
            continue
        ax.scatter(nec_ntm_md[m], lnra_hd_md[m],
                    s=4, alpha=0.5, c=state_colors[s], label=f"state {s}")
    ax.axvline(3.94, color="k", ls="--", lw=0.7,
                label="v0.3 COM restraint (3.94 Å)")
    ax.axvline(11.0, color="red", ls=":", lw=1.0,
                label="metad barrier (11 Å)")
    ax.set_xlabel("NEC–NTM COM distance (Å)")
    ax.set_ylabel("LNR-A → HD-N COM distance (Å)")
    ax.set_title("(a) MD-only by VAMPnet state\n"
                  "v0.5 H3: states 0+3 = MD equilibrium sub-basins")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 6)

    # Panel (b): All sources by source on (LNR-A → HD-N, Rg).
    ax = axes[1]
    src_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, sname in enumerate(src_names):
        m = src_idx == i
        ax.scatter(lnra_hd[m], rg[m], s=4, alpha=0.4,
                    c=src_colors[i], label=sname)
    ax.set_xlabel("LNR-A → HD-N COM distance (Å)")
    ax.set_ylabel("Radius of gyration of NEC (Å)")
    ax.set_title("(b) All 5 sources by sampler family\n"
                  "v0.6 H3 biology: Boltz-2 compactness bias visible")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("2D landscape: where each sampler lives on the Notch1 NEC "
                  "feature plane", fontsize=14)
    fig.tight_layout()
    fig_path = FIG_DIR / "notch1_h3_2d_landscape.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
