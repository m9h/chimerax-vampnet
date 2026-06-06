"""2D FES post-processing for the v0.7 W4c metadynamics on
(NEC-NTM COM, LNR-A → HD-N COM) on Notch1 apo.

Reconstructs the 2D FES via Gaussian-sum from each walker's HILLS
file (pure numpy, no plumed sum_hills dependency), averages
across walkers, and overlays the v0.5 5-source ensemble points on
the same 2D plane for direct visualization of where each sampler
lives relative to the underlying FES.

  $ .venv/bin/python md/metad_2d_fes_postprocess.py

Outputs:
  md/figures/notch1_metad_2d_fes.png — contour overlay
  md/notch1_metad_2d_fes_data.json — backing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "md" / "figures"
KT_310K = 8.314e-3 * 310.0   # kJ/mol
sys.path.insert(0, str(ROOT / "md"))


def _sum_hills_2d(hills_path: Path,
                    d1_grid: np.ndarray, d2_grid: np.ndarray,
                    biasfactor: float = 10.0):
    """Build a 2D Gaussian-sum from a HILLS file.
    HILLS columns: time d1 d2 sigma_d1 sigma_d2 height biasf
    Returns (n_d1, n_d2) bias array in kJ/mol."""
    hills = []
    with open(hills_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            t, d1, d2, sig1, sig2, hh, bf = map(float, parts[:7])
            hills.append((d1, d2, sig1, sig2, hh))
    if not hills:
        raise RuntimeError(f"no hills in {hills_path}")
    print(f"    {len(hills)} Gaussians from {hills_path}")

    # Build the 2D bias grid.
    D1, D2 = np.meshgrid(d1_grid, d2_grid, indexing="ij")
    bias = np.zeros_like(D1)
    for d1, d2, sig1, sig2, hh in hills:
        bias += hh * np.exp(-((D1 - d1) ** 2) / (2 * sig1 ** 2)
                              -((D2 - d2) ** 2) / (2 * sig2 ** 2))
    fes = -bias * biasfactor / (biasfactor - 1.0)
    return fes


def main():
    walker_dirs = [Path(f"/tmp/metad_2d_walker_{w}") for w in [1, 2, 3]]
    walker_dirs = [d for d in walker_dirs if (d / "HILLS").exists()]
    print(f"=== 2D metad FES merge on {len(walker_dirs)} walkers ===")

    d1_grid = np.linspace(0.1, 1.6, 150)   # nm, NEC-NTM COM (mask d1<0.1)
    d2_grid = np.linspace(1.8, 5.0, 150)   # nm, LNR-A → HD-N COM

    fes_list = []
    for d in walker_dirs:
        fes = _sum_hills_2d(d / "HILLS", d1_grid, d2_grid)
        fes -= fes.min()
        fes_list.append(fes)
    fes_arr = np.array(fes_list)
    merged_fes = fes_arr.mean(axis=0)
    merged_fes -= merged_fes.min()
    fes_std = fes_arr.std(axis=0)
    print(f"merged 2D FES: range {merged_fes.min():.1f} to {merged_fes.max():.1f} kJ/mol")

    # Load v0.5 5-source ensemble for point overlay.
    # We need d1 (NEC-NTM only defined for MD) and d2 (LNR-A → HD-N
    # defined for all sources). For each source we compute d2; for MD
    # we additionally compute d1.
    from notch1_h3_multisource import _load_md_apo_nec_ca

    LNR_A_LOCAL = slice(0, 35)
    HD_N_LOCAL  = slice(119, 174)

    def _com_distance(coords, a, b):
        return np.linalg.norm(coords[:, a, :].mean(axis=1)
                                - coords[:, b, :].mean(axis=1), axis=-1)

    ZEN = Path("/data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/"
                "v0.3/notch1_apo_v3")
    # MD: get both chain A NEC and chain B NTM CAs.
    md_nec_list, md_ntm_list = [], []
    for r in range(3):
        d = np.load(ZEN / f"replica_{r}/ca_traj.npz")
        coords = d["coords_A"]
        chains = d["chains"].astype("U2")
        md_nec_list.append(coords[:, chains == "A", :])
        md_ntm_list.append(coords[:, chains == "B", :])
    md_nec = np.concatenate(md_nec_list, 0)
    md_ntm = np.concatenate(md_ntm_list, 0)
    md_d1_A = np.linalg.norm(md_nec.mean(1) - md_ntm.mean(1), axis=-1)  # Å
    md_d2_A = _com_distance(md_nec, LNR_A_LOCAL, HD_N_LOCAL)

    # Generative sources: only d2 available (NEC-only).
    gen_d2 = {}
    for tag, path in [("MarS-FM", "notch1_apo_NEC200_marsfm.npz"),
                       ("BioEmu", "notch1_NEC_bioemu200.npz"),
                       ("Boltz-2", "notch1_NEC_boltz200.npz"),
                       ("AlphaFlow", "notch1_NEC_af200.npz")]:
        p = ROOT / path
        if not p.exists():
            continue
        ca = np.load(p, allow_pickle=True)["coords_ca"].astype(np.float32)
        d2 = _com_distance(ca, LNR_A_LOCAL, HD_N_LOCAL)
        gen_d2[tag] = d2
        print(f"  {tag}: d2 mean={d2.mean():.1f} ± {d2.std():.1f} Å, n={len(d2)}")

    print(f"  MD: d1 mean={md_d1_A.mean():.1f}±{md_d1_A.std():.1f} Å, "
          f"d2 mean={md_d2_A.mean():.1f}±{md_d2_A.std():.1f} Å")

    # Save JSON backing.
    json_path = ROOT / "md" / "notch1_metad_2d_fes_data.json"
    json_path.write_text(json.dumps({
        "system": "notch1_apo_v3",
        "n_walkers": len(walker_dirs),
        "d1_grid_nm": d1_grid.tolist(),
        "d2_grid_nm": d2_grid.tolist(),
        "fes_kjmol":  merged_fes.tolist(),
        "fes_std_kjmol": fes_std.tolist(),
    }, indent=2))
    print(f"wrote {json_path}")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    # Mask out high-FES regions for clean contour plot.
    fes_plot = np.where(merged_fes < 200, merged_fes, np.nan)
    contour = ax.contourf(d1_grid * 10, d2_grid * 10, fes_plot.T,
                            levels=np.arange(0, 161, 10), cmap="viridis_r",
                            extend="max")
    cb = plt.colorbar(contour, ax=ax, label="Free energy (kJ/mol)")
    ax.contour(d1_grid * 10, d2_grid * 10, fes_plot.T,
                levels=[10, 30, 60, 100, 130], colors="white", linewidths=0.5,
                alpha=0.5)

    # Overlay v0.5 5-source ensemble (only MD has d1; generative
    # sources are plotted as marginal histograms along the d2 axis).
    ax.scatter(md_d1_A, md_d2_A, s=3, alpha=0.15, c="red", label="MD apo")

    # Generative d2 as marginal strips at the left edge.
    src_colors = {"MarS-FM": "#2ca02c", "BioEmu": "#d62728",
                    "Boltz-2": "#9467bd", "AlphaFlow": "#8c564b"}
    for i, (tag, d2) in enumerate(gen_d2.items()):
        x_offset = 0.5 + 0.4 * i  # Å, marker on left edge
        ax.scatter([x_offset] * len(d2), d2, s=4, alpha=0.4,
                    c=src_colors.get(tag, "k"), label=f"{tag} (d2 only)")

    ax.axvline(3.94, color="white", ls="--", lw=0.7, label="v0.3 COM restraint (3.94 Å)")
    ax.set_xlabel("d1 = NEC–NTM COM distance (Å)")
    ax.set_ylabel("d2 = LNR-A → HD-N COM distance (Å)")
    ax.set_title(f"Notch1 apo 2D metad FES (v0.7 W4c, {len(walker_dirs)} walkers × ~30 ns)\n"
                  f"MD as scatter; generative samplers shown as d2 marginals at left edge")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 15)
    ax.set_ylim(18, 50)
    fig.tight_layout()
    fig_path = FIG_DIR / "notch1_metad_2d_fes.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
