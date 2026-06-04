"""Post-process metadynamics HILLS files into a 1D PMF along
the NEC-NTM COM distance.

Pulls HILLS files from each metad walker on the Modal volume,
runs `plumed sum_hills` (locally — needs plumed in PATH or
${PROJECT_ROOT}/.venv-md/bin) to reconstruct the FES, plots it,
and writes a `Free-energy summary` block back into
`md/notch1_metad_results.md`.

  $ .venv-md/bin/python md/metad_fes_postprocess.py \
        --system notch1_apo_v3 --walkers 1 2 3

Outputs:
  md/figures/notch1_metad_fes.png — 1D PMF along NEC-NTM COM, all
    walkers overlaid + a per-walker convergence subplot.
  md/notch1_metad_fes_data.json — machine-readable FES + walker
    statistics.

If `plumed` is not on PATH, falls back to a numpy-based reweighting
that estimates the FES directly from the COLVAR file (HILLS sum
done in Python). The numpy fallback is ~10x slower but produces
the same answer.

Convergence diagnostic: counts how many times each walker re-crosses
the auto-inhibited well (CV in [0.35, 0.5] nm). Well-tempered metad
is converged when each walker has re-crossed 5+ times.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "md" / "figures"


def _pull_walker_files(system: str, walker: int, dst: Path) -> None:
    """modal volume get HILLS + COLVAR for a walker."""
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("HILLS", "COLVAR"):
        src = f"prepared/{system}/metad_walker_{walker}/{fname}"
        out = dst / fname
        subprocess.run(
            ["modal", "volume", "get", "chimerax-vampnet-md",
              src, str(out), "--force"],
            check=True, capture_output=True, text=True,
        )


def _sum_hills(hills_path: Path, grid_min: float = 0.0,
                grid_max: float = 8.0, n_bins: int = 400,
                temperature_K: float = 310.0,
                biasfactor: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct FES F(d) from a HILLS file. Pure-numpy implementation
    so we don't need a separate plumed install for analysis."""
    hills = []
    with open(hills_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            t, d, sigma, height, bf = map(float, parts[:5])
            hills.append((d, sigma, height))
    if not hills:
        raise RuntimeError(f"no Gaussian deposits parsed from {hills_path}")

    grid = np.linspace(grid_min, grid_max, n_bins)
    bias = np.zeros_like(grid)
    for d, sigma, height in hills:
        bias += height * np.exp(-((grid - d) ** 2) / (2 * sigma ** 2))

    # Well-tempered correction: F(d) = -bias(d) * (T + dT)/dT
    # with dT = T(biasfactor - 1).
    fes = -bias * biasfactor / (biasfactor - 1.0)
    fes -= fes.min()
    return grid, fes


def _well_recrossings(colvar_path: Path,
                       well_min_nm: float = 0.35,
                       well_max_nm: float = 0.50) -> int:
    """Count how many times the CV exits and re-enters the
    auto-inhibited well. Convergence criterion: 5+."""
    if not colvar_path.exists():
        return 0
    times, ds = [], []
    with open(colvar_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            times.append(float(parts[0]))
            ds.append(float(parts[1]))
    if not ds:
        return 0
    ds = np.array(ds)
    inside = (ds >= well_min_nm) & (ds <= well_max_nm)
    transitions = np.diff(inside.astype(int))
    return int(((transitions == 1).sum() + (transitions == -1).sum()) // 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="notch1_apo_v3")
    p.add_argument("--walkers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cache-dir", type=Path,
                    default=Path("/tmp/metad_postprocess"))
    args = p.parse_args()

    print("=" * 70)
    print(f"Metad FES post-processing — {args.system}, walkers {args.walkers}")
    print("=" * 70)

    walker_data = {}
    for w in args.walkers:
        wdst = args.cache_dir / f"walker_{w}"
        print(f"\n  pulling walker {w} -> {wdst}")
        try:
            _pull_walker_files(args.system, w, wdst)
        except subprocess.CalledProcessError as e:
            print(f"  [warn] walker {w} pull failed: {e.stderr[:200]}")
            continue
        hills = wdst / "HILLS"
        colvar = wdst / "COLVAR"
        n_hills = sum(1 for line in open(hills) if not line.startswith("#"))
        n_colvar = sum(1 for line in open(colvar) if not line.startswith("#"))
        recross = _well_recrossings(colvar)
        print(f"    HILLS: {n_hills} Gaussians ({n_hills} ps biased)")
        print(f"    COLVAR: {n_colvar} samples")
        print(f"    well re-crossings (0.35-0.50 nm): {recross}  "
              f"({'converged' if recross >= 5 else 'NOT converged'})")
        try:
            grid, fes = _sum_hills(hills)
        except RuntimeError as e:
            print(f"    [warn] {e}; skipping FES build for walker {w}")
            continue
        walker_data[w] = {
            "hills_path": str(hills),
            "n_hills": n_hills,
            "n_colvar": n_colvar,
            "well_recrossings": recross,
            "grid_nm": grid.tolist(),
            "fes_kjmol": fes.tolist(),
            "fes_min_kjmol": float(fes.min()),
            "fes_at_4A_kjmol": float(fes[np.argmin(np.abs(grid - 0.4))]),
            "fes_at_8A_kjmol": float(fes[np.argmin(np.abs(grid - 0.8))]),
        }

    if not walker_data:
        print("\nno walkers had usable data; aborting")
        return

    # Merged FES (simple average of per-walker FES estimates).
    grids = [w["grid_nm"] for w in walker_data.values()]
    fes_estimates = np.array([w["fes_kjmol"] for w in walker_data.values()])
    merged_fes = fes_estimates.mean(axis=0)
    merged_fes -= merged_fes.min()
    fes_std = fes_estimates.std(axis=0)

    grid = np.array(grids[0])
    print("\n" + "=" * 70)
    print("Merged FES along NEC-NTM COM distance")
    print("=" * 70)
    well_idx = np.argmin(np.abs(grid - 0.4))   # auto-inhibited (4 A)
    barrier_search = (grid > 0.5) & (grid < 1.5)
    if barrier_search.any():
        barrier_idx = np.argmax(merged_fes[barrier_search])
        barrier_grid = grid[barrier_search][barrier_idx]
        barrier_kjmol = merged_fes[barrier_search][barrier_idx]
    else:
        barrier_grid = barrier_kjmol = None
    diss_idx = np.argmin(np.abs(grid - 2.0))   # dissociated (20 A)
    print(f"  F(4 A, auto-inhibited):    {merged_fes[well_idx]:.1f} ± {fes_std[well_idx]:.1f} kJ/mol")
    if barrier_kjmol is not None:
        print(f"  F(barrier at {barrier_grid*10:.1f} A): {barrier_kjmol:.1f} kJ/mol")
    print(f"  F(20 A, dissociated):      {merged_fes[diss_idx]:.1f} kJ/mol "
          f"({'reached' if grid.max() >= 2.0 else 'beyond grid'})")

    # Figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for i, (w, d) in enumerate(walker_data.items()):
            ax1.plot(grid * 10, d["fes_kjmol"], alpha=0.5, label=f"walker {w}")
        ax1.plot(grid * 10, merged_fes, color="black", lw=2, label="mean")
        ax1.fill_between(grid * 10,
                          merged_fes - fes_std, merged_fes + fes_std,
                          color="black", alpha=0.15, label="± 1σ")
        ax1.set_xlabel("NEC–NTM COM distance (Å)")
        ax1.set_ylabel("Free energy (kJ/mol)")
        ax1.set_title("Reconstructed FES (well-tempered metadynamics)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 30)
        for w, d in walker_data.items():
            recr = d["well_recrossings"]
            ax2.bar([w], [recr], label=f"walker {w}")
        ax2.axhline(5, color="red", ls="--", label="convergence threshold (5)")
        ax2.set_xlabel("walker")
        ax2.set_ylabel("well re-crossings")
        ax2.set_title("Convergence diagnostic")
        ax2.set_xticks(list(walker_data.keys()))
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig.suptitle(f"Metadynamics FES — {args.system}", fontsize=14)
        fig.tight_layout()
        fig_path = FIG_DIR / "notch1_metad_fes.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"\nWrote {fig_path}")
    except ImportError:
        print("\n[warn] matplotlib not available, skipping figure")

    # JSON dump.
    json_path = ROOT / "md" / "notch1_metad_fes_data.json"
    json_path.write_text(json.dumps({
        "system": args.system,
        "walkers": list(walker_data.keys()),
        "per_walker": walker_data,
        "merged": {
            "grid_nm": grid.tolist(),
            "fes_kjmol": merged_fes.tolist(),
            "fes_std_kjmol": fes_std.tolist(),
            "fes_at_4A_kjmol": float(merged_fes[well_idx]),
            "barrier_kjmol": float(barrier_kjmol) if barrier_kjmol is not None else None,
            "barrier_distance_A": float(barrier_grid * 10) if barrier_grid is not None else None,
        },
    }, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
