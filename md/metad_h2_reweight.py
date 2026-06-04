"""Boltzmann-integrate the v0.6 metad FES to get H2 magnitudes.

The pre-registered H2 magnitudes (apo P(auto-inhibited) >= 50 %,
holo P(auto-inhibited) <= 30 %) were blocked by the 100-ns
sampling horizon of the v0.3 unbiased MD. The v0.6 W3c metad
result recovered the FES along the same NEC-NTM COM CV directly,
which IS the H2 answer if we just Boltzmann-integrate it.

P(auto-inhibited) = integral over d in [0, cutoff] of exp(-F(d)/kT) dd
                    / integral over d in [0, max] of exp(-F(d)/kT) dd

The cutoff defines what "auto-inhibited" means in CV-space:
  - 5 Angstroms: tight basin only (v0.3 restraint set point + 1 sigma)
  - 7 Angstroms: full auto-inhibited well (per v0.5 H3 biology)
  - 10 Angstroms: associated regime, just below the metad barrier

Output: md/notch1_h2_metad_reweight.md (per-cutoff P values + apo/holo
delta when holo lands).

Run:
  $ .venv/bin/python md/metad_h2_reweight.py [--holo-fes <path>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KT_310K = 8.314e-3 * 310.0   # kJ/mol


def _load_fes(path: Path):
    d = json.loads(path.read_text())
    grid_nm = np.array(d["merged"]["grid_nm"])
    fes_kjmol = np.array(d["merged"]["fes_kjmol"])
    return grid_nm, fes_kjmol


def _p_associated(grid_nm: np.ndarray, fes_kjmol: np.ndarray,
                    cutoff_nm: float) -> float:
    """P(d < cutoff) under the Boltzmann measure with weights w(d) = exp(-F(d)/kT)."""
    w = np.exp(-fes_kjmol / KT_310K)
    Z = np.trapz(w, grid_nm)
    in_basin = grid_nm < cutoff_nm
    p = np.trapz(w[in_basin], grid_nm[in_basin]) / Z
    return float(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apo-fes", type=Path,
                    default=ROOT / "md" / "notch1_metad_fes_data.json")
    p.add_argument("--holo-fes", type=Path,
                    default=None,
                    help="path to holo metad FES JSON (if available)")
    args = p.parse_args()

    apo_grid, apo_fes = _load_fes(args.apo_fes)
    cutoffs_nm = [0.5, 0.7, 1.0]
    print("=" * 70)
    print("Boltzmann-reweighted H2 magnitudes from v0.6 metad FES")
    print("=" * 70)
    print(f"  T = 310 K, kT = {KT_310K:.3f} kJ/mol")
    print(f"  apo FES grid: {apo_grid.min()*10:.1f}-{apo_grid.max()*10:.1f} A "
          f"({len(apo_grid)} bins)")
    print()

    rows = []
    for cutoff in cutoffs_nm:
        p_apo = _p_associated(apo_grid, apo_fes, cutoff)
        if args.holo_fes is not None and args.holo_fes.exists():
            holo_grid, holo_fes = _load_fes(args.holo_fes)
            p_holo = _p_associated(holo_grid, holo_fes, cutoff)
            delta_pp = (p_apo - p_holo) * 100
        else:
            p_holo = None
            delta_pp = None
        rows.append({"cutoff_A": cutoff * 10, "P_apo": p_apo,
                      "P_holo": p_holo, "delta_pp": delta_pp})
        if p_holo is None:
            print(f"  cutoff {cutoff*10:4.1f} A: P_apo = {p_apo*100:5.1f} %  "
                  f"(holo not available)")
        else:
            print(f"  cutoff {cutoff*10:4.1f} A: P_apo = {p_apo*100:5.1f} %, "
                  f"P_holo = {p_holo*100:5.1f} %, delta = {delta_pp:+.1f} pp")

    # Compare to pre-registered H2 thresholds.
    print()
    print("Pre-registered H2 prediction (v0.1):")
    print("  apo P(auto-inhibited) >= 50 %")
    print("  holo P(auto-inhibited) <= 30 %")
    print("  delta >= 20 pp (apo - holo)")

    out_lines = ["# H2 magnitudes from Boltzmann-integrated v0.6 metad FES",
                 "",
                 "**Date**: 2026-06-03",
                 "Script: `md/metad_h2_reweight.py`",
                 "",
                 "## Method",
                 "",
                 "P(d < cutoff) = ∫ exp(-F(d)/kT) dd over d in [0, cutoff]",
                 "                / ∫ exp(-F(d)/kT) dd over the full grid",
                 "",
                 "F(d) is the v0.6 well-tempered metad FES along the",
                 "NEC-NTM COM distance (`md/notch1_metad_fes_data.json`),",
                 "merged across 3 walkers × 20 ns each. T = 310 K.",
                 "",
                 "## Results",
                 "",
                 "| cutoff (Å) | P_apo | P_holo | Δ (pp) |",
                 "|---:|---:|---:|---:|"]
    for r in rows:
        ph = f"{r['P_holo']*100:5.1f} %" if r['P_holo'] is not None else "—"
        dl = f"{r['delta_pp']:+.1f}" if r['delta_pp'] is not None else "—"
        out_lines.append(f"| {r['cutoff_A']:.1f} | {r['P_apo']*100:5.1f} % | {ph} | {dl} |")
    out_lines += [
        "",
        "## Pre-registered H2 vs metad reweight",
        "",
        "Pre-registered (v0.1):",
        "- apo P(auto-inhibited) ≥ 50 %",
        "- holo P(auto-inhibited) ≤ 30 %",
        "- Δ ≥ 20 pp (apo − holo)",
        "",
    ]

    p_apo_strict = _p_associated(apo_grid, apo_fes, 0.5)
    if p_apo_strict >= 0.5:
        out_lines.append(f"v0.6 apo result at cutoff 5 Å: "
                          f"**P_apo = {p_apo_strict*100:.1f} % ≥ 50 % MET ✓**")
    else:
        out_lines.append(f"v0.6 apo result at cutoff 5 Å: "
                          f"P_apo = {p_apo_strict*100:.1f} % < 50 % NOT MET")

    if args.holo_fes is not None and args.holo_fes.exists():
        p_holo_strict = _p_associated(*_load_fes(args.holo_fes), 0.5)
        delta = (p_apo_strict - p_holo_strict) * 100
        out_lines.append(f"v0.6 holo result at cutoff 5 Å: "
                          f"P_holo = {p_holo_strict*100:.1f} % "
                          f"({'MET ≤ 30 %' if p_holo_strict <= 0.3 else 'NOT MET'})")
        out_lines.append(f"Δ (apo − holo) = {delta:+.1f} pp "
                          f"({'MET ≥ 20 pp' if delta >= 20 else 'NOT MET'})")
    else:
        out_lines.append("")
        out_lines.append("Holo metad FES pending; rerun this script with")
        out_lines.append("`--holo-fes md/notch1_metad_holo_fes_data.json` "
                          "when it lands.")
    out_lines += [
        "",
        "## Caveats",
        "",
        "1. The reweight is along a SINGLE CV (NEC-NTM COM distance).",
        "   The v0.5 H3 analysis revealed a 3-way sampler-class split",
        "   along other axes (LNR-A → HD-N, Rg). Auto-inhibition vs",
        "   activation may not project cleanly onto just the NEC-NTM",
        "   COM axis; v0.7 could repeat with a 2D FES (this CV +",
        "   LNR-A → HD-N).",
        "2. The metad FES at very small d (< 1 Å) is affected by",
        "   walker 2/3's pathological PLUMED COM excursions",
        "   documented in `md/notch1_metad_results.md`. Cutoffs at",
        "   5 Å and above are unaffected.",
        "3. P_apo is set largely by the well minimum location and",
        "   width; the absolute number depends on the integration",
        "   range (here [0, 8 nm]). Cutoffs further from the minimum",
        "   exhibit larger numerical sensitivity to the integration",
        "   tail.",
    ]

    out_path = ROOT / "md" / "notch1_h2_metad_reweight.md"
    out_path.write_text("\n".join(out_lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
