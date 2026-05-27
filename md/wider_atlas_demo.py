"""Run the full ATLAS demo on a curated diverse-fold set.

Sequentially fetches 4 proteins from ATLAS covering all-alpha, mostly-
beta, mixed alpha/beta, and large-multi-domain regimes. For each:
  - downloads the 600 MB protein-only archive (skip if cached)
  - parses .xtc into (N, A, 3) CA coords via mdtraj
  - fits a small VAMPnet
  - records the 4-state populations + slowest implied timescales

Writes a markdown summary table to md/wider_atlas_results.md.

  $ .venv/bin/python md/wider_atlas_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ATLAS_ROOT = Path("/data/datasets/chimerax-vampnet/atlas")
HERE = Path(__file__).resolve().parent

# (pdb_chain, expected fold class)
SET = [
    ("1ail_A", "all-alpha"),
    ("2ppp_A", "mostly-beta"),
    ("1k5n_A", "mixed-alpha/beta"),
    ("4dja_A", "large-mostly-alpha"),
]


def run_demo(pdb_chain: str) -> dict:
    """Fetch (cached) + analyze. Returns the parsed result dict."""
    sysdir = ATLAS_ROOT / f"atlas_{pdb_chain}"
    if not (sysdir / "replica_0" / "traj.xtc").exists():
        print(f"[wider] downloading {pdb_chain}")
        subprocess.run(
            [sys.executable, str(HERE / "atlas_fetch.py"),
             pdb_chain, str(ATLAS_ROOT)],
            check=True,
        )
    else:
        print(f"[wider] {pdb_chain} cached")

    print(f"[wider] analyzing {pdb_chain}")
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, str(HERE / "atlas_demo.py"),
         pdb_chain, "--n-states", "4", "--lag", "50", "--epochs", "40"],
        capture_output=True, text=True, check=True,
    )
    elapsed = time.time() - t0
    log = out.stdout

    # Parse the demo output for the two summary lines.
    pops, its_ns = [], []
    for line in log.splitlines():
        if "state populations:" in line:
            pops = [float(s.rstrip("%")) for s in line.split(":", 1)[1].split(",")]
        elif "implied timescales:" in line:
            its_ns = [float(s.replace("ns", "").strip()) for s in line.split(":", 1)[1].split(",")]

    meta = json.loads((sysdir / "metadata.json").read_text())
    return {
        "pdb_chain": pdb_chain,
        "protein": meta.get("protein_name", "?")[:60],
        "length": meta.get("length"),
        "alpha_pct": meta.get("alpha%"),
        "beta_pct": meta.get("beta%"),
        "rg_A": meta.get("avg_gyration"),
        "rmsf_A": meta.get("avg_RMSF"),
        "populations_pct": pops,
        "implied_timescales_ns": its_ns,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    results = []
    for pdb_chain, fold in SET:
        try:
            r = run_demo(pdb_chain)
            r["expected_fold"] = fold
            results.append(r)
        except subprocess.CalledProcessError as e:
            print(f"[wider] FAIL {pdb_chain}: {e}")
            print(e.stdout, file=sys.stderr)
            print(e.stderr, file=sys.stderr)

    # Write markdown summary.
    out_md = HERE / "wider_atlas_results.md"
    with out_md.open("w") as f:
        f.write("# Wider ATLAS demo — bundle robustness across folds\n\n")
        f.write("Each row is the chimerax-vampnet bundle's full analysis of one\n")
        f.write("public MD trajectory from ATLAS (Vander Meersche et al. 2024) —\n")
        f.write("no MD generation, no manual prep. The fetcher + bundle handle\n")
        f.write("download, parse, feature, fit, and timescale extraction.\n\n")
        f.write("| PDB | Protein | L | α% | β% | Fold | States populations (%) | Slowest IT (ns) | wall (s) |\n")
        f.write("|---|---|---:|---:|---:|---|---|---:|---:|\n")
        for r in results:
            pops = "/".join(f"{p:.0f}" for p in r.get("populations_pct", []))
            its = ", ".join(f"{t:.1f}" for t in r.get("implied_timescales_ns", []))
            f.write(f"| `{r['pdb_chain']}` | {r['protein']} | {r['length']} | "
                    f"{r['alpha_pct']} | {r['beta_pct']} | {r['expected_fold']} | "
                    f"{pops} | {its} | {r['elapsed_s']} |\n")
        f.write("\nRaw results: ")
        f.write(json.dumps(results, indent=2))

    print(f"\n[wider] wrote {out_md}")
    for r in results:
        print(f"  {r['pdb_chain']:>10}: pops={r.get('populations_pct')} "
              f"timescales(ns)={r.get('implied_timescales_ns')}")


if __name__ == "__main__":
    main()
