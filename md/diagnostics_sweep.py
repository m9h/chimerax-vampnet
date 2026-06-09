"""One-shot diagnostic sweep over every adapter npz in the working directory.

Cross-system summary of all .npz files: prints a one-line-per-file table
with method, sample count, accept_rate or ESS_IS/N, distinct-fraction,
Rg mean±std, and pathology counts. The Rg std column is the quantitative
form of the v0.7+v0.8 H3 "generative collapse" finding — per-source
shell width on the same target system.

  .venv/bin/python md/diagnostics_sweep.py              # all *.npz in cwd
  .venv/bin/python md/diagnostics_sweep.py *.npz        # explicit list
  .venv/bin/python md/diagnostics_sweep.py --dir paper-data/ensembles/

Output is fixed-width text suitable for committing into paper supplements
or for piping into `column -t` for further alignment. Files with parsing
errors (non-adapter npz, missing coords_ca, etc.) print "!!" lines so
they don't silently drop from the report.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from mcmc_diagnostics import flag_pathologies, npz_to_inference_data, summarize


def sweep_one(path: str) -> str:
    try:
        idata, npz_dict = npz_to_inference_data(path)
        summary = summarize(idata, npz_dict)
        warns = flag_pathologies(summary)
    except Exception as e:
        return f"!! diagnostic failed: {e}"

    n = summary["n_total"]
    method = summary.get("method", "-")
    ar = summary.get("accept_rate", float("nan"))
    df = summary.get("distinct_fraction", float("nan"))
    ess_is = summary.get("ess_is", "-")
    fails = sum(1 for p in warns if p.startswith("FAIL"))
    warn_count = sum(1 for p in warns if p.startswith("WARN"))
    rg = summary["observables"].get("Rg", {})
    rg_str = f"Rg={rg.get('mean', 0):.2f}±{rg.get('std', 0):.2f}"
    return (f"n={n:>4}  method={method:>4}  ar={ar:>6.3f}  "
            f"distinct={df:>4.2f}  ess_is={ess_is:<4}  {rg_str:<20}  "
            f"FAIL×{fails} WARN×{warn_count}")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="*",
                    help="One or more npz paths or glob patterns (default: *.npz)")
    p.add_argument("--dir", default=".",
                    help="Directory to search if no positional args given")
    args = p.parse_args(argv)

    if args.paths:
        # Expand any glob patterns the shell didn't.
        expanded: list[str] = []
        for p_ in args.paths:
            hits = glob.glob(p_)
            expanded.extend(hits if hits else [p_])
        paths = sorted(set(expanded))
    else:
        paths = sorted(glob.glob(str(Path(args.dir) / "*.npz")))

    if not paths:
        print(f"[sweep] no npz files found", file=sys.stderr)
        return 1

    width = max(len(Path(p).name) for p in paths) + 2
    for path in paths:
        line = sweep_one(path)
        print(f"{Path(path).name:<{width}s} {line}")
    return 0


if __name__ == "__main__":
    # Allow `python md/diagnostics_sweep.py` from any cwd to find sibling module.
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(_main())
