"""MCMC + ensemble diagnostics post-processor for chimerax-vampnet adapters.

A pure-Python, pure-local tool — no Modal, no GPU, no image rebuilds. Operates
on the .npz files emitted by every `md/<method>_modal.py` adapter
(Timewarp, AlphaFlow, BioEmu, Boltz-2, MarS-FM, ESMFold2, UMA, ...) and
emits the canonical Bayesian-MCMC diagnostic suite via arviz:

  ESS    — effective sample size per observable
  R-hat  — Gelman-Rubin convergence (requires multi-chain; warns if absent)
  MCSE   — Monte-Carlo standard error per observable
  autocorr time — integrated autocorrelation per observable
  + pathology flags: low/high acceptance, non-mixing chains, low ESS,
    high R-hat, all-identical samples, high autocorr

Why this exists (the v0.10 W1 motivation): our v0.9 Timewarp adapter
returned 0% MH acceptance on 50,000 proposals — the chain never moved.
A proper diagnostic suite would have surfaced this in seconds (ESS≈1,
identical-frame ratio = 100%) instead of via manual inspection of npz
contents. And the same diagnostics let us quantify the v0.7+v0.8 H3
finding ("generative samplers collapse to modal state") as actual numbers
(BioEmu Rg-std=0.02 -> ESS ≈ 0 on the Rg axis), not just visual scatter.

  # smoke against the v0.9 broken-MH Timewarp output:
  .venv/bin/python md/mcmc_diagnostics.py ad2_timewarp.npz

  # full report w/ trace+rank plots:
  .venv/bin/python md/mcmc_diagnostics.py notch1_NEC_esmfold2200.npz \\
      --plot --out diag_esmfold2.png

  # machine-readable for CI gates:
  .venv/bin/python md/mcmc_diagnostics.py X.npz --json > diag.json

Observable choices: radius of gyration (CA), end-to-end CA-CA distance,
mean adjacent CA-CA distance (chain-quality sanity), per-chain Rg if
chain_id is multi-valued. These are the three coarsest scalar functions
of any backbone ensemble; ESS on these is a sharp lower bound on the
quality of *any* downstream feature (VAMPnet, MSM, H3 column).

Multi-chain detection: if the npz has a `chain_idx` field (one int per
sample, see v0.10 W2 multi-chain-fanout extension), samples are split into
chains and R-hat becomes available. Otherwise treated as a single chain
with a `R-hat: N/A` note.

Thresholds (rule-of-thumb defaults, tunable on the CLI):
  accept_rate < 0.05 or > 0.7  -> WARN
  ESS < 50                      -> WARN
  R-hat > 1.05                  -> WARN  (Stan default split-rhat threshold)
  distinct samples < 1% of N   -> WARN  (catches a stuck chain immediately)
  integrated autocorr > 0.2 N  -> WARN  (insufficient mixing for the run length)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------- observable functions


def radius_of_gyration(coords_ca: np.ndarray) -> np.ndarray:
    """Rg per frame, in whatever length units coords_ca uses (typically Å)."""
    com = coords_ca.mean(axis=1, keepdims=True)  # (N, 1, 3)
    sq = ((coords_ca - com) ** 2).sum(axis=-1)   # (N, n_ca)
    return np.sqrt(sq.mean(axis=-1))             # (N,)


def end_to_end_distance(coords_ca: np.ndarray) -> np.ndarray:
    """CA[0] -> CA[-1] Euclidean distance per frame."""
    return np.linalg.norm(coords_ca[:, -1, :] - coords_ca[:, 0, :], axis=-1)


def mean_adjacent_ca_distance(coords_ca: np.ndarray) -> np.ndarray:
    """Average |CA_{i+1} - CA_i| per frame — a chain-quality sanity number;
    healthy protein backbones cluster tightly around 3.8 Å regardless of fold."""
    diffs = np.linalg.norm(coords_ca[:, 1:, :] - coords_ca[:, :-1, :], axis=-1)
    return diffs.mean(axis=-1)


_ALL_OBSERVABLES = {
    "Rg":             radius_of_gyration,
    "end_to_end":     end_to_end_distance,
    "ca_ca_adjacent": mean_adjacent_ca_distance,
}


def _applicable_observables(coords_ca: np.ndarray) -> dict[str, callable]:
    """Filter to observables that are meaningful for this system's CA count.
    A single-CA system (e.g. alanine dipeptide) has no end-to-end and no
    adjacent CA-CA — only Rg (trivially 0 in that case, but defined)."""
    n_ca = coords_ca.shape[1]
    out = {"Rg": _ALL_OBSERVABLES["Rg"]}
    if n_ca >= 2:
        out["end_to_end"] = _ALL_OBSERVABLES["end_to_end"]
        out["ca_ca_adjacent"] = _ALL_OBSERVABLES["ca_ca_adjacent"]
    return out


def compute_observables(coords_ca: np.ndarray) -> dict[str, np.ndarray]:
    return {name: fn(coords_ca) for name, fn in _applicable_observables(coords_ca).items()}


# ------------------------------------------------------------ npz -> arviz idata


def _split_into_chains(values_per_sample: dict[str, np.ndarray],
                       chain_idx: np.ndarray | None
                       ) -> dict[str, np.ndarray]:
    """Reshape each observable from (N,) -> (n_chains, n_draws).

    If chain_idx is provided, sort each chain to its own row. Otherwise
    treat the whole sequence as a single chain. arviz expects (chain, draw)
    layout per variable."""
    if chain_idx is None:
        return {k: v[np.newaxis, :] for k, v in values_per_sample.items()}
    chains = np.unique(chain_idx)
    # Pad short chains to the max length with NaN so arviz can ingest.
    counts = [int((chain_idx == c).sum()) for c in chains]
    n_draws = max(counts)
    out = {}
    for k, v in values_per_sample.items():
        m = np.full((len(chains), n_draws), np.nan, dtype=v.dtype)
        for i, c in enumerate(chains):
            row = v[chain_idx == c]
            m[i, : len(row)] = row
        out[k] = m
    return out


def npz_to_inference_data(path: str | Path):
    """Load an adapter npz and convert to arviz.InferenceData.

    Returns (idata, npz_dict) — npz_dict carries through the source fields
    (accept_rate, elapsed_seconds, plddt, ...) that arviz wouldn\\'t hold
    natively but that flag_pathologies wants.
    """
    import arviz as az
    d = np.load(str(path), allow_pickle=True)
    files = set(d.files)

    if "coords_ca" not in files:
        raise KeyError(f"npz {path} is missing 'coords_ca' — not an adapter output?")
    coords_ca = d["coords_ca"]
    if coords_ca.ndim != 3:
        raise ValueError(f"coords_ca must be (N, n_ca, 3); got {coords_ca.shape}")

    observables = compute_observables(coords_ca)
    chain_idx = d["chain_idx"] if "chain_idx" in files else None
    posterior = _split_into_chains(observables, chain_idx)

    idata = az.from_dict(posterior=posterior)

    npz_dict = {k: d[k] for k in files}
    return idata, npz_dict


# ---------------------------------------------------------------- summarisation


def summarize(idata, npz_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute ESS / R-hat / MCSE / autocorr per observable + scalar adapter
    metadata (accept_rate etc). Returns a dict suitable for printing or JSON."""
    import arviz as az
    posterior = idata.posterior
    n_chains = int(posterior.sizes["chain"])
    n_draws = int(posterior.sizes["draw"])
    n_total = n_chains * n_draws

    ess = az.ess(idata).to_dict()
    mcse = az.mcse(idata).to_dict()
    summary_table = az.summary(idata, kind="stats").to_dict()

    rhat = az.rhat(idata).to_dict() if n_chains >= 2 else None

    out: dict[str, Any] = {
        "n_chains": n_chains, "n_draws": n_draws, "n_total": n_total,
        "observables": {},
    }
    # Iterate the observables actually present in the posterior (not the
    # full _ALL_OBSERVABLES catalogue — single-CA systems drop some).
    for name in list(posterior.data_vars):
        per = {
            "mean": float(summary_table["data_vars"][name]["data"][0])
                    if isinstance(summary_table.get("data_vars"), dict)
                    else float(posterior[name].mean()),
            "std": float(posterior[name].std()),
            "ess": float(ess["data_vars"][name]["data"]) if name in ess["data_vars"] else float("nan"),
            "mcse": float(mcse["data_vars"][name]["data"]) if name in mcse["data_vars"] else float("nan"),
            "autocorr_int": float(_integrated_autocorr(np.asarray(posterior[name]).ravel())),
        }
        if rhat is not None and name in rhat["data_vars"]:
            per["rhat"] = float(rhat["data_vars"][name]["data"])
        out["observables"][name] = per

    # Sample-diversity check — catches the v0.9 Timewarp 0-accept failure
    # directly, regardless of what arviz reports.
    coords_ca = None
    if "coords_ca" in npz_dict:
        cc = npz_dict["coords_ca"]
        if cc.ndim == 3:
            coords_ca = cc
    if coords_ca is not None and len(coords_ca) > 1:
        diffs = np.abs(np.diff(coords_ca, axis=0)).mean(axis=(1, 2))
        n_distinct = int((diffs > 1e-6).sum())
        out["distinct_frame_pairs"] = n_distinct
        out["distinct_fraction"] = n_distinct / len(diffs)

    if "accept_rate" in npz_dict:
        out["accept_rate"] = float(npz_dict["accept_rate"])
    if "elapsed_seconds" in npz_dict:
        out["elapsed_seconds"] = float(npz_dict["elapsed_seconds"])
    # IS-mode adapter outputs (md/timewarp_modal.py IS branch and any
    # future flow-based IS sampler) carry extra weight diagnostics.
    if "method" in npz_dict:
        out["method"] = str(npz_dict["method"])
    for k in ("ess_is", "log_evidence", "temperature_K"):
        if k in npz_dict:
            out[k] = float(npz_dict[k])
    if "plddt" in npz_dict:
        plddt = np.asarray(npz_dict["plddt"]).astype(float)
        plddt = plddt[~np.isnan(plddt)]
        if plddt.size:
            out["plddt_mean"] = float(plddt.mean())
            out["plddt_min"] = float(plddt.min())

    return out


# ------------------------------------------------------------- pathology rules


def flag_pathologies(summary: dict[str, Any],
                     ess_min: float = 50.0, rhat_max: float = 1.05,
                     accept_lo: float = 0.05, accept_hi: float = 0.7,
                     distinct_frac_min: float = 0.01,
                     autocorr_frac_max: float = 0.2) -> list[str]:
    """Rule-based warnings. Each entry is one flag line, prefixed with
    severity tag ('WARN' or 'FAIL'). FAIL means the chain is unusable
    as-is; WARN means it samples but with caveats."""
    warns: list[str] = []
    n = summary["n_total"]

    if "accept_rate" in summary:
        a = summary["accept_rate"]
        method = summary.get("method", "mh")
        if method == "is":
            # accept_rate for IS-mode outputs is ESS_IS / N; the fix is
            # different (flow-vs-Boltzmann mismatch, not chain mixing).
            if a < accept_lo:
                warns.append(
                    f"FAIL ESS_IS/N={a:.3f} < {accept_lo}; IS weights "
                    f"are pathologically skewed — flow proposals don't "
                    f"match Boltzmann target. Likely causes: units bug "
                    f"(coords scale mismatch), wrong topology/force "
                    f"field, or model weights not loaded correctly."
                )
        else:  # mh (or unspecified, treat as mh)
            if a < accept_lo:
                warns.append(
                    f"FAIL accept_rate={a:.3f} < {accept_lo}; MH chain is "
                    f"stuck (try method='is' for diagnostic clarity, or "
                    f"longer num_proposal_steps if proposals are merely "
                    f"too narrow)"
                )
            elif a > accept_hi:
                warns.append(
                    f"WARN accept_rate={a:.3f} > {accept_hi}; proposals "
                    f"are too local (consider higher num_proposal_steps)"
                )

    if "distinct_fraction" in summary and summary["distinct_fraction"] < distinct_frac_min:
        warns.append(
            f"FAIL only {summary['distinct_frame_pairs']}/{n-1} consecutive "
            f"frame pairs differ; chain produced (near-)identical samples"
        )

    for name, per in summary["observables"].items():
        if per["ess"] < ess_min:
            warns.append(
                f"WARN observable={name}: ESS={per['ess']:.1f} < {ess_min}; "
                f"too few effectively-independent samples — extend the run"
            )
        if "rhat" in per and per["rhat"] > rhat_max:
            warns.append(
                f"WARN observable={name}: R-hat={per['rhat']:.3f} > {rhat_max}; "
                f"chains haven't converged on this variable"
            )
        if per["autocorr_int"] > autocorr_frac_max * n:
            warns.append(
                f"WARN observable={name}: integrated autocorr "
                f"{per['autocorr_int']:.1f} > {autocorr_frac_max:.0%} of "
                f"chain length; samples are highly correlated"
            )

    if "rhat" not in next(iter(summary["observables"].values()), {}):
        warns.append(
            "INFO single chain detected (no chain_idx field); R-hat "
            "unavailable. Fanout >=2 independent chains for convergence "
            "diagnostics (see v0.10 W2)."
        )

    return warns


# -------------------------------------------------- standalone autocorr helper


def _integrated_autocorr(x: np.ndarray) -> float:
    """FFT-based integrated autocorrelation time; truncated at first
    negative lag. Canonical home for the project's autocorr helper —
    centralized here in v0.10 (was previously duplicated in
    md/timewarp_modal.py)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return 1.0
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    neg = np.where(acf < 0)[0]
    cutoff = int(neg[0]) if len(neg) else min(n, 100)
    return 1.0 + 2.0 * float(acf[1:cutoff].sum())


# ----------------------------------------------------------- formatting + CLI


def format_report(summary: dict[str, Any], warns: list[str]) -> str:
    """Human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"chains: {summary['n_chains']}    "
                 f"draws/chain: {summary['n_draws']}    "
                 f"total: {summary['n_total']}")
    for k in ("method", "accept_rate", "distinct_fraction",
              "ess_is", "log_evidence", "temperature_K",
              "plddt_mean", "elapsed_seconds"):
        if k in summary:
            v = summary[k]
            if isinstance(v, str):
                lines.append(f"{k:>20s}: {v}")
            else:
                lines.append(f"{k:>20s}: {v:.4g}")
    lines.append("")
    lines.append(f"{'observable':>16s}  {'mean':>10s} {'std':>10s} "
                 f"{'ess':>8s} {'rhat':>6s} {'mcse':>10s} {'τ_int':>8s}")
    lines.append("-" * 72)
    for name, per in summary["observables"].items():
        rhat_str = f"{per['rhat']:.3f}" if "rhat" in per else "  N/A"
        lines.append(
            f"{name:>16s}  {per['mean']:>10.3f} {per['std']:>10.3f} "
            f"{per['ess']:>8.1f} {rhat_str:>6s} {per['mcse']:>10.3g} "
            f"{per['autocorr_int']:>8.2f}"
        )
    lines.append("")
    if warns:
        lines.append("PATHOLOGIES:")
        for w in warns:
            lines.append(f"  {w}")
    else:
        lines.append("PATHOLOGIES: none (all thresholds passed)")
    lines.append("=" * 72)
    return "\n".join(lines)


def report(path: str | Path, plot_path: str | Path | None = None,
            as_json: bool = False) -> dict[str, Any]:
    """Top-level convenience: load + summarise + warn + optional plot."""
    idata, npz_dict = npz_to_inference_data(path)
    summary = summarize(idata, npz_dict)
    warns = flag_pathologies(summary)
    summary["pathologies"] = warns

    if as_json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(format_report(summary, warns))

    if plot_path is not None:
        import arviz as az
        import matplotlib.pyplot as plt
        n_obs = len(summary["observables"])
        fig, axes = plt.subplots(n_obs, 2, figsize=(10, 3 * n_obs))
        # arviz handles single-chain by producing one trace line; multi-chain
        # produces a line per chain on the same axes.
        az.plot_trace(idata, axes=axes)
        fig.suptitle(f"{Path(path).name} — MCMC trace + posterior")
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {plot_path}")

    return summary


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="Path to an adapter-emitted .npz")
    p.add_argument("--plot", action="store_true",
                    help="Write trace+posterior plot to --out (default <path>.diag.png)")
    p.add_argument("--out", default=None, help="Output PNG path for --plot")
    p.add_argument("--json", action="store_true",
                    help="Emit JSON-only output for CI gates")
    args = p.parse_args(argv)

    plot_path = None
    if args.plot:
        plot_path = args.out or (str(Path(args.path).with_suffix(".diag.png")))

    summary = report(args.path, plot_path=plot_path, as_json=args.json)
    # Exit non-zero on any FAIL flag so CI can gate on this.
    return 1 if any(w.startswith("FAIL") for w in summary["pathologies"]) else 0


if __name__ == "__main__":
    sys.exit(_main())
