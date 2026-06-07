"""Composite objective for the chimerax-vampnet RSI loop.

Scores an attempt from the metrics the project already pre-registers,
deliberately weighting *validated physical coverage* over raw novelty
so the loop cannot reward-hack by emitting unphysical "new states"
(design rationale: md/rsi_lab_design.md, "anti-reward-hacking").

The score is information-per-dollar: the weighted metric improvement
divided by the run cost. Each component reads a key from `row["result"]`
and is None (ignored) when the attempt did not measure it — backfilled
historical rows have empty result dicts and score None until enriched.

Expected `result` keys (all optional):
  h2_directional_pp   apo-minus-holo auto-inhibition gap, percentage pts
  h2_magnitude_met    bool — did BOTH pre-registered thresholds hold
  h3_unique_states    int  — count of states unique to this source AND
                             validated (see validate.py); raw, un-
                             validated novelty does NOT belong here
  slowest_its_ns      float — slowest implied timescale (vampnet timescales)
  baseline_its_ns     float — the timescale this attempt is improving on
"""

from __future__ import annotations

from ledger import load_ledger

# Component weights. Magnitude (the standing open problem) and validated
# unique coverage are worth the most; timescale growth and directional
# replication less. Tune as the loop runs.
WEIGHTS = {
    "h2_magnitude": 5.0,     # boolean: the hard, unsolved H2 criterion
    "h3_unique": 3.0,        # per validated unique state
    "h2_directional": 1.0,   # per percentage point of apo>holo gap
    "its_growth": 2.0,       # per doubling of slowest implied timescale
}


def score_components(result: dict) -> dict:
    """Return per-component raw contributions (None where unmeasured)."""
    import math

    comp: dict[str, float | None] = {k: None for k in WEIGHTS}

    if result.get("h2_magnitude_met") is not None:
        comp["h2_magnitude"] = WEIGHTS["h2_magnitude"] * (
            1.0 if result["h2_magnitude_met"] else 0.0)

    if result.get("h3_unique_states") is not None:
        comp["h3_unique"] = WEIGHTS["h3_unique"] * float(result["h3_unique_states"])

    if result.get("h2_directional_pp") is not None:
        # Only positive (correct-direction) gaps earn credit.
        comp["h2_directional"] = WEIGHTS["h2_directional"] * max(
            0.0, float(result["h2_directional_pp"]))

    its, base = result.get("slowest_its_ns"), result.get("baseline_its_ns")
    if its is not None and base and base > 0:
        comp["its_growth"] = WEIGHTS["its_growth"] * math.log2(its / base)

    return comp


def score(row: dict) -> dict:
    """Score one ledger row. Returns {raw, per_dollar, components, n_measured}.

    `raw` sums the measured components; `per_dollar` divides by cost
    (the sample-efficiency objective). Both are None when nothing was
    measured — a backfilled historical row scores None until enriched."""
    comp = score_components(row.get("result", {}))
    measured = {k: v for k, v in comp.items() if v is not None}
    if not measured:
        return {"raw": None, "per_dollar": None, "components": comp,
                "n_measured": 0}
    raw = sum(measured.values())
    cost = row.get("cost_usd")
    per_dollar = (raw / cost) if (cost and cost > 0) else None
    return {"raw": raw, "per_dollar": per_dollar, "components": comp,
            "n_measured": len(measured)}


def rank(rows: list[dict] | None = None) -> list[tuple[dict, dict]]:
    """Score and rank all ledger rows by raw score (measured rows first)."""
    rows = rows if rows is not None else load_ledger()
    scored = [(r, score(r)) for r in rows]
    return sorted(scored, key=lambda rs: (rs[1]["raw"] is not None,
                                          rs[1]["raw"] or 0.0), reverse=True)


if __name__ == "__main__":
    measured = [(r, s) for r, s in rank() if s["n_measured"]]
    print(f"{len([r for r in load_ledger()])} rows; "
          f"{len(measured)} have measured outcomes to score\n")
    for r, s in measured:
        pd = f"{s['per_dollar']:.2f}/$" if s["per_dollar"] else "—"
        print(f"  {r['id']:<30s} raw={s['raw']:>6.2f}  {pd}")
    if not measured:
        print("  (no rows scored yet — enrich row['result'] dicts to "
              "activate scoring; backfilled rows start empty by design)")
