"""Tests for the RSI lab (md/rsi/) — ledger, backfill parser, objective.

These modules (v0.7.5) shipped without tests. They are stdlib-only
(no torch/numpy), so they exercise fully in CI and locally.
"""

import sys
import tempfile
from pathlib import Path

RSI = Path(__file__).resolve().parent.parent / "md" / "rsi"
sys.path.insert(0, str(RSI))


# ----------------------------------------------------------------- ledger

def test_new_row_defaults_and_override():
    import ledger
    r = ledger.new_row("abc", verdict="met", cost_usd=3.0)
    assert r["id"] == "abc"
    assert r["verdict"] == "met"
    assert r["cost_usd"] == 3.0
    # defaults
    assert r["parent_id"] is None
    assert r["validated"] == {"physicality": None, "committor": None,
                              "bootstrap": None}


def test_new_row_rejects_bad_verdict():
    import ledger
    import pytest
    with pytest.raises(ValueError):
        ledger.new_row("x", verdict="totally-bogus")


def test_ledger_round_trip(tmp_path):
    import ledger
    p = tmp_path / "l.jsonl"
    rows = [ledger.new_row("a", verdict="met", date="2026-01-01"),
            ledger.new_row("b", verdict="blocked", date="2026-01-02")]
    ledger.write_ledger(rows, p)
    back = ledger.load_ledger(p)
    assert [r["id"] for r in back] == ["a", "b"]  # sorted by date
    assert ledger.load_by_id(p)["b"]["verdict"] == "blocked"


def test_append_row_rejects_duplicate(tmp_path):
    import ledger
    import pytest
    p = tmp_path / "l.jsonl"
    ledger.append_row(ledger.new_row("a"), p)
    with pytest.raises(ValueError):
        ledger.append_row(ledger.new_row("a"), p)


def test_upsert_replaces(tmp_path):
    import ledger
    p = tmp_path / "l.jsonl"
    ledger.append_row(ledger.new_row("a", verdict="pending"), p)
    ledger.upsert_row(ledger.new_row("a", verdict="met"), p)
    assert ledger.load_by_id(p)["a"]["verdict"] == "met"
    assert len(ledger.load_ledger(p)) == 1


# ---------------------------------------------------------------- backfill

def test_verdict_prefers_status_line_over_body():
    import backfill
    # The regression that motivated status-scoping: a doc whose Status
    # says MET but whose body uses the word "falsifies" must read 'met'.
    status = "pre-registered H3 prediction **MET** with 5 of 5 sources."
    body = "If it reached state 2 that falsifies the taxonomy. " + status
    assert backfill._parse_verdict(status, body) == "met"


def test_verdict_falls_back_to_body_when_status_silent():
    import backfill
    status = "Phase-1 inference-only API IMPLEMENTED and HONORED."
    body = status + " This is a falsified hypothesis."
    assert backfill._parse_verdict(status, body) == "falsified"


def test_verdict_not_met_and_blocked():
    import backfill
    assert backfill._parse_verdict("magnitudes NOT met; directional met",
                                   "") == "not_met"
    assert backfill._parse_verdict("directional MET. Magnitudes blocked",
                                   "") == "blocked"


def test_parse_costs_prefers_total_line():
    import backfill
    text = ("intermediate ~$0.30 here\n"
            "| **Total v0.3 spend** | | | **~$115** |\n")
    best, mentions = backfill._parse_costs(text)
    assert best == 115.0
    assert "~$0.30" in mentions


def test_parse_costs_no_total_leaves_none():
    import backfill
    best, mentions = backfill._parse_costs("a ~$0.30 run and a $5 run")
    assert best is None
    assert len(mentions) == 2


def test_parse_doc_on_synthetic_file(tmp_path):
    import backfill
    doc = tmp_path / "demo_results.md"
    doc.write_text("# Demo result — v0.9 W2\n\n"
                   "**Date**: 2026-06-07\n"
                   "**Status**: prediction **MET**.\n\n"
                   "Total cost ~$12.\n")
    f = backfill.parse_doc(doc)
    assert f["title"] == "Demo result — v0.9 W2"
    assert f["date"] == "2026-06-07"
    assert f["version"] == "v0.9"
    assert f["verdict"] == "met"
    assert f["source_doc"] == "md/demo_results.md"


# --------------------------------------------------------------- objective

def test_objective_empty_result_scores_none():
    import objective
    s = objective.score({"result": {}, "cost_usd": 5.0})
    assert s["raw"] is None
    assert s["per_dollar"] is None
    assert s["n_measured"] == 0


def test_objective_composite_and_per_dollar():
    import objective
    row = {"cost_usd": 10.0, "result": {
        "h2_magnitude_met": True,      # 5.0
        "h3_unique_states": 2,         # 3.0 * 2 = 6.0
        "h2_directional_pp": 4.0,      # 1.0 * 4 = 4.0
    }}
    s = objective.score(row)
    assert s["raw"] == 5.0 + 6.0 + 4.0
    assert s["per_dollar"] == (15.0 / 10.0)
    assert s["n_measured"] == 3


def test_objective_negative_directional_earns_nothing():
    import objective
    s = objective.score({"cost_usd": 1.0,
                         "result": {"h2_directional_pp": -3.0}})
    # wrong-direction gap clamps to 0, but it WAS measured.
    assert s["components"]["h2_directional"] == 0.0
    assert s["raw"] == 0.0


def test_objective_its_growth_log2():
    import objective
    # doubling the slowest implied timescale -> log2(2)=1 * weight 2 = 2.0
    s = objective.score({"cost_usd": None,
                         "result": {"slowest_its_ns": 200.0,
                                    "baseline_its_ns": 100.0}})
    assert abs(s["raw"] - 2.0) < 1e-9
    assert s["per_dollar"] is None  # no cost -> no per-dollar
