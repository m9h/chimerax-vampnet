"""Append-only experiment ledger for the chimerax-vampnet RSI lab.

The ledger is the archive + lineage that the recursive-self-improvement
loop reads and writes (design: md/rsi_lab_design.md). One JSONL row per
experiment attempt. `parent_id` makes it a *lineage* (Darwin-Gödel
style) rather than a flat log: each attempt descends from the config it
mutated.

Schema (all rows):

  id            unique slug, e.g. "v0.7.5-esmfold2-notch1"
  parent_id     id of the attempt this one descends from (None for roots)
  title         human-readable one-liner
  date          ISO date the attempt was run / recorded
  version       project version tag parsed from the title (e.g. "v0.7")
  hypothesis    what the attempt expected to show (pre-registered)
  prereg_metric the metric + threshold, written BEFORE the run
  config        dict describing how to reproduce (adapter, system, ...)
  cost_usd      best-effort dollar cost (None if unknown)
  cost_mentions all dollar figures found in the source doc (transparency)
  result        dict of measured outcomes (None/empty until run)
  verdict       met | not_met | falsified | passed | blocked | pending
  validated     {physicality, committor, bootstrap} gate verdicts
  lesson        structured lesson extracted from the attempt
  tags          free-text tags (hypothesis ids, system, sampler, ...)
  source_doc    path to the md/*_results.md this row was backfilled from

Backfill (md/rsi/backfill.py) is the ONE writer permitted to refresh
the doc-derived fields (title/date/version/cost*/verdict/source_doc).
Human- or agent-authored fields (parent_id/hypothesis/prereg_metric/
lesson/validated/config/result) are preserved across re-backfills.
"""

from __future__ import annotations

import json
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent / "ledger.jsonl"

# Fields the backfill parser owns and may overwrite on every run.
DOC_DERIVED_FIELDS = (
    "title", "date", "version", "cost_usd", "cost_mentions",
    "verdict", "status_text", "source_doc",
)

VERDICTS = ("met", "not_met", "falsified", "passed", "blocked", "pending")


def new_row(id: str, **kwargs) -> dict:
    """Build a ledger row with all fields defaulted. kwargs override."""
    row = {
        "id": id,
        "parent_id": None,
        "title": None,
        "date": None,
        "version": None,
        "hypothesis": None,
        "prereg_metric": None,
        "config": {},
        "cost_usd": None,
        "cost_mentions": [],
        "result": {},
        "verdict": "pending",
        "status_text": None,
        "validated": {"physicality": None, "committor": None, "bootstrap": None},
        "lesson": "",
        "tags": [],
        "source_doc": None,
    }
    row.update(kwargs)
    if row["verdict"] not in VERDICTS:
        raise ValueError(f"verdict {row['verdict']!r} not in {VERDICTS}")
    return row


def load_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    """Read all rows. Returns [] if the ledger does not exist yet."""
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_by_id(path: Path = LEDGER_PATH) -> dict[str, dict]:
    """Read the ledger as an {id: row} mapping."""
    return {r["id"]: r for r in load_ledger(path)}


def write_ledger(rows: list[dict], path: Path = LEDGER_PATH) -> None:
    """Overwrite the ledger with `rows` (sorted by date then id for a
    stable, diff-friendly file)."""
    rows = sorted(rows, key=lambda r: (r.get("date") or "", r["id"]))
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


def append_row(row: dict, path: Path = LEDGER_PATH) -> None:
    """Append a single row (the normal write path for new attempts)."""
    if row.get("id") in load_by_id(path):
        raise ValueError(f"id {row['id']!r} already in ledger; use update_row")
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_row(row: dict, path: Path = LEDGER_PATH) -> None:
    """Insert or replace a row by id, rewriting the file."""
    by_id = load_by_id(path)
    by_id[row["id"]] = row
    write_ledger(list(by_id.values()), path)


if __name__ == "__main__":
    # Quick status summary.
    rows = load_ledger()
    print(f"{LEDGER_PATH}: {len(rows)} rows")
    from collections import Counter
    for verdict, n in Counter(r["verdict"] for r in rows).most_common():
        print(f"  {verdict:<10s} {n}")
    total = sum(r["cost_usd"] for r in rows if r.get("cost_usd"))
    print(f"  known cost total: ~${total:.2f}")
