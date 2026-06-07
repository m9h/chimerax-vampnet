"""Backfill the RSI ledger from the existing md/*_results.md corpus.

Zero-compute foundation step (md/rsi_lab_design.md, rollout step 1):
turn the hand-written results-doc archive — including the negative
results — into a structured, searchable ledger the loop can read.

This parser is intentionally conservative: it extracts only fields it
can read reliably from the markdown (title, date, version, cost
mentions, a verdict heuristic) and leaves the judgement fields
(hypothesis, prereg_metric, parent_id, lesson, validated, config,
result) for a human or the agent to enrich. Re-running is safe: it
MERGES, refreshing only the doc-derived fields (DOC_DERIVED_FIELDS) and
preserving any enrichment already in the ledger.

  $ python md/rsi/backfill.py            # write/merge md/rsi/ledger.jsonl
  $ python md/rsi/backfill.py --dry-run  # print rows, write nothing
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ledger import (DOC_DERIVED_FIELDS, LEDGER_PATH, load_by_id, new_row,
                    write_ledger)

MD_DIR = Path(__file__).resolve().parent.parent  # the md/ directory

# Verdict heuristics, checked in priority order against the doc text.
# (regex, verdict). First match wins.
VERDICT_RULES = [
    (r"\bfalsifi", "falsified"),
    (r"not\s+met|NOT\s+met", "not_met"),
    (r"\bblocked\b|NaN|hung|timed?\s*out", "blocked"),
    (r"\bMET\b|prediction\s+met|\bmet\b", "met"),
    (r"\bPASSED\b|validated end-to-end|\bvalidated\b", "passed"),
]

# Dollar figures: ~$45, $30, $0.30, $5-50, $5 - 50.
_DOLLAR = re.compile(r"~?\$\s?(\d+(?:\.\d+)?)(?:\s?-\s?(\d+(?:\.\d+)?))?")
_VERSION = re.compile(r"\bv(\d+\.\d+(?:\.\d+)?)")
_DATE = re.compile(r"\*\*Date\*\*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")
_H1 = re.compile(r"^#\s+(.*\S)", re.MULTILINE)
_STATUS = re.compile(r"\*\*Status\*\*:\s*(.+)", re.IGNORECASE)


def _slug(stem: str) -> str:
    """Filename stem -> ledger id (drop the _results suffix)."""
    return stem[:-8] if stem.endswith("_results") else stem


def _parse_costs(text: str) -> tuple[float | None, list[str]]:
    """Return (best_cost_usd, all_mentions). Prefers a 'cumulative' or
    'total' line if one carries a dollar figure; otherwise leaves the
    headline cost as None (we do NOT guess a precise number) but still
    records every mention for transparency."""
    mentions = []
    for m in _DOLLAR.finditer(text):
        mentions.append(m.group(0).strip())
    best = None
    for line in text.splitlines():
        low = line.lower()
        if ("cumulative" in low or "total" in low) and "$" in line:
            dm = _DOLLAR.search(line)
            if dm:
                # Range "$5-50" -> take the upper bound as the figure.
                best = float(dm.group(2) or dm.group(1))
                break
    return best, mentions


def _scan_verdict(text: str) -> str:
    for pat, verdict in VERDICT_RULES:
        if re.search(pat, text):
            return verdict
    return "pending"


def _parse_verdict(status_text: str | None, body: str) -> str:
    """Prefer the doc's own **Status** line (its self-summary); only
    fall back to scanning the whole body when the status line carries no
    verdict keyword. This avoids false positives from incidental words
    in the discussion (e.g. a pre-registered prediction that uses the
    word "falsifies")."""
    if status_text:
        v = _scan_verdict(status_text)
        if v != "pending":
            return v
    return _scan_verdict(body)


def parse_doc(path: Path) -> dict:
    text = path.read_text()
    title_m = _H1.search(text)
    title = title_m.group(1) if title_m else path.stem
    date_m = _DATE.search(text)
    ver_m = _VERSION.search(title) or _VERSION.search(text)
    cost, mentions = _parse_costs(text)
    status_m = _STATUS.search(text)
    status_text = status_m.group(1).strip() if status_m else None

    fields = {
        "title": title,
        "date": date_m.group(1) if date_m else None,
        "version": ("v" + ver_m.group(1)) if ver_m else None,
        "cost_usd": cost,
        "cost_mentions": mentions,
        "verdict": _parse_verdict(status_text, text),
        "status_text": status_text,
        "source_doc": f"md/{path.name}",
    }
    return fields


def backfill(dry_run: bool = False) -> list[dict]:
    docs = sorted(MD_DIR.glob("*_results.md"))
    existing = load_by_id()
    out: dict[str, dict] = dict(existing)  # start from what's there

    for doc in docs:
        rid = _slug(doc.stem)
        doc_fields = parse_doc(doc)
        if rid in out:
            # Merge: refresh only doc-derived fields, preserve enrichment.
            row = out[rid]
            for k in DOC_DERIVED_FIELDS:
                row[k] = doc_fields[k]
        else:
            row = new_row(rid, **doc_fields)
            # Seed tags from the id for quick filtering.
            row["tags"] = sorted(set(re.split(r"[_\-]", rid)))
        out[rid] = row

    rows = list(out.values())
    if dry_run:
        for r in sorted(rows, key=lambda r: (r.get("date") or "", r["id"])):
            cost = f"${r['cost_usd']:.2f}" if r["cost_usd"] is not None else "?"
            print(f"  {r['id']:<34s} {r.get('date') or '----------':<10s} "
                  f"{r['verdict']:<10s} {cost:>8s}  {r['title'][:48]}")
        print(f"\n{len(rows)} rows ({len(docs)} docs parsed); dry-run, "
              f"nothing written")
        return rows

    write_ledger(rows)
    print(f"wrote {len(rows)} rows to {LEDGER_PATH} ({len(docs)} docs parsed)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print parsed rows, write nothing")
    args = ap.parse_args()
    backfill(dry_run=args.dry_run)
