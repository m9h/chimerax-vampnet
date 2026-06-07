# `md/rsi/` — recursive self-improvement lab

Harness for closing the chimerax-vampnet research loop into an
archived, evolvable, self-improving cycle. Design + rationale:
[`../rsi_lab_design.md`](../rsi_lab_design.md).

## Status by rollout step

| Step | What | State |
|---|---|---|
| 1 | Ledger + backfill + objective | **done** (`ledger.py`, `backfill.py`, `objective.py`) |
| 2 | One autonomous turn (ESMFold2 run via the driver) | stub (`driver.py`) |
| 3 | Validation gates (physicality / committor / bootstrap) | stub (`validate.py`) |
| 4 | ShinkaEvolve-style config evolution | not started |

## Files

- `ledger.py` — append-only experiment archive + lineage (JSONL). Row
  schema, IO helpers (`load_ledger`, `append_row`, `upsert_row`,
  `write_ledger`). Run `python ledger.py` for a status summary.
- `backfill.py` — parse the existing `md/*_results.md` corpus (incl.
  negative results) into `ledger.jsonl`. Idempotent: re-running
  refreshes only doc-derived fields and preserves human/agent
  enrichment (`parent_id`, `hypothesis`, `prereg_metric`, `lesson`,
  `config`, `result`, `validated`). `--dry-run` to preview.
- `objective.py` — composite, anti-reward-hacking scorer. Weights
  validated coverage and the unsolved H2 magnitude over raw novelty;
  reports information-per-dollar. `python objective.py` to rank.
- `validate.py` — STUB. The three gates a "new state" claim must pass
  before it scores.
- `driver.py` — STUB. The propose→run→score→validate→archive cycle.
- `ledger.jsonl` — the archive itself (generated).

## Quickstart

```bash
cd md/rsi
python backfill.py        # build/refresh ledger.jsonl from the docs
python ledger.py          # verdict + cost summary
python objective.py       # rank scored attempts (empty until enriched)
```

## Conventions

- The ledger is the source of truth for "what's been tried." Before
  proposing an experiment, search it (don't re-run a falsified config).
- `prereg_metric` is written **before** a run, never after.
- Backfilled historical rows start with empty `result`/`config` and a
  `pending`/heuristic verdict; enrich them as the loop touches them.
- A unique-state claim scores in `objective` only after all three
  `validate` gates pass — novelty alone never counts.
