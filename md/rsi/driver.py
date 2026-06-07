"""RSI loop driver — STUB (rollout step 2).

NOT YET IMPLEMENTED beyond the control-flow sketch. Closes the
propose -> run -> score -> validate -> archive cycle that the project
currently does by hand. The pieces it orchestrates already exist:

  propose   : pick the next experiment. Step-2 target is NOT open-ended
              idea generation — it is the already-specified ESMFold2
              6th-source run (md/esmfold2_modal.py, wired into
              multisource_h3.py). One concrete, pre-registered turn to
              prove the loop closes.
  run       : invoke the Modal adapter for the chosen config.
  score     : objective.score on the measured result.
  validate  : validate.validate_state_claim on any unique-state claim.
  archive   : ledger.append_row, then auto-draft the md/*_results.md
              entry for human review BEFORE commit (AI-Scientist step).

Safeguards (design: md/rsi_lab_design.md, "what to NOT do"):
  - prereg_metric is written to the ledger BEFORE the run.
  - src/ self-modification (Darwin-Gödel step) is gated by `pytest
    tests/` staying green; not in scope for step 2.
  - every turn is dollar-costed and budget-bounded; no silent escalation.
"""

from __future__ import annotations

from ledger import append_row, new_row  # noqa: F401  (used once implemented)


def propose() -> dict:
    """Return the next attempt as a ledger row (verdict=pending), with
    its prereg_metric filled in BEFORE running. STUB — step-2 default is
    the ESMFold2 Notch1 NEC run."""
    raise NotImplementedError(
        "step 2: return new_row(... prereg_metric=...) for the ESMFold2 run")


def run(row: dict) -> dict:
    """Execute the attempt's config (Modal adapter) and return measured
    result fields for row['result']. STUB."""
    raise NotImplementedError("step 2: shell out to the chosen md/*_modal.py")


def turn(budget_usd: float | None = None) -> dict:
    """One propose->run->score->validate->archive cycle. STUB."""
    raise NotImplementedError(
        "step 2: wire propose/run/objective.score/validate/append_row")


if __name__ == "__main__":
    raise SystemExit("driver.py is a stub (rollout step 2); see "
                     "md/rsi_lab_design.md for the build order")
