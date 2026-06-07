"""Validation gates for RSI "new coverage" claims — STUB (rollout step 3).

NOT YET IMPLEMENTED. This file pins the interface the loop will call
before any generative-discovered VAMPnet state is allowed to count as
real coverage in objective.py (`h3_unique_states`). The whole point of
these gates is anti-reward-hacking: a generative model can trivially
emit an "unseen" conformation that is simply unphysical, so novelty
alone must never score (design: md/rsi_lab_design.md).

Three gates, each returning a structured verdict dict
{passed: bool, detail: str, ...}:

  physicality(coords) — clash / bond-geometry / cis-peptide screen.
      Generalises the existing per-source BioEmu physicality filter to
      every source. A state built from frames that fail this is void.

  committor(state_mean_pdb, ...) — seed short MD from the state's mean
      structure and check it does NOT immediately relax out of the
      basin. Reuses the morph-and-reequilibrate path already prototyped
      in examples/live_adaptive_sampling_notch1.py (Phase 2) — here as
      a *validator*, not a sampler. A state no MD will stay in is not a
      metastable state.

  bootstrap(assignments, ...) — the per-state per-source occupancy must
      survive an independent VAMPnet seed and a bootstrap CI excluding
      zero. Builds on md/notch1_h3_bootstrap_results.md.

A unique-state claim counts for objective.h3_unique only if all three
gates pass.
"""

from __future__ import annotations


def physicality(coords) -> dict:
    raise NotImplementedError(
        "physicality gate: clash/bond/cis-peptide screen (rollout step 3)")


def committor(state_mean_pdb: str, n_replicas: int = 3, ns: float = 5.0) -> dict:
    raise NotImplementedError(
        "committor gate: short MD from state mean, basin-retention check "
        "(rollout step 3; reuse live_adaptive_sampling_notch1 Phase-2 path)")


def bootstrap(assignments, sources, n_boot: int = 1000) -> dict:
    raise NotImplementedError(
        "bootstrap gate: CI-excludes-zero on per-state occupancy "
        "(rollout step 3; build on notch1_h3_bootstrap_results.md)")


def validate_state_claim(coords, state_mean_pdb, assignments, sources) -> dict:
    """Run all three gates; a claim is valid only if all pass. STUB."""
    raise NotImplementedError(
        "wire physicality + committor + bootstrap once each gate lands")
