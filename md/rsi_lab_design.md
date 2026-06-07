# Recursive self-improvement lab — design for chimerax-vampnet

**Date**: 2026-06-07
**Status**: design proposal (no harness built yet)
**Inspiration**: Sakana AI's RSI Lab (sakana.ai/rsi-lab) — The AI
Scientist (Nature, Mar 2026), Darwin Gödel Machine, ShinkaEvolve.

## Why this fits *this* project specifically

The interesting observation is that chimerax-vampnet already contains
most of the moving parts of an autonomous research loop; they're just
not closed into a self-improving cycle yet:

| RSI primitive | What already exists here |
|---|---|
| **Experiment executor** | the `md/*_modal.py` adapters (MD, AlphaFlow, BioEmu, Boltz-2, MarS-FM, ESMFold2, metad) — each a one-command, costed, reproducible run |
| **Agent control loop** | `examples/live_adaptive_sampling_notch1.py` + `src/mcp_server.py` — an agent already decides a rare state → launches Modal MD → ingests → refits → re-scores |
| **Objective / scorer** | `md/multisource_h3.py`, `vampnet timescales`, the H2/H3 verdicts — pre-registered, quantitative |
| **Archive** | the `md/*_results.md` corpus — *including negative results* (`marsfm_multichain_phase1_results.md`, W3d NaN'd) |
| **Pre-registration** | H2/H3 predictions written *before* the data (now also the ESMFold2 rows) — the anti-reward-hacking discipline Sakana emphasises |

So the RSI ask is not "adopt a new framework" — it's "promote the
human-in-the-loop research cadence into a closeable, archived,
evolvable loop, and let the agent run more turns of it autonomously
between human checkpoints."

## The three Sakana systems, mapped to concrete moves here

### 1. The AI Scientist → autonomous experiment→writeup loop
The manual cadence today is: pick an experiment → run a Modal adapter
→ eyeball the result → hand-write `md/<x>_results.md`. The AI-Scientist
move is to let the agent close that loop: propose the next experiment
from the open/queued lists already at the bottom of each results doc,
run it, score it against the pre-registered metric, and *auto-draft*
the results-doc entry (which a human reviews before commit).

Lowest-risk first target: the **ESMFold2 6th-source run** is already
fully specified (adapter + wiring + pre-registered prediction). It is
a perfect first autonomous turn — generate, refit `multisource_h3.py`,
fill in the predicted-vs-observed table, draft the verdict paragraph.

### 2. Darwin Gödel Machine → the agent edits the *bundle's own code*
DGM keeps an evolving lineage of agent variants that rewrite their own
codebase, gated by a benchmark. Here the analogue is the agent
proposing patches to `src/` (a new featurizer in `featurize.py`, a new
loader, a new CV), **gated by the existing test suite** (`tests/`,
18 tests + the `.cxc` integration). A variant is only admitted to the
lineage if tests stay green *and* it improves the objective. The test
suite is the verification gate that makes self-modification safe — it
already exists, which is the hard part.

### 3. ShinkaEvolve → sample-efficient evolution of experiment configs
Sakana's emphasis is sample-efficiency on modest compute — which is
*exactly* this project's ethos (~$1 generative runs, ~$5 metad
walkers, "$0 local CPU refit"). The ShinkaEvolve move is to treat the
experiment configs as an evolvable population: e.g. metad CV choices
(the `(d1,d2)` pair in `notch1_metad_2d_fes_results.md`), restraint
constants, VAMPnet hyperparams (lag, n_states, n_features), sampler
seeds/temperatures. Maintain a population, novelty-filter (don't re-run
near-duplicate configs), and spend the next dollar on the
highest-expected-information config. The token/$ budget discipline is
already native to the project.

## The objective function (and why it resists reward hacking)

The central risk in *conformational* RSI: an agent can trivially
"discover a new VAMPnet state" by producing **unphysical** structures
(exploded geometry, cis-peptides, steric clashes). The project already
knows this — BioEmu frames are "filtered for physicality". So the
objective must be composed of metrics that are expensive to game:

1. **H2 directional + magnitude** — apo > holo auto-inhibition, with
   the pre-registered magnitude thresholds (holo ≤ 30 %, apo ≥ 50 %).
   Magnitude is the standing open problem; a real improvement here is
   hard to fake.
2. **H3 unique-state coverage** — a source earns credit for a state
   *only if* the state survives validation (below), not merely for
   landing frames somewhere new.
3. **Slowest implied timescale** (`vampnet timescales`) — the
   adaptive-sampling demo already optimises this (94 → 201 ns).
4. **Cost** — every run is dollar-costed; the objective is
   information-per-dollar, not raw metric.

**Anti-reward-hacking safeguards** (the part that must be built in from
the start, per Sakana's "verifiable safeguards"):

- **Physicality gate** — any generative-discovered state must pass a
  clash/bond-geometry filter before it counts (extend the BioEmu
  filter to all sources).
- **MD-committor validation** — a generative-only state is only "real
  coverage" if short MD seeded from its mean structure stays in the
  basin (doesn't immediately relax back). This is precisely the
  `live_adaptive_sampling_notch1.py` Phase-2 morph-and-reequilibrate
  path — reused as a *validator*, not just a sampler.
- **Replication** — a claimed state must survive an independent VAMPnet
  seed and a bootstrap CI excluding zero (the project already has
  `notch1_h3_bootstrap_results.md`).
- **Pre-registration** — the metric and threshold are written to the
  ledger *before* the run (as the ESMFold2 rows now do by hand).

## Minimal harness to build (`md/rsi/`)

```
md/rsi/
  ledger.jsonl        # append-only archive: one row per attempt
  objective.py        # composite scorer over H2/H3/timescale/cost
  validate.py         # physicality + committor + bootstrap gates
  driver.py           # propose -> run adapter -> score -> validate ->
                      #   append ledger -> (human review) -> commit doc
```

Ledger row schema (the archive + the "structured lessons from
failures" Sakana extracts):

```json
{
  "id": "v0.7.5-esmfold2-notch1",
  "parent_id": "v0.5-5source",
  "hypothesis": "ESMFold2 saturates state 1 (AF3-class)",
  "prereg_metric": "frac_esmfold2_in_state1 >= 0.9",
  "config": {"adapter": "esmfold2_modal", "system": "notch1_apo_v3", "n_samples": 200},
  "cost_usd": 1.0,
  "result": {"frac_in_state1": null, "unique_states": []},
  "verdict": "pending",
  "validated": {"physicality": null, "committor": null, "bootstrap": null},
  "lesson": ""
}
```

`parent_id` is what makes it a *lineage* (DGM-style) rather than a flat
log — each attempt descends from the configuration it mutated.

## Suggested rollout (sample-efficient, low-risk first)

1. **Ledger + objective only** — retro-fill the existing `md/*_results.md`
   corpus into `ledger.jsonl`. Zero new compute; immediately gives the
   agent a searchable archive of what's been tried and what failed.
2. **One autonomous turn** — the ESMFold2 run end-to-end through the
   driver, human-reviewed. Proves the loop closes.
3. **Validator gates** — wire the physicality + committor + bootstrap
   checks as hard gates on "new state" claims.
4. **Evolution** — only then turn on ShinkaEvolve-style population
   search over metad CVs / hyperparams, with novelty filtering.

## What to deliberately NOT do

- No unsupervised self-modification of `src/` without the test-suite
  gate (DGM without the benchmark is how you "pass benchmarks but fail
  in deployment").
- No optimising a proxy (e.g. "number of states") instead of validated
  physical coverage — that is the reward-hack this domain invites.
- No silent compute escalation — every turn is dollar-costed and
  budget-bounded, consistent with the project's existing cost notes.

## See also
- `examples/live_adaptive_sampling_notch1.py` — the existing proto-loop
- `src/mcp_server.py` — the agent control surface
- `md/notch1_h3_results.md`, `md/hsp90_ntd_h3_results.md` — the
  pre-registration discipline this would systematise
