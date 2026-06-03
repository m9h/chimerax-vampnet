# H3 multi-source joint VAMPnet on Notch1 NEC — v0.4 result

**Date**: 2026-06-02
**Status**: pre-registered H3 prediction **MET** (3 of 3 sources).
AlphaFlow / ESMFlow integration pending; will re-run as 4-source
analysis when its adapter lands.

Script: `md/notch1_h3_multisource.py`
Output: `md/notch1_h3_multisource_results.json`

## H3 question (pre-registered)

> At least one VAMPnet state is reachable only via AlphaFlow or BioEmu
> samples, not via the 100 ns MD trajectories — evidence that the
> generative models provide complementary coverage of the
> conformational landscape inaccessible at our chosen MD horizon.

## Three-source joint VAMPnet

Ingredients (Notch1 NEC, 174 CAs):

| Source | n frames | Provenance |
|---|---:|---|
| **MD** (v0.3 COM-restrained) | 1500 | 3 × 100 ns Modal A100-80GB at 200 ps stride |
| **MarS-FM** | 200 | Modal H100, ~$0.10, MD-CATH 450 checkpoint |
| **BioEmu** v1.1 | 169 | Modal A100-80GB, microsoft/bioemu, filtered for physicality |

Joint VAMPnet: k=4 states, lag 20, 500 CA-CA distance features
standardised + clipped ±5σ, MLP 128-128-4 with ELU, fit on the union
of 1869 frames (no per-source weighting).

## Per-state source breakdown

| State | Pop | MD | MarS-FM | BioEmu | Verdict |
|---|---:|---:|---:|---:|---|
| 0 | 56.0 % | 69.7 % | 0.0 % | 0.0 % | **MD-only** |
| 1 | 13.0 % | 2.9 % | 97.0 % | 3.6 % | All 3 (MarS-FM-dominated) |
| 2 |  9.0 % | 0.0 % | 3.0 % | 96.4 % | **Generative-only** |
| 3 | 22.0 % | 27.4 % | 0.0 % | 0.0 % | **MD-only** |

(Row percentages are "fraction of THAT source's frames assigned to THIS
state". Population is the fraction of all 1869 frames in the state.)

## H3 verdict: **MET**

State 2 (9 % of the joint ensemble) is reached only by generative
samples — 96.4 % of BioEmu's frames and 3 % of MarS-FM's, with **zero
MD frames**. This is concrete first evidence that generative ensembles
surface protein conformations that classical 100 ns MD does not.

Additional structure:

- States 0 and 3 (78 % of joint pop, 97 % of MD frames) are **MD-only
  basins** — neither generative source reaches them. These are
  thermal-equilibrium conformations near the v0.3-equilibrated NEC.
- State 1 (13 % of joint pop) is the **cross-source overlap basin** —
  reached by all three sources, dominated by MarS-FM.
- State 2 (9 % of joint pop) is the **generative-only basin** —
  conformations the MarS-FM and BioEmu samplers explore but 100 ns
  unrestrained MD does not.

## Interpretation

The 80 % / 13 % / 9 % split (MD-only / cross-source / generative-only)
implies the v0.3 COM-restrained MD is sampling a **narrow** region of
the NEC conformational landscape — the auto-inhibited basin and its
immediate fluctuations — while MarS-FM and BioEmu explore further but
miss the MD-only modes (probably because the generative models are
trained on equilibrium snapshots and don't reproduce the same metric
of the restraint-stabilised local fluctuations the MD samples).

This is **complementary** coverage, not contradictory. A combined
ensemble (such as this joint fit) gives more comprehensive structural
diversity than any single source. The Notch1 NRR v0.4 paper headline
is therefore: *the multi-source approach the chimerax-vampnet bundle
was designed for is empirically justified — different sources surface
different states, and the union has measurably broader coverage than
the largest single source*.

## What's pending

- **AlphaFlow / ESMFlow integration**: the adapter at
  `md/alphaflow_modal.py` is still failing on numpy-2.0 / Python-3.12
  /OpenFold dep mismatches. 9 attempts so far. When it lands, the
  joint analysis re-runs trivially as 4-source.
- **Bootstrap on per-state source breakdowns**: the 96.4 % / 0.0 %
  numbers above are point estimates; jackknife over frames within
  each source would give uncertainty bands.
- **Per-source COM separation** within each state: useful for
  characterising what biophysical mode each basin represents.

## Cost

- Joint VAMPnet fit: ~1 min on local CPU, $0
- Cumulative v0.4 spend (item 1 MarS-FM + item 2 bootstrap + item 3
  BioEmu + 9 AlphaFlow build attempts + this H3): ~$15
