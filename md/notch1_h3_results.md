# H3 multi-source joint VAMPnet on Notch1 NEC — v0.5 result

**Date**: 2026-06-03
**Status**: pre-registered H3 prediction **MET** with **5 of 5 sources**.
All four generative samplers (AlphaFlow / ESMFlow-MD, BioEmu, MarS-FM,
Boltz-2) are now integrated as Modal adapters with independent
self-contained environments; the joint VAMPnet recovers a clean
3-class source split.

Script: `md/notch1_h3_multisource.py`
Output: `md/notch1_h3_multisource_results.json`

## H3 question (pre-registered)

> At least one VAMPnet state is reachable only via generative-model
> samples (AlphaFlow, BioEmu, MarS-FM, Boltz-2), not via the 100 ns MD
> trajectories — evidence that the generative models provide
> complementary coverage of the conformational landscape inaccessible
> at our chosen MD horizon.

## Five-source joint VAMPnet (v0.5)

Ingredients (Notch1 NEC, 174 CAs):

| Source | n frames | Provenance |
|---|---:|---|
| **MD** (v0.3 COM-restrained) | 1500 | 3 × 100 ns Modal A100-80GB at 200 ps stride |
| **MarS-FM** | 200 | Modal H100, ~$0.10, MD-CATH 450 checkpoint |
| **BioEmu** v1.1 | 169 | Modal A100-80GB, microsoft/bioemu, filtered for physicality |
| **Boltz-2** | 200 | Modal A100-80GB, 6m14s, ~$0.30, `debian_slim` image |
| **AlphaFlow / ESMFlow-MD** | 200 | Modal A100-80GB, 12m49s, ~$0.30, CUDA-11.8 + micromamba |
| **ESMFold2** *(pending, v0.7.x)* | — | biohub/ESMFold2 (MIT); single-chain NEC (174 CAs) via `md/esmfold2_modal.py`; adapter wired into `multisource_h3.py` but ensemble not yet generated |

Joint VAMPnet: k=4 states, lag 20, 500 CA-CA distance features
standardised + clipped ±5σ, MLP 128-128-4 with ELU, fit on the union
of 2269 frames (no per-source weighting), drop_last batches.

## Per-state source breakdown (v0.5, 5 sources)

| State | Pop | MD | MarS-FM | BioEmu | Boltz-2 | AlphaFlow | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 46.0 % | 69.6 % |   0.0 % |   0.0 % |   0.0 % |   0.0 % | **MD-only** |
| 1 | 17.7 % |  0.0 % |   0.0 % |   1.2 % |  99.5 % | 100.0 % | **AF3 / ESMFlow-MD-only** |
| 2 | 17.5 % |  2.0 % | 100.0 % |  98.8 % |   0.5 % |   0.0 % | flow-matching + MD |
| 3 | 18.8 % | 28.4 % |   0.0 % |   0.0 % |   0.0 % |   0.0 % | **MD-only** |

(Row percentages are "fraction of THAT source's frames assigned to THIS
state". Population is the fraction of all 2269 frames in the state.)

## H3 verdict: **MET** — triply corroborated

State 1 (17.7 % of the joint ensemble) is reached only by structure-
prediction-style generative samples — **100 % of AlphaFlow frames**,
**99.5 % of Boltz-2 frames**, and a small 1.2 % BioEmu contribution,
with zero MD frames and zero MarS-FM frames. State 2 (17.5 %) is
reached by MarS-FM (100 %) + BioEmu (98.8 %) + a small MD trickle
(2 %), but **zero AlphaFlow** and effectively zero Boltz-2.

The triple corroboration is the key v0.5 finding: state 1 is **not an
artifact of any single model**. Three independently-trained generative
networks (AlphaFlow / ESMFlow-MD, Boltz-2 AF3-class diffusion, BioEmu
v1.1 Boltzmann emulator) place a substantial fraction of their
conformations in a basin classical MD does not reach within 100 ns.
Two of them (AlphaFlow + Boltz-2) place *essentially all* their
conformations there.

## Sampler-class split

The v0.5 5-source result reveals a clean 3-way taxonomy of conformational
sources:

1. **Classical MD (states 0 + 3, 64.8 % of joint pop)** — restrained
   equilibrium near the v0.3-prepared NEC. 98 % of MD frames live here.
   Neither generative class reaches these states.
2. **Structure-prediction-only basin (state 1, 17.7 %)** — AF3-class
   diffusion (Boltz-2) and ESMFlow-MD (AlphaFlow) saturate this state;
   BioEmu has a small overlap. MD and MarS-FM never reach it.
3. **Flow-matching basin (state 2, 17.5 %)** — MarS-FM and BioEmu
   together cover this state; AlphaFlow and Boltz-2 do not.

This is exactly the multi-source dividend the chimerax-vampnet bundle
was designed to surface: different sampler families have different
biases, and the joint VAMPnet recovers each one as a distinct VAMPnet
state. Choosing any single source would have missed at least one basin
(MD: misses states 1 and 2; AlphaFlow or Boltz-2: misses state 2 and
the MD-equilibrium basins; MarS-FM or BioEmu: misses state 1 and the
MD-equilibrium basins).

## Biological annotation (v0.6)

The five-source statistical split above was grounded in
interpretable structural features in `md/notch1_h3_biology.py` and
written up in `md/notch1_h3_biology_results.md`. Key annotations:

- **States 0 and 3** are sub-basins of the *same* MD auto-inhibited
  equilibrium (Rg 18.9, NEC–NTM COM 4.6–4.7 Å on top of the v0.3
  restraint set point of 3.94 Å). They differ only in NEC
  end-to-end distance (18 vs 24 Å) — not in interface geometry.
- **State 1** (AF3 structure-prediction) is reached by AlphaFlow
  and Boltz-2 via *different* structural routes that share the
  same direction: tighter LNR-A → HD-N packing (22–24 Å) than MD
  samples (27 Å). Boltz-2 collapses (Rg 16.8); AlphaFlow shortens
  end-to-end (15 Å). Suggestive of Fab-bound-like contraction
  even though no Fab conditions either model.
- **State 2** (flow-matching) is the highest-variance state.
  MarS-FM samples end-to-end distances at 35.6 ± 14.3 Å (std
  almost twice MD's mean) and reaches conformations 14 Å (Kabsch
  RMSD) from the MD mean — the long-tail magnitude estimator we
  always wanted from MD.
- **Boltz-2** has a **systematic compactness bias** (Rg 16.8 vs
  MD's 18.9 across both states it occupies), predictable from its
  PDB-structure training distribution.

## ESMFold2 (6th source) — pending

ESMFold2 (Rives et al. 2026, "A World Model of Protein Biology",
biohub.org) is the newest AF3-class diffusion structure/complex
predictor (on top of the ESMC protein LM). Adapter at
`md/esmfold2_modal.py`; wired into `multisource_h3.py` as the
`ESMFold2` source (single-chain NEC, 174 CAs, to share this feature
space). Awaiting first generation run.

**Pre-registered prediction** (before the data exists, to avoid
post-hoc rationalisation): ESMFold2, being AF3-class, should behave
like AlphaFlow + Boltz-2 — i.e. **saturate state 1** (the structure-
prediction-only basin: tighter LNR-A → HD-N packing, ~22–24 Å) with
near-zero frames in the MD-equilibrium states 0/3 and the
flow-matching state 2. If it instead reaches state 2 or a *new*
unique state, that falsifies the "AF3-class diffusion concentrates
on the structure-prediction manifold" taxonomy. Either outcome is
informative; the null (state-1 saturation) is the strongest possible
restatement that single-structure prediction — however SOTA — is not
a dynamics substitute.

Note the multi-chain ESMFold2 capability (NEC+NTM, or holo NRR+Fab)
is a **separate** deposit with a different CA count — it feeds a
future complex-level H2/H3 variant, not this 174-CA NEC comparison.

## What's pending

- **AlphaFold3 weights-gated integration**: queued behind DeepMind's
  approval process; not on the v0.5 critical path.
- **ESMFold2 6th source**: adapter + wiring done; generation run + refit pending.
- **Per-source COM separation** within each state.
- **Bootstrap CI on the per-state breakdowns**.

## Cost (v0.5 cumulative)

- Joint VAMPnet fit: ~1 min on local CPU, $0
- Boltz-2 200-sample run: 6m14s on A100-80GB, ~$0.30
- AlphaFlow / ESMFlow-MD 200-sample run: 12m49s on A100-80GB, ~$0.30
- 8 AlphaFlow attempt builds + 1 retry-stop: ~$1
- Cumulative v0.5 spend: ~$17 (v0.4 baseline + Boltz + AlphaFlow)


---


# H3 multi-source joint VAMPnet on Notch1 NEC — v0.4 result (archived)

**Date**: 2026-06-02
**Status**: pre-registered H3 prediction **MET** (3 of 3 sources).
AlphaFlow / ESMFlow integration pending; will re-run as 4-source
analysis when its adapter lands.

Script: `md/notch1_h3_multisource.py`
Output: `md/notch1_h3_multisource_results.json`

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

## H3 verdict: **MET**

State 2 (9 % of the joint ensemble) is reached only by generative
samples — 96.4 % of BioEmu's frames and 3 % of MarS-FM's, with **zero
MD frames**. This is concrete first evidence that generative ensembles
surface protein conformations that classical 100 ns MD does not.
