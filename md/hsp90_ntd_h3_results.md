# H3 multi-source joint VAMPnet on Hsp90 NTD — v0.7 result

**Date**: 2026-06-05
**System**: human Hsp90α N-terminal domain (residues 17-223; 207 CAs)
**Status**: 5 of 5 sources integrated; preliminary single-system pilot.

## apo / holo glossary

- **apo** (Greek ἀπό): the protein *without* its bound ligand
  ("apoprotein"). In v0.7 Hsp90 work, both MD trajectories are apo
  simulations (no ligand present in the MD).
- **holo** (Greek ὅλος): the protein *with* its bound ligand
  ("holoprotein"). In v0.7 Hsp90, no MD here actually contains a
  ligand — see naming note below.

Naming note (v0.7.1 correction): the Hsp90 trajectory previously
labeled "MD_holo" is renamed **MD_apoFromHolo** — it's an apo MD
seeded from PDB 1YET (the geldanamycin-bound crystal), with the
ligand stripped during prep (PDBFixer's `removeHeterogens`). The
old label was misleading because no ligand was ever present in the
simulation. A true ligand-bound holo MD would need GDM
parameterized via Amber GAFF / OpenFF; queued for later work.

In Notch1 v0.3+ the apo/holo labels remain accurate — the anti-NRR
Fab really is present in the holo MD (3L95 chains kept).

Scripts:
- `md/multisource_h3.py --system hsp90_ntd`
- `md/hsp90_ntd_h3_biology.py` (4-source biology preview; 5-source
  inline addendum in this document)

## Five sources

| Source | n frames | Provenance |
|---|---:|---|
| **MD** | 45 000 | 3 × 300 ns on Modal A100-80GB, 1YER apo prepped via PDBFixer (heterogens stripped, no chain restraint), 20 ps stride, ~$30 |
| **MarS-FM** | 200 | Modal H100, MD-CATH 450 checkpoint, ~$0.10 |
| **BioEmu** v1.1 | 188 | Modal A100-80GB, microsoft/bioemu, filtered for physicality, ~$0.50 |
| **Boltz-2** | 200 | Modal A100-80GB, boltz-community/boltz2, ~$0.30 |
| **AlphaFlow** / ESMFlow-MD | 200 | Modal A100-80GB, bjing-mit/alphaflow:esmflow\_md\_base\_202402, ~$0.30 |

## Per-state per-source breakdown

| State | Pop | MD | MarS-FM | BioEmu | Boltz-2 | AlphaFlow | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 29.4 % | 29.9 % |  0.0 % |   0.0 % |   0.0 % |   0.0 % | **MD-only** (extended, e2e 40 Å) |
| 1 | 22.5 % | 21.4 % | 53.0 % | 100.0 % | 100.0 % | 100.0 % | **all-5 consensus** (mid-range) |
| 2 | 31.3 % | 31.8 % |  0.0 % |   0.0 % |   0.0 % |   0.0 % | **MD-only** (e2e 36, lid open) |
| 3 | 16.8 % | 16.9 % | 47.0 % |   0.0 % |   0.0 % |   0.0 % | MD + MarS-FM (extended, e2e 41) |

## 5-source biology features (Å)

| state | source | n | Rg | e2e | ATP-COM | lid-Rg | RMSD vs 1YER |
|---|---|---:|---|---|---|---|---|
| 0 | MD | 13 461 | 16.6 ± 0.1 | 40.2 ± 3.0 | 21.2 ± 0.7 | 9.0 ± 0.1 | 1.5 ± 0.2 |
| 1 | MD | 9 616 | 16.7 ± 0.1 | 36.0 ± 2.5 | 22.3 ± 1.0 | 9.0 ± 0.2 | 1.7 ± 0.4 |
| 1 | MarS-FM | 106 | 16.8 ± 0.8 | 34.4 ± 9.2 | 22.1 ± 6.2 | 9.2 ± 1.2 | 7.6 ± 2.6 |
| 1 | BioEmu | 188 | 16.6 ± 0.3 | 33.4 ± 3.2 | 21.7 ± 1.7 | 8.9 ± 0.8 | 3.0 ± 0.9 |
| 1 | Boltz-2 | 200 | **16.4 ± 0.0** | 33.4 ± 0.5 | 22.2 ± 0.3 | 9.0 ± 0.2 | **1.3 ± 0.1** |
| 1 | AlphaFlow | 200 | 16.7 ± 0.1 | 33.7 ± 1.9 | 22.5 ± 1.2 | 9.0 ± 0.6 | 2.3 ± 0.3 |
| 2 | MD | 14 310 | 16.6 ± 0.1 | 35.9 ± 2.5 | 21.1 ± 0.8 | 9.3 ± 0.2 | 1.5 ± 0.2 |
| 3 | MD | 7 613 | 16.7 ± 0.1 | 41.2 ± 3.0 | 21.5 ± 1.0 | 9.4 ± 0.2 | 2.4 ± 0.1 |
| 3 | MarS-FM | 94 | 16.6 ± 0.8 | 34.8 ± 8.3 | 22.4 ± 5.2 | 9.4 ± 1.2 | 7.0 ± 2.3 |

(BioEmu, Boltz-2, AlphaFlow have 0 frames in states 0/2/3 — they are
entirely in state 1.)

## H3 verdict on Hsp90 NTD: **OPPOSITE of Notch1**

**Notch1 NRR** (v0.5/v0.6): MD limited to the auto-inhibited basin
(states 0+3); AlphaFlow + Boltz-2 + BioEmu surfaced an
MD-inaccessible "structure-prediction" basin (state 1); MarS-FM +
BioEmu surfaced a separate flow-matching basin (state 2). 3-way
sampler-class taxonomy; generative ensembles ADDED coverage MD
couldn't reach.

**Hsp90 NTD** (v0.7): MD is the broad explorer covering states 0, 2,
3 (76 % of frames) — the protein is naturally flexible at 310 K with
no auto-inhibitory constraint. ALL generative samplers
(AlphaFlow 100 %, Boltz-2 99.5 %, BioEmu 100 %) collapse onto a
single "consensus" state 1 (RMSD 1-3 Å vs the 1YER apo crystal).
MarS-FM is the only generative sampler that reaches state 3 (47 %
of its frames), an MD-extended state. Generative ensembles
SUBTRACT from MD's coverage on this flexible system.

## Why the inversion

Hsp90 NTD is a single-domain monomeric ~220-residue enzyme with no
auto-inhibitory contact. Its equilibrium ensemble is BROAD — three
distinct end-to-end distances (~36, ~40, ~41 Å) appear within
300 ns of unbiased MD per replica. The pre-trained generative
models (AlphaFlow, Boltz-2, BioEmu) all learn from PDB-style
single-structure data and naturally produce conformations close to
the *most populated* crystal structure (1YER, RMSD 1.3-3 Å). They
have neither the training signal nor the architecture to reproduce
the lateral conformational excursions MD samples natively.

Notch1 NRR is the opposite: it's a two-chain restraint-stabilised
complex whose equilibrium ensemble is NARROW (single
auto-inhibited basin) on the unbiased 100-ns MD horizon. There the
generative models — which freely sample conformations including
ones the restrained MD can't reach — *expand* the coverage.

## Boltz-2 compactness bias replicated

Boltz-2 on Hsp90 NTD: Rg **16.4 ± 0.0 Å** across 200 samples
(zero std visible at this precision), RMSD vs 1YER apo crystal
**1.3 ± 0.1 Å**. This is essentially Boltz-2 returning the same
crystal-like structure 200 times with negligible diffusion
diversity. Replicates the v0.6 Notch1 finding of a systematic
compactness bias in Boltz-2 — and on Hsp90 the bias manifests as
near-zero conformational diversity rather than a smaller Rg
relative to MD (because the Hsp90 MD also has small Rg, the bias
shows up in the variance instead).

## Implications for the pipeline

The v0.5 / v0.6 Notch1 result was framed as "generative ensembles
add coverage MD can't reach". The v0.7 Hsp90 result shows this
characterisation is **system-dependent**:

- For systems with **constrained equilibrium dynamics** (auto-
  inhibited complexes, multi-domain assemblies held by interface
  contacts), MD is the limited sampler and generative ensembles
  EXPAND coverage. → Notch1 NRR.
- For systems with **broad equilibrium dynamics** (flexible
  monomeric enzymes), MD samples natively and generative ensembles
  CONCENTRATE on the modal conformation. → Hsp90 NTD.

The MarS-FM exception is interesting: on both Notch1 and Hsp90,
MarS-FM has the highest conformational variance of any generative
sampler. Flow-matching in MSM state space appears to add value
across system classes, while AF3-class diffusion samplers
(AlphaFlow, Boltz-2) consistently concentrate near the training-
data manifold.

## Cost (v0.7 cumulative for Hsp90 NTD)

- Hsp90 apo prep: ~$5
- 3 × 300 ns apo MD on A100-80GB: ~$30 (faster than estimated at
  ~800 ns/day on the 220-residue monomer vs Notch1's ~300 ns/day
  on 234 CAs + Fab)
- 4 generative ensembles (200 samples each): ~$1.20
- 5-source VAMPnet fit + biology features: $0 (local CPU)
- Total so far: ~$36

Pending: true ligand-bound holo MD (geldanamycin parameterized via
Amber GAFF / OpenFF; v0.7.x), cryptic-pocket analysis with
heavy-atom SASA via mdtraj.shrake_rupley or fpocket (v0.7.x), and
AF3 6-source when DeepMind weights arrive.

## See also

- `md/hsp90_ntd_h3_multisource_results.json` (machine-readable
  state breakdown)
- `md/hsp90_ntd_h3_biology_5source_results.json` (per-state
  per-source feature means)
- `md/figures/hsp90_ntd_h3_biology_preliminary.png` (4-source
  scatter; will be superseded by a 5-source variant)
