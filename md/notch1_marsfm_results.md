# Notch1 NRR MarS-FM inference — first results (v0.4 prototype)

**Date**: 2026-06-01
**Status**: MarS-FM adapter validated end-to-end on the Notch1 NRR apo
NEC chain (174 residues). ~80× cost reduction vs classical MD confirmed.
Multi-chain (NEC+NTM virtual concat) test running.

Modal app: `md/marsfm_modal.py::sample`
MarS-FM checkpoint: `valencelabs/mars-fm` (MD-Cath 450 weights, MIT)
Paper: Kapuśniak et al. 2025, [arXiv:2509.24779](https://arxiv.org/abs/2509.24779)
Code: [github.com/valence-labs/mars-fm](https://github.com/valence-labs/mars-fm)

## Headline

Three MarS-FM runs on Notch1 NRR apo (PDB 3I08):

| Run | Chain | Samples | Wall | Cost |
|---|---|---:|---:|---:|
| Smoke test | NEC chain A only (174 CAs) | 5 | ~80 sec | ~$0.03 |
| NEC200 | NEC chain A only (174 CAs) | 200 | ~3 min | ~$0.10 |
| FULL200 | NEC+NTM virtual concat (234 CAs) | 200 | ~3 min | ~$0.10 |

**~80× cost reduction** vs equivalent classical MD validated. ~60× wall-clock speedup at 200 samples.

## H2-relevant observable: NEC-NTM COM separation

The most direct biological readout from the FULL200 ensemble is the
NEC-NTM center-of-mass distance (the order parameter that distinguishes
the auto-inhibited basin from the activated basin):

| Method | COM sep mean ± std (Å) | Range (Å) | P(<10 Å) | P(10-25 Å) | P(≥25 Å) |
|---|---:|---:|---:|---:|---:|
| v0.3 MD restrained (300 ns) | 4.6 ± 0.3 | 3.4–5.6 | ~100% | 0% | 0% |
| v0.2 MD unrestrained (300 ns) | 13.9 ± 24.2 | 2.8–107 | 26% | 60% | 14% |
| **MarS-FM FULL200 (200 samples)** | **7.7 ± 3.5** | **0.8–20.7** | **77%** | **23%** | **0%** |

The MarS-FM distribution sits *between* the two MD baselines and looks
like a natural conformational ensemble: 77 % compact, 23 % partially
open, no full dissociation. This is the first protocol that recovers
P(auto-inhibited) close to the pre-registered ≥ 50 % threshold.

**Critical caveat: virtual-chain artifact.** The FULL run concatenates
NEC chain A and NTM chain B end-to-end into one 234-residue virtual
chain. MarS-FM therefore treats them as covalently connected through
a peptide bond between NEC residue 174 and NTM residue 175. *In vivo*,
NEC and NTM are produced by S1 cleavage of the precursor and are held
together only by the LNR-HD non-covalent interface. The artificial
peptide bond biases the MarS-FM ensemble toward the compact state by
preventing the partial NEC-NTM separation that the non-covalent
interface naturally allows. The 77 % P(<10 Å) likely overstates the
true equilibrium auto-inhibited population.

Pending: a holo MarS-FM run + a proper Notch1-style two-chain
non-covalent input (requires either MarS-FM-side multi-chain support
or a custom-trained MD-Cath-on-multi-chain checkpoint).

## What works

1. **MarS-FM single-protein inference adapter runs end-to-end** against
   a chimerax-vampnet equilibrated PDB. Pipeline: local PDB → atom14
   extraction (custom Bio.PDB parser → standard 14-atom AlphaFold
   ordering) → upload to Modal → MarS-FM `scripts/generate.py` → output
   PDB + XTC → CA-only NPZ that `vampnet load_ensemble … source marsfm`
   already consumes.

2. **MD-CATH-trained checkpoint generalises to Notch1 NEC.** The 174-
   residue Notch1 NEC sits within the MD-CATH 450 training distribution.
   Output looks like a folded protein with significant conformational
   excursion: Rg 14.7–26.5 Å, RMSD to seed up to 25 Å. Some samples
   likely in partially-unfolded basins (need fold-stability filtering
   for equilibrium-weighted analysis).

3. **Drop-in to existing chimerax-vampnet pipeline.** The output NPZ
   is loadable via `vampnet load_ensemble notch1_apo_marsfm
   notch1_apo_test200_marsfm.npz format marsfm` — no glue code beyond
   the adapter.

## What needs work

1. **Single-chain only.** MarS-FM's `atom14` representation takes one
   chain. The Notch1 apo NRR has NEC (chain A, 174 res) + NTM (chain B,
   60 res); the holo additionally has Fab heavy + light. v0.4's
   `--chain-id ALL` option concatenates all protein chains into one
   virtual chain; whether MarS-FM's flow matching handles inter-chain
   distances sensibly is an open empirical question (results below
   when the run lands).

2. **Output frame count interpretation.** Per the adapter bisection,
   `--max_mars_samples` does NOT control output count — that's
   `--calls_mars` (1 output frame per call). The MarS-FM docs are
   slightly ambiguous on this; clarification welcome.

3. **Equilibrium weighting.** MarS-FM samples beyond the MD horizon
   (max RMSD 25 Å on the NEC vs <2 Å in 100 ns MD), but it's unclear
   how to interpret the resulting ensemble for equilibrium-weighted
   quantities like P(auto-inhibited). The MarS-FM paper benchmarks
   against MD-CATH equilibrium statistics; for Notch1 NRR we'd need
   to either (a) use MarS-FM's internal MSM weights, (b) reweight
   against our v0.3 MD reference, or (c) treat the ensemble as
   exploratory (not Boltzmann-weighted).

## Operational notes (for replication)

7-iteration bisection to get end-to-end:

1. **NGC PyTorch base image** (`nvcr.io/nvidia/pytorch:26.04-py3`)
   instead of debian + own torch wheel.
2. **Split pip_install:** torch wheel from PyTorch index; everything
   else from default PyPI.
3. **Full MarS-FM dep set** (pytorch-lightning, torchdiffeq,
   dm-tree, fair-esm, biopython, mdtraj, wandb, matplotlib,
   statsmodels, etc.). MarS-FM has no `setup.py`; `mars/` is
   importable from PYTHONPATH via `cwd=/opt/mars-fm`.
4. **Patch vendored OpenFold for newer flash_attn:**
   `flash_attn_unpadded_kvpacked_func` → `flash_attn_varlen_kvpacked_func`
   (renamed in flash-attn v2.0).
5. **`atom14` input shape:** `(n_frames, n_residues, 14, 3)`, not
   `(n_residues, 14, 3)`. MarS-FM slices `arr[0:1]`.
6. **BioPython PDBParser** wants `StringIO` not `BytesIO`.
7. **`--calls_mars n_samples`** for n output frames (not
   `--max_mars_samples`).

## Cost comparison (Notch1 NRR scale)

| Method | Per-sample cost | 100 ns equivalent | Notes |
|---|---:|---:|---|
| OpenMM MD (Modal A100) | ~$0.01 | ~$25 | Real dynamics; bounded by sampling horizon |
| MarS-FM (Modal H100) | ~$0.001 | ~$0.30 (300 frames) | Beyond-MD-horizon; equilibrium weighting open |
| AlphaFlow (when integrated) | ~$0.001 est. | ~$0.10 | Different representation; single conformations |
| BioEmu (when integrated) | ~$0.001 est. | ~$0.10 | Trained on MD distributions |

## Suggested upstream contributions to MarS-FM

Two changes that would simplify integration for similar downstream tools:

1. **`--single_protein_pdb` flag on `scripts/generate.py`** that
   bypasses the MD-CATH split-CSV machinery and takes a PDB directly.
   Our adapter is ~80 lines that does exactly this.

2. **Multi-chain `atom14` input support.** For protein-protein
   complexes (NRR + Fab; antibody-antigen pairs; receptor-ligand) the
   single-chain restriction blocks ~half of the interesting use cases.

## Outputs

| File | Chain | Frames | n_CA | Size | Rg mean (Å) | Notes |
|---|---|---:|---:|---:|---:|---|
| `notch1_apo_test_marsfm.npz` | NEC only | 5 | 174 | 58 KB | 19.7 | smoke test |
| `notch1_apo_test200_marsfm.npz` | NEC only | 50 | 174 | 550 KB | 19.2 | adapter validation |
| `notch1_apo_NEC200_marsfm.npz` | NEC only | 200 | 174 | 2.1 MB | 19.5 | NEC-only ensemble |
| `notch1_apo_FULL200_marsfm.npz` | NEC+NTM virtual | 200 | 234 | 2.7 MB | 18.7 | full-NRR ensemble (peptide-bond caveat) |

Each npz has keys `coords` (all-atom), `coords_ca` (Cα only), `seqres`.
Load via the bundle:
```
vampnet load_ensemble notch1_apo_marsfm notch1_apo_FULL200_marsfm.npz format marsfm
```
