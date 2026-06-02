# MarS-FM multi-chain `atom14` Phase-1 — empirical validation

**Date**: 2026-06-01
**Status**: Phase-1 inference-only chain-aware API IMPLEMENTED and HONORED.
Effect on output distribution is minimal — motivates Phase 2 (pair
representation + retrain).

Reference plan: `/home/mhough/.claude/plans/twinkly-whistling-bonbon.md`.
Modified MarS-FM fork at `/tmp/mars-fm-fork/` (will push to GitHub
once we decide on upstream PR strategy).

## What we implemented

Minimal Phase-1 changes (~50 LOC across 3 upstream files + 30 LOC in
our adapter):

| File | Change |
|---|---|
| `mars/model/model.py` | Added `_chain_aware_pos_embed(chain_id)` method + `chain_id` kwarg to `forward`. When `chain_id` is provided, the positional embedding resets per-chain instead of using a single `np.arange(crop)` index. |
| `mars/model/module.py:_build_model_kwargs` | Passes `batch["chain_id"]` to `model.forward` if present. Backward compatible. |
| `scripts/generate.py:load_starting_structure` | Loads `<name>_chain_id.npy` alongside `<name>.npy` if it exists; adds to item dict. Backward compatible. |
| `md/marsfm_modal.py` (our adapter) | Emits per-residue `chain_id.npy` alongside the atom14 input; switched the Modal image to overlay our fork via `add_local_dir`. |

Verification (single-chain regression): the smoke-test confirmation
line `[generate] loaded chain_id from ... 2 chains, lengths [174, 60]`
fires for multi-chain inputs; falls back silently for single-chain.

## Empirical comparison

Four 200-sample runs on Notch1 NRR, two each for apo (3I08) and holo
(3L95 NRR chains X+K). Compared the existing virtual-concat baseline
(one virtual chain, `--chain-id ALL` collapses everything into one
sequence) to the new Phase-1 API (real per-residue `chain_id`,
positional embedding resets at chain breaks).

| Run | n_chains | NEC-NTM COM mean ± std (Å) | P(<10 Å) | Rg mean ± std (Å) |
|---|---:|---:|---:|---:|
| apo virtual-concat | 1 (forced) | 7.75 ± 3.50 | 77.0 % | 18.72 ± 1.73 |
| **apo chain-API**  | 2 (real)   | **7.75 ± 3.50** | **77.5 %** | 18.72 ± 1.73 |
| holo virtual-concat | 1 (forced) | 8.16 ± 3.67 | 71.0 % | 18.99 ± 1.49 |
| **holo chain-API** | 2 (real)   | **8.16 ± 3.67** | **71.0 %** | 18.99 ± 1.49 |

**Distribution-level metrics agree to 0.5 pp**, suggesting at first
glance that the API is a no-op. But coordinate-level inspection
shows otherwise:

| Coordinate comparison (apo virtual vs apo chain-API) | Value |
|---|---:|
| Max abs coord diff | 1.17 Å |
| Mean abs coord diff | 0.0065 Å |
| Fraction of atom-coord values bitwise identical | 48.8 % |
| Fraction within 0.01 Å | 67.2 % |
| Per-frame RMSD A-vs-B (matched index) | 0.018 ± 0.025 Å, max 0.148 Å |

So the chain_id signal **is** being propagated — outputs differ in ~half
the atoms — but the per-frame differences are tiny (sub-Å) and don't
move the population-level statistics. The positional embedding is a
small minority of the model's input signal; the IPA frames and aatype
embeddings dominate.

## H2 directional under Phase-1

| Protocol | Apo P(auto-inh.) | Holo P(auto-inh.) | Δ |
|---|---:|---:|---:|
| MarS-FM virtual-concat | 77.0 % | 71.0 % | **+6.0 pp** |
| MarS-FM Phase-1 chain-API | 77.5 % | 71.0 % | **+6.5 pp** |

Direction preserved; magnitude essentially unchanged. The Phase-1 API
does not move us closer to the pre-registered holo ≤ 30 % threshold.

## Interpretation

The Phase-1 plan was honest about its limitation:

> The trained model has learned peptide-bond locality from MD-Cath, so
> it will still implicitly assume adjacent residues are bonded. The
> chain-aware positional embedding just stops telling the model
> "residue 175 is at position 175"; instead it says "residue 175 is at
> position 0 of chain 1". Whether this is *enough* to produce
> non-pathological ensembles on real multi-chain inputs is the
> empirical question this phase answers.

The empirical answer: **Phase-1 alone is not enough**. The positional
embedding contribution to the MarS-FM output is too small to move
distribution-level metrics on real multi-chain inputs. The model
inherits its multi-chain behaviour primarily from the IPA frames and
aatype embeddings, neither of which know about chains under Phase-1.

This **strongly motivates Phase 2**: to actually shift the output
distribution, we need to either

1. **Add a chain-aware pair representation** (`c_z > 0` in
   `ipa_args`) with AlphaFold-Multimer-style relative position
   encoding (within-chain relpos + sentinel cross-chain bin), and
2. **Retrain** with multi-chain data so the new pair signal is
   actually used by the model weights, OR
3. **Construct a chain-break attention mask** that prevents the
   transformer layers from attending across chain boundaries during
   inference (a more aggressive Phase-1.5 hack, no retrain).

Option 3 is the next thing worth trying inference-time before
committing to a full retrain. Option 1+2 is the proper solution.

## Negative-result value

This result is not a failure — it's a **falsified hypothesis** with
clear directional implication. Specifically:

- **Validates** that the multi-chain API is implementable as a
  backward-compatible inference-only change (no upstream regression
  risk).
- **Falsifies** the naive hope that swapping the positional embedding
  is sufficient to make the model chain-aware at inference time.
- **Identifies the next experiment** (attention masking at chain
  breaks; Option 3 above) and the eventual fix (pair representation
  + retrain).

## Files + outputs

- Fork source: `/tmp/mars-fm-fork/` (modified MarS-FM, ~50 LOC patch)
- Adapter: `md/marsfm_modal.py` (emits `chain_id.npy`, overlays fork)
- New trajectories on disk:
  - `notch1_apo_chainAPI_smoke_marsfm.npz` (5-frame smoke test)
  - `notch1_apo_chainAPI200_marsfm.npz` (200-frame apo, real chain_id)
  - `notch1_holo_chainAPI200_marsfm.npz` (200-frame holo, real chain_id)

## Next steps to discuss with the MarS-FM team

1. **Open the upstream issue** referencing both
   (a) the chimerax-vampnet Notch1 NRR result and
   (b) this falsified Phase-1 negative-result as motivation for a
   pair-representation channel + multi-chain retrain.
2. **Offer a draft PR** of the Phase-1 API as the API surface for
   Phase 2 to plug into. The patch is small and backward-compatible.
3. **Offer the Notch1 NRR v0.3 MD trajectories** (apo + holo with the
   working COM-distance restraint) as held-out multi-chain
   evaluation data for the Phase-2 retrain.
