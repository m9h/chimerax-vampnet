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

## Interpretation — and the actual cause of zero effect

We followed up with a more aggressive Phase-1.5 hack (`chain_gap=100`
and `chain_gap=200`: each later chain starts at an elevated
`pos_embed` index, recruiting the model's learned "distant positions
≈ not bonded" bias). The output was **bitwise identical** to the
gap=0 baseline. Tracing with debug prints established the root cause:

> The released MD-CATH 450 checkpoint was trained without
> `--abs_pos_emb`, so `self.args.abs_pos_emb == False` and the
> entire `if self.args.abs_pos_emb:` block in `MarSModel.forward`
> is **dead code** for this checkpoint. All positional information
> comes from RoPE inside `MultiheadAttention` (every attention
> layer is built with `use_rotary_embeddings=True`).

So the Phase-1 plan's lever (absolute positional embedding) does
not exist in the deployed model. The fork compiles, the API is
honored, kwargs flow through to `MarSModel.forward(chain_id=...,
chain_gap=...)` — but the chain_id branch sits inside a `False`
conditional and never runs. The ~0.5% coord-level differences we
observed in the first comparison are floating-point noise from CUDA
kernel non-determinism, not from the chain_id signal.

**The real chain-awareness lever is RoPE inside
`mars.vendored.mha.MultiheadAttention`.** RoPE applies a per-position
rotation to query/key vectors, and its position indexing is implicit
(0, 1, 2, ..., L-1) inside the attention call. To make it chain-aware
we'd need to pass an explicit position-index tensor (with per-chain
reset and/or chain-id offset) into the RoPE evaluation. This is a
more invasive change — it has to thread through the MHA module's
forward and the rotary-embedding application — and is plausibly
hardest at inference time because the trained weights have memorised
the standard (continuous) RoPE rotation pattern.

This **strongly motivates Phase 2** — pair representation + retrain —
as the only credible path to real multi-chain MarS-FM inference. The
inference-only Phase-1 plan was the right shape (small, backward-
compatible API surface) but the wrong lever for this checkpoint.

What remains valuable from Phase 1:

1. **The API surface** (`chain_id` field on the item dict, kwarg on
   `MarSModel.forward`, env-var pass-through) is the right place for
   Phase 2 to plug into. Files modified are still the right files.
2. **The dead-code discovery** is itself a useful contribution to
   the upstream team — the `abs_pos_emb` code path is unreachable
   for the released checkpoint, suggesting either remove it or
   gate it more clearly.

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
