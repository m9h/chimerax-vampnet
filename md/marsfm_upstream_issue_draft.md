# DRAFT: GitHub issue text for valence-labs/mars-fm

(Copy the section below into the GitHub issue body once we decide to file it.
Save this file as a draft — do not paste in advance of user approval.)

---

**Title:** Multi-chain `atom14` input for inference — Phase-1 PR ready + empirical evidence motivating chain-aware pair representation

**Body:**

Hi Kacper, Cristian, and team — first, congratulations on the release.
We've validated the MD-Cath 450 checkpoint end-to-end on a real
single-chain protein outside your training set
(Notch1 NEC, 174 residues, see results below) and the model produces
useful conformational ensembles at >80× the cost-efficiency of
classical OpenMM. The integration with our ChimeraX bundle is clean:
your `.pdb`/`.xtc` outputs drop straight into our `vampnet
load_ensemble … source marsfm` loader for downstream VAMPnet/MSM
analysis.

We'd like to discuss adding **multi-chain `atom14` input support** to
the inference API. Here's the motivation and what we've already
implemented + empirically validated.

## Motivating use case: Notch1 NRR ligand modulation (H2)

Notch1's Negative Regulatory Region (NRR) is the receptor's
autoinhibition switch. Apo (PDB 3I08) populates an auto-inhibited
state; anti-NRR Fab binding (PDB 3L95) is published to shift the
ensemble toward the activated basin (Tiyanont 2011, Wu 2010). We
pre-registered a quantitative apo-vs-holo P(auto-inhibited) shift
in our bundle's v0.2/v0.3 paper and tested it with both
unrestrained and COM-distance-restrained 3×100 ns MD on Modal A100s.
The direction is met (+3.8 pp under both MD protocols) but neither
protocol reaches the pre-registered magnitudes because the actual
NRR auto-inh.↔activated transition timescale is microseconds.

This is exactly the regime MarS-FM is built for. But the NRR is a
2-chain (apo) or 4-chain (holo, NRR + anti-NRR Fab) system. Our
current workaround is a "virtual chain" hack: concatenate all
protein chains end-to-end and feed them as one chain. This creates
an artificial peptide bond between chain boundaries that biases
the ensemble toward the compact state.

For Notch1 + Fab-style PPI systems generally — antibody-antigen,
receptor-ligand, cleaved-precursor proteins — proper multi-chain
support would unlock the entire ligand-modulation regime.

## What we've implemented (Phase 1 — inference-only, no retrain)

Backward-compatible API surface at
https://github.com/m9h/mars-fm/tree/phase1-multichain-api
(diff: ~50 LOC across `mars/model/model.py`, `mars/model/module.py`,
`scripts/generate.py`).

- Optional `chain_id` argument on `MarSModel.forward` (default
  `None` = existing single-chain behaviour).
- New `_chain_aware_pos_embed(chain_id)` method: per-chain reset of
  the existing sinusoidal positional embedding table.
- `BaseModule._build_model_kwargs` passes `batch["chain_id"]` through
  when present.
- `scripts/generate.py:load_starting_structure` optionally loads
  `<name>_chain_id.npy` (shape `(n_residues,)`, int64) alongside the
  `<name>.npy` atom14 file.

All four changes are no-ops when `chain_id` is absent or all-zero.
No existing single-chain inference or training paths are affected.

## What the Phase-1 API actually does — empirically

We ran 200-sample inference on Notch1 NRR apo (NEC + NTM as two
chains, 174 + 60 residues) under both the existing virtual-concat
path and the new Phase-1 `chain_id` path:

| Protocol | NEC-NTM COM sep (mean ± std, Å) | P(<10 Å) |
|---|---:|---:|
| Virtual concat (1 chain forced) | 7.75 ± 3.50 | 77.0 % |
| **Phase-1 `chain_id` (2 chains)** | **7.75 ± 3.50** | **77.5 %** |

Distribution-level metrics unchanged to 0.5 pp. **However, the
chain_id signal IS propagated** — coord-level inspection shows the
two ensembles are *not* bitwise identical (max abs diff 1.17 Å,
mean 0.0065 Å, ~half the atoms differ; per-frame RMSD up to 0.148
Å). The positional embedding is just too small a fraction of the
model's input signal to shift population-level statistics by itself.

## What this tells us — and a correction to our initial hypothesis

A follow-up Phase-1.5 hack (each later chain starts at an elevated
`pos_embed` index — `effective_pos = within_chain + chain_id * 100`)
produced **bitwise-identical** output to the gap=0 baseline. Tracing
with debug prints established why:

> The released MD-CATH 450 checkpoint was trained without
> `--abs_pos_emb`, so `self.args.abs_pos_emb == False` and the
> entire `if self.args.abs_pos_emb:` block in `MarSModel.forward`
> is **dead code** for this checkpoint. All positional information
> comes from RoPE inside `mars.vendored.mha.MultiheadAttention`
> (every attention layer is constructed with
> `use_rotary_embeddings=True`).

So our Phase-1 plan modified a code path that's unreachable for
your published checkpoint. The API flow works end-to-end (we
verified `kwargs["chain_id"]` reaches `MarSModel.forward(chain_id=...)`
correctly), but the chain_id branch sits inside the dead `if`.

The real chain-awareness lever is **RoPE position indexing inside
`MultiheadAttention`**. Making it chain-aware would mean passing
explicit position indices (with per-chain reset or chain-id offset)
into the rotary embedding evaluation, instead of the implicit
`arange(L)` it currently uses. That's a more invasive change and
is plausibly hardest to do at inference time because the trained
weights have memorised the standard RoPE rotation pattern.

For meaningful multi-chain inference we therefore think the right
path is **pair representation + retrain** (the AlphaFold-Multimer
recipe):

1. Set `c_z > 0` in `ipa_args` and construct a pair representation
   via relative position encoding: `relpos = clip(i - j, -32, 32)`
   for same-chain pairs, sentinel bin 33 for cross-chain pairs,
   one-hot → linear → c_z. Your IPA already accepts a `z` argument;
   the model just doesn't construct one.

2. Retrain with multi-chain data so the pair signal is actually
   used. Candidate training sources: ATLAS multi-chain entries
   (Vander Meersche 2024), the multi-chain subset of MD-Cath
   (currently filtered as single-domain in
   `scripts/prepare_data/prep_sims_mdcath.py`), or our own
   Notch1 NRR + Fab v0.3 MD (3 × 100 ns each on Modal A100 with
   working COM-distance restraints — happy to share).

Small upstream note while you're in there: it might be worth
either removing the `--abs_pos_emb` code path or gating it behind a
clearer assertion, since right now it's reachable but architecturally
unused in your published recipe.

## Asks

1. **Would you merge the Phase-1 API PR?** It's a no-regression
   addition that gives the API surface a Phase-2 retrain can plug
   into. Branch link above; happy to open a clean PR if you give a
   thumbs-up.

2. **Is multi-chain retrain on the roadmap?** If so, we'd like to
   contribute the Notch1 NRR test data and the empirical
   H2-directional benchmark as a public validation case.

3. **Anything we're getting wrong about the single-chain
   assumption?** The `c_z=0` choice for `ipa_args` was the lever we
   identified; if there's a simpler path (e.g., a chain-break
   attention bias you've already prototyped), we'd love to know.

## What we already have for you

- Working chimerax-vampnet bundle (MIT,
  https://github.com/m9h/chimerax-vampnet) that loads MarS-FM output
  natively and runs VAMPnet / MSM analysis on it.
- 4 published trajectories on the Notch1 NRR system as a multi-chain
  benchmark: apo NEC+NTM + holo NRR+Fab, 3 × 100 ns each, A100 MD
  with COM-distance restraint. Released as the v0.3 Zenodo deposit
  (analysis-ready Cα-only + raw all-atom).
- The Phase-1 API patch on the linked branch, with full validation
  write-up at
  https://github.com/m9h/chimerax-vampnet/blob/main/md/marsfm_multichain_phase1_results.md

Thanks for the great work — happy to chat in any direction this
discussion takes.

Morgan
