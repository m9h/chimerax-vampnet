# Notch1 NRR apo/holo H2 analysis — v0.2 first attempt

**Date**: 2026-05-28
**Status**: pre-registered H2 NOT met; result invalidated by holo-prep bug.

Modal run: `md/notch1_h2_modal.py::analyze` against the corrected (May-27)
trajectories on volume `chimerax-vampnet-md`.

## Headline

|  | Apo | Holo |
|---|---:|---:|
| n frames analysed (3 reps, stride 5) | 3000 | 3000 |
| NEC CAs | 174 | 230 |
| NTM CAs | 60 | 212 |
| NEC-NTM COM sep (mean ± std) | 13.9 ± 24.2 Å | 104.9 ± 25.8 Å |
| NEC-NTM COM sep range | 2.8 – 107.2 Å | 56.0 – 153.3 Å |
| VAMPnet states | 4 | 4 |
| Auto-inhibited state population | 21.6 % | 20.8 % |
| Slowest implied timescale | 59.0 ns | (none resolved) |

**Pre-registered H2**: apo P(auto-inhibited) ≥ 0.5 ; holo ≤ 0.3. → NOT met.

## Why this result is not a real H2 test

Two independent prep bugs invalidate the comparison:

### Bug 1 — holo system contains no Notch1 NRR

`md/3l95_holo.pdb` was generated with
`python filter_chains.py 3l95.pdb 3l95_holo.pdb A B H L` (per `md/README.md`).
But the raw `3l95.pdb` is the **dimer** asymmetric unit:

| Chain | CAs | Identity (from N-terminal residues) |
|---|---:|---|
| A | 212 | Fab light, copy 1 (DIQ N-term) |
| B | 220 | Fab heavy, copy 1 (EVQ N-term) |
| **X** | **230** | **Notch1 NRR, copy 1 (resid 1447–1727)** |
| L | 212 | Fab light, copy 2 |
| H | 219 | Fab heavy, copy 2 |
| **Y** | **216** | **Notch1 NRR, copy 2 (resid 1461–1726)** |

The `A B H L` filter kept **all four Fab chains** and dropped both NRR copies.
The May-27 holo redo simulated 3×100 ns of two Fab antibodies in solvent —
no Notch1. The 877-CA system reported above contains 4 Fab chains and
zero NRR atoms. The "NEC vs NTM" COM separation of ~105 Å is the gap
between the two Fab copies, not anything biological.

**Fix**: re-filter with `X H L` (NRR + one Fab) or `X Y H L` (both NRR +
Fab dimer). The NRR in 3L95 is a single chain spanning NEC+NTM rather
than the proteolytically cleaved A+B split present in 3I08, because the
crystallised construct is the uncleaved precursor.

### Bug 2 — apo MD lacks membrane-anchor restraints

Apo NEC-NTM COM separation ranges 2.8–107.2 Å with std 24.2 Å. The
upper-tail excursions to 50–107 Å indicate the NTM heterodimerisation
domain is partially dissociating from the LNR cap during the 100 ns
trajectory — physically unrealistic for the membrane-anchored
protein *in vivo*, where the transmembrane helix downstream of NTM
tethers the fragment to the lipid bilayer.

This depresses the apo auto-inhibited population (state 2: 21.6 %,
8 Å) below the pre-registered 0.5 threshold, because the simulation
spends time in dissociated configurations the bound apo form would
not access. The auto-inhibited state itself *is* identified (closest
NEC-NTM separation = 8.1 Å; the next-closest state at 11.7 Å looks
like a transition intermediate) — H1 (state resolution) is provisionally
met. H2 (population shift) cannot be tested at this prep.

**Fix**: positional restraints on NTM residues bordering the missing
transmembrane region, or harmonic COM-distance restraint between NEC
and NTM centres. Both are paper-roadmap v0.2 items.

## What is salvageable from this run

- **Apo bundle pipeline ran end-to-end** on a system the bundle has
  never seen before (large multi-chain solvated MD), recovering a
  non-degenerate 4-state decomposition with a non-trivial slow
  timescale and an auto-inhibited state at the lowest COM separation.
  This validates the bundle's analysis side.
- **The pre-registration mechanism itself works**: the script's
  per-state COM-separation labelling automatically tagged the
  smallest-separation state as auto-inhibited, no human in the loop.
- **The H2 result is reportable as a diagnostic finding**: the
  v0.1 prep pipeline had two independent silent failure modes
  (wrong-chain filter, missing membrane anchor) that only surfaced
  at downstream analysis. The fact that the system completed MD
  cleanly with no chain-selection errors despite being two Fabs in a
  box is itself a lesson about MD-prep observability.

## Re-run requirements

1. Fix `md/3l95_holo.pdb` (re-filter raw `3l95.pdb` with `X H L`).
2. Add membrane-anchor restraints in `md/prep.py` (Tier-2 NRR systems).
3. Re-prep apo + holo on Modal (~30 min wall, ~$2).
4. Re-launch 3×100 ns × 2 systems on Modal (~22 hr wall, ~$150).
5. Re-run `md/notch1_h2_modal.py::analyze`.

Total: ~1 day wall, ~$150.

## Hyperparameters used here (for the v0.2 re-run)

- VAMPnet: MLP 128-128-`n_states`, ELU, softmax output.
- `n_states = 4`, lag = 50 frames = 1.0 ns (at 20 ps/frame × stride 5).
- Features: 500 random CA-CA pair distances on the NRR fragment,
  per-pair z-scored + clipped to ±5σ to handle outlier-driven
  ill-conditioning of the Koopman covariance.
- `epsilon = 1e-3` regularisation on the eigendecomposition
  (default 1e-6 fails to converge on large solvated systems).
- 60 epochs Adam, lr 5e-4, batch 512.
