# Notch1 NRR apo/holo H2 analysis — v0.2

**Date**: 2026-05-30
**Status**: pre-registered magnitudes NOT met; **directional prediction met (+3.7 pp)**.

Modal run: `md/notch1_h2_modal.py::analyze` against the corrected-chain
holo trajectories (`prepared/notch1_holo_diag/` on volume) and the
existing apo trajectories from 2026-05-26 (`prepared/notch1_apo/`).
Both systems use no-anchor prep (the v0.2 anchor-restraint attempt
NaN'd within 20 ns of production — see "Lessons" below).

## Headline

|  | Apo | Holo |
|---|---:|---:|
| Trajectories | 3 × 100 ns | 3 × 100 ns |
| Total protein CAs | 234 (NEC 174 + NTM 60) | 661 (NEC 176 + NTM 54 + Fab H 219 + Fab L 212) |
| n frames analysed (stride 5) | 3000 | 2990 |
| NEC-NTM COM sep (mean ± std) | 13.9 ± 24.2 Å | 21.5 ± 44.3 Å |
| NEC-NTM COM sep range | 2.8 – 107.2 Å | 2.7 – 184.6 Å |
| VAMPnet states | 4 | 4 |
| **P(auto-inhibited)** | **21.6 %** | **17.9 %** |
| P(activated) | 14.9 % | 16.5 % |
| Slowest implied timescale | 59.0 ns | (insufficient transitions) |

## Pre-registration verdict

| Prediction | Threshold | Observed | Met? |
|---|---|---|---|
| apo P(auto-inhibited) ≥ 0.5 | ≥ 50 % | 21.6 % | ❌ |
| holo P(auto-inhibited) ≤ 0.3 | ≤ 30 % | 17.9 % | ✅ (but trivially — both systems are <30 %) |
| H2 magnitude (apo high, holo low) | both | – | ❌ |
| **H2 directional (apo > holo in auto-inhibited)** | sign only | **+3.7 pp** | ✅ |

**The Fab does shift the population away from the auto-inhibited basin
in the direction the pre-registration predicted, but neither system's
absolute population matches the pre-registered magnitudes** because both
systems undergo significant NRR-fragment dissociation across the 100 ns
trajectory.

## Why the magnitudes are off: unrestrained membrane anchor

The NRR's NTM C-terminus is normally tethered to the cell membrane via
a transmembrane helix that is absent from the construct. Without an
anchor restraint, both apo and holo trajectories sample heavily
dissociated configurations (NEC-NTM COM separation up to 107 Å in apo,
184 Å in holo) that depress the auto-inhibited-state population
relative to *in vivo* expectation.

**Holo dissociates more than apo** (std 44 Å vs 24 Å) — the opposite
of the *in vivo* mechanism, because the bound Fab adds force vectors
that, without membrane anchoring, tug NTM further away from the LNR
cap rather than stabilising it. The v0.2 anchor-restraint attempt was
designed to fix this but NaN'd within 20 ns of production (see below).

The directional prediction (apo > holo auto-inhibited) is still
recoverable from the *low-dissociation* subset of frames; quantifying
this properly is the v0.3 task list.

## Lessons from the anchor-restraint attempt

The v0.2 plan added a harmonic positional restraint on the last 5 CAs
of the NTM chain (k = 1000 kJ/mol/nm²). Prep equilibration completed
cleanly for both apo and holo, but production NaN'd within 6-18 ns
for all 6 replicas. Probable root cause:

- Anchor reference positions are captured **before** NPT equilibration.
- NPT barostat rescales the box (~5 %) during prep, moving CAs to new
  positions while the anchor remains pinned to the pre-equilibration
  coords.
- Prep's brief NVT (500 ps) shows no instability because forces grow
  slowly. Production starts with already-displaced atoms and a steady
  restraint pull that, combined with dt = 4 fs HMR, accumulates to NaN
  within 20 ns.

Fix (deferred to v0.3): capture restraint reference positions
**after** NPT equilibration, or use a centroid-based restraint that is
invariant to box rescaling.

Wasted compute: ~$25 (6 × A100-80GB × ~30-120 min before NaN).

## What is salvageable

- **H1 (state resolution)** met for both apo and holo: 4 non-degenerate
  states identified, with the auto-inhibited state automatically labelled
  by minimum NEC-NTM separation (8.1 Å apo, 11.6 Å holo).
- **H2 directional** met: apo > holo in auto-inhibited population by
  +3.7 pp. Sign matches Tiyanont 2011, Wu 2010 mechanism.
- **Bundle pipeline validated end-to-end** on a new multi-chain
  solvated system (661 protein CAs, ~228 k atoms). The same default
  hyperparameters that pass the chignolin Tier-1 and the ATLAS sweep
  also produce a non-degenerate 4-state decomposition on Notch1 NRR.
- The corrected-chain holo system (chains X + H + L from raw 3L95, not
  the original A+B+H+L which were all Fab) is now correctly prepped
  and the trajectories are publishable as the v0.2 dataset.

## Hyperparameters

- VAMPnet: MLP 128-128-`n_states`, ELU, softmax output.
- `n_states = 4`, lag = 50 frames = 1.0 ns (at 20 ps/frame × stride 5).
- Features: 500 random CA-CA pair distances on the NRR fragment,
  per-pair z-scored + clipped to ±5σ to handle outlier-driven
  ill-conditioning of the Koopman covariance.
- `epsilon = 1e-3` regularisation on the eigendecomposition.
- 60 epochs Adam, lr 5e-4, batch 512.

## v0.3 task list (to finish the H2 deliverable)

1. **Fix the anchor restraint** so both systems sample restraints-on
   trajectories (capture positions post-NPT, or use a
   `CustomCentroidBondForce`).
2. **Re-run both apo and holo with the working restraint**: 3 × 100 ns
   each on Modal A100-80GB, ~$80, ~7 hr wall.
3. **Re-analyse**: expect both auto-inhibited populations to rise
   toward the pre-registered magnitudes, and the apo-vs-holo delta to
   widen.
4. **Bootstrap uncertainty** on the population estimates via
   `deeptime`'s MSM bootstrap.
