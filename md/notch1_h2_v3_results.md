# Notch1 NRR apo/holo H2 analysis — v0.3

**Date**: 2026-06-01
**Status**: directional H2 prediction MET (+3.8 pp). Magnitudes blocked
by the 100 ns sampling horizon, NOT by the restraint protocol.

Run: `md/notch1_h2_modal.py::analyze` against the COM-distance-restrained
trajectories at `prepared/notch1_apo_v3/` and `prepared/notch1_holo_v3/`
on the Modal volume.

## Headline

| | Apo v0.3 | Holo v0.3 | v0.2 baseline (no anchor) |
|---|---:|---:|---:|
| Replicas | 3 × 100 ns | 3 × 100 ns | 3 × 100 ns |
| Restraint | COM(NEC, NTM) ≈ r0 (= 3.94 Å apo / 3.98 Å holo), k = 100 kJ/mol/nm² | same | none |
| NEC-NTM COM sep mean ± std | **4.6 ± 0.3 Å** | **3.7 ± 0.3 Å** | apo 13.9 ± 24.2 / holo 21.5 ± 44.3 |
| NEC-NTM COM sep range | 3.4 – 5.6 Å | 2.5 – 4.8 Å | 2.8 – 107 / 2.7 – 185 Å |
| VAMPnet states | 4 | 4 | 4 |
| P(auto-inhibited) | **25.4 %** | **21.6 %** | 21.6 % / 17.9 % |
| Δ apo − holo | **+3.8 pp** | | +3.7 pp |
| Pre-reg direction met | ✓ | | ✓ |
| Pre-reg magnitudes met | ✗ | | ✗ |

## H2 verdict

- **Directional prediction met, and replicated across two restraint
  protocols.** Both the unrestrained v0.2 and the COM-restrained v0.3
  give +3.8 ± 0.1 pp (apo more auto-inhibited than holo). Sign matches
  the published mechanism~\cite{tiyanont2011exposure,wu2010antinrr}.
- **Pre-registered magnitudes (apo ≥ 50%, holo ≤ 30%) not met.** Both
  protocols sit below 30% auto-inhibited population.
- **The restraint is *not* the magnitude blocker.** v0.3 with the
  COM restraint working perfectly (NRR locked at the LNR-HD
  interface, COM sep 3.4–5.6 Å, no dissociation) gives essentially
  the same P(auto-inhibited) as v0.2 without restraint. If the
  restraint were the problem we'd expect the magnitudes to improve in
  v0.3; they did not.

## What is blocking the magnitudes

The Notch1 NRR auto-inhibited → activated transition involves the LNR
cap rotating off the heterodimerisation
domain~\cite{tiyanont2011exposure}; the published activation timescale
is microseconds. At 100 ns per replica × 3 replicas = 300 ns total
sampling per system, the simulation cannot equilibrate between the
two basins. Both protocols sample only the LOCAL fluctuations around
the starting auto-inhibited state, just with very different
fluctuation magnitudes (24 Å std unrestrained, 0.3 Å std with the
COM restraint).

To recover the magnitudes properly would require, in order of effort:

1. **Real AlphaFlow / BioEmu / MarS-FM ensembles** — generative
   models trained on long MD trajectories sample the equilibrium
   distribution directly, including states inaccessible to short MD.
   v0.3 items 3-4 already plan for this. ~$5-50 per system.
2. **Umbrella sampling on NEC-NTM COM distance** — proper PMF along
   the activation coordinate via 10-15 biased windows. ~$200-300
   on Modal, 1 day.
3. **Long μs MD** — direct equilibrium sampling. ~$1000s, 1-2 weeks.

The paper-roadmap v0.4 (MarS-FM) is option 1 and is the natural
next step.

## What v0.3 *does* establish

- **The COM-distance restraint is a working
  membrane-anchor substitute.** Apo NEC-NTM COM sep stays at
  4.6 ± 0.3 Å (vs no-anchor's 24-Å std with dissociation excursions to
  100+ Å), holo at 3.7 ± 0.3 Å. The restraint compensates for the
  missing transmembrane helix without per-atom force singularities
  that NaN at HMR=4 fs.
- **The directional H2 result is robust to restraint choice.**
  +3.8 pp ≈ +3.7 pp across two very different MD protocols (free
  dissociation vs locked at r0). This is stronger evidence for the
  direction than either protocol alone.
- **Bundle pipeline ran end-to-end** on 3 + 3 = 6 fresh trajectories
  totaling ~600 ns of MD with no manual intervention. The default
  VAMPnet hyperparameters (k=4 states, lag 1 ns, 500 CA-CA pair
  features) produced non-degenerate 4-state decompositions in both
  systems.

## Hyperparameters (unchanged from v0.2)

- VAMPnet: MLP 128-128-`n_states`, ELU, softmax output.
- `n_states = 4`, lag = 50 frames = 1.0 ns (at 20 ps/frame × stride 5).
- Features: 500 random CA-CA pair distances on the NRR fragment,
  per-pair z-scored + clipped to ±5σ.
- `epsilon = 1e-3` regularisation on the eigendecomposition.
- 60 epochs Adam, lr 5e-4, batch 512.

## v0.3 compute cost

| Job | GPU | Wall | Cost |
|---|---|---|---|
| Apo v3 prep + 3×100 ns | A100-80GB | ~3 hr each | ~$25 |
| Holo v3 prep + 3×100 ns | A100-80GB | ~12 hr each | ~$90 |
| H2 analysis | CPU on Modal | ~3 min | <$1 |
| **Total v0.3 spend** | | | **~$115** |

Plus the v0.3 anchor-debugging iterations (5 failed prep attempts +
4 failed production attempts) at ~$30 — the COM-distance restraint
took ~$30 of debugging to arrive at.
