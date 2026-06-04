# H2 magnitudes from Boltzmann-integrated v0.6 metad FES

**Date**: 2026-06-03
Script: `md/metad_h2_reweight.py`

## Method

P(d < cutoff) = ∫ exp(-F(d)/kT) dd over d in [0, cutoff]
                / ∫ exp(-F(d)/kT) dd over the full grid

F(d) is the v0.6 well-tempered metad FES along the
NEC-NTM COM distance (`md/notch1_metad_fes_data.json`),
merged across 3 walkers × 20 ns each. T = 310 K.

## Results

| cutoff (Å) | P_apo | P_holo | Δ (pp) |
|---:|---:|---:|---:|
| 5.0 |  98.4 % | — | — |
| 7.0 | 100.0 % | — | — |
| 10.0 | 100.0 % | — | — |

## Pre-registered H2 vs metad reweight

Pre-registered (v0.1):
- apo P(auto-inhibited) ≥ 50 %
- holo P(auto-inhibited) ≤ 30 %
- Δ ≥ 20 pp (apo − holo)

v0.6 apo result at cutoff 5 Å: **P_apo = 98.4 % ≥ 50 % MET ✓**

Holo metad FES pending; rerun this script with
`--holo-fes md/notch1_metad_holo_fes_data.json` when it lands.

## Caveats

1. The reweight is along a SINGLE CV (NEC-NTM COM distance).
   The v0.5 H3 analysis revealed a 3-way sampler-class split
   along other axes (LNR-A → HD-N, Rg). Auto-inhibition vs
   activation may not project cleanly onto just the NEC-NTM
   COM axis; v0.7 could repeat with a 2D FES (this CV +
   LNR-A → HD-N).
2. The metad FES at very small d (< 1 Å) is affected by
   walker 2/3's pathological PLUMED COM excursions
   documented in `md/notch1_metad_results.md`. Cutoffs at
   5 Å and above are unaffected.
3. P_apo is set largely by the well minimum location and
   width; the absolute number depends on the integration
   range (here [0, 8 nm]). Cutoffs further from the minimum
   exhibit larger numerical sensitivity to the integration
   tail.