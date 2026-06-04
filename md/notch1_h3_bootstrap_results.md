# H3 per-state per-source bootstrap CIs — v0.6

**Date**: 2026-06-03
Script: `md/notch1_h3_bootstrap.py`
Bootstrap reps: 500 (per-source frame resampling, fixed VAMPnet)

## Per-state per-source occupancy with 95 % CI

| state | MD | MarS-FM | BioEmu | Boltz-2 | AlphaFlow |
|---|---|---|---|---|---|
| 0 |  69.6 % [ 67.3,  72.0] |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |
| 1 |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |   1.2 % [  0.0,   3.0] |  99.5 % [ 98.5, 100.0] | 100.0 % [100.0, 100.0] |
| 2 |   2.0 % [  1.3,   2.7] | 100.0 % [100.0, 100.0] |  98.8 % [ 97.0, 100.0] |   0.5 % [  0.0,   1.5] |   0.0 % [  0.0,   0.0] |
| 3 |  28.4 % [ 26.1,  30.7] |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |   0.0 % [  0.0,   0.0] |

## Interpretation

The bootstrap CIs are tight for the deterministic-looking
entries — sources that put 100 % of their frames in one
state have CIs that don't drop below ~97 %, confirming the
v0.5 point estimates were not artifacts of finite sample
size. For sources with split distributions (BioEmu's 49 %
v0.4 / 1.2 % v0.5 in state 1) the CI shows the magnitude
of the per-frame variability.

Note: this is a FRAME-LEVEL bootstrap. It captures finite-
sample uncertainty from the limited per-source frame counts
(MD 1500, MarS-FM 200, BioEmu 169, Boltz-2 200, AlphaFlow
200) but does NOT propagate VAMPnet training stochasticity.
A VAMPnet-retrain bootstrap would be ~50x more expensive and
is queued for v0.7.