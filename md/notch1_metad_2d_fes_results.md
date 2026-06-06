# Notch1 apo 2D metad — v0.7 W4c result

**Date**: 2026-06-06
**System**: Notch1 NRR apo (v0.3 prepared, NEC chain A + NTM chain B
+ solvent under v0.3 COM-distance restraint at 3.94 Å)

Script: `md/metad_2d_fes_postprocess.py`
Adapter: `md/notch1_metad_modal.py::fanout_2d`
Backing data: `md/notch1_metad_2d_fes_data.npz`

## What was run

3 independent walkers × ~30 ns each of well-tempered 2D
metadynamics on Modal A100-80GB. Total cost ~$5. The two CVs:

- **d1** = NEC–NTM COM distance (chain A → chain B), same axis as
  the v0.6 1D metad and the v0.3 restraint.
- **d2** = intra-NEC LNR-A → HD-N COM distance (local CA indices
  0:35 → 119:174), the orthogonal axis that the v0.5 H3 biology
  analysis flagged as discriminating the AF3-class state 1 (22-24
  Å) from MD/MarS-FM states (27-29 Å).

PLUMED parameters:
- HEIGHT = 1.2 kJ/mol
- SIGMA = (0.05, 0.08) nm — d1 same as 1D, d2 slightly wider to
  match its larger native variation
- PACE = 1 ps
- BIASFACTOR = 10
- TEMP = 310 K
- 2D GRID 0:8 nm × 1:5 nm with 200 × 150 bins

## CV exploration per walker

| Walker | d1 range (nm) | d2 range (nm) | n Gaussians |
|---|---|---|---:|
| 1 | [0.00, 0.71] | [1.84, 4.84] | ~19 500 |
| 2 | [0.00, 1.44] | [1.87, 4.93] | ~19 500 |
| 3 | [0.03, 0.82] | [1.92, 5.01] | ~19 500 |

Walker 2 reached the dissociation regime (d1 up to 14.4 Å, past
the v0.6 1D barrier at 11 Å). Walkers 1 and 3 explored the basin
and the entry of the barrier but didn't fully cross.

The v0.6 1D metad PLUMED COM-distance artifact (d1 excursions to
~0 nm under strong bias) is also present here — the bimodal d1
explored range "[0.00, ...]" reflects PLUMED's COM-distance scalar
passing through zero when the two CA groups' centroids cross in
3-space. Mask d1 < 0.1 nm at analysis time.

## Merged 2D FES

58 640 total Gaussians across 3 walkers reconstructed onto a 100 ×
100 grid (d1 0.1–1.6 nm × d2 1.8–5.0 nm) via Gaussian sum, mean
across walkers, well-tempered correction (γ = 10):

  F(d1, d2) range: **0–407 kJ/mol**

The FES is stored in `md/notch1_metad_2d_fes_data.npz` (keys
`d1_grid_nm`, `d2_grid_nm`, `fes_kjmol`) for downstream rendering
or analysis.

## What this 2D map tells us (vs the 1D result)

The v0.6 1D metad on d1 alone recovered a dissociation barrier of
115 kJ/mol at d1 = 11 Å (apo). The v0.7 2D map extends this to a
joint FES on (d1, d2). The expected features:

- **Auto-inhibited basin** at (d1, d2) ≈ (0.4 nm, 2.7-2.9 nm) —
  v0.3 restraint set point + v0.5 MD-mean LNR-A→HD-N (~27-29 Å).
- **AF3-class state 1 region** at d2 ≈ 2.2-2.4 nm (tighter LNR-A
  → HD-N). Whether this overlaps an FES basin or sits on a slope
  is the testable hypothesis: if AlphaFlow/Boltz-2 sample a real
  metastable basin, we should see a local FES minimum near
  d2 = 2.2-2.4 nm and the same d1 as MD; if they sample a slope,
  no minimum.
- **Dissociation channel** along d1 → 1-2 nm, with d2 free.
  Walker 2 sampled this regime.

The figure (which would overlay MD scatter + generative-source
d2 marginals onto the FES contours) hung during rendering due to
heavy system CPU contention from other workloads (load average
11+ during the v0.7 W4c run). The FES data is saved; the figure
can be re-rendered from `md/metad_2d_fes_postprocess.py` in a
quieter environment.

## Open / queued

- **2D FES figure rendering** (blocked on quieter system; just
  matplotlib contour from the saved .npz).
- **2D FES → 1D marginal**: integrate exp(-F(d1,d2)/kT) along d2
  to recover the 1D F(d1) and cross-check against the v0.6 result.
- **Per-generative-source-d2 marginal overlay**: identify whether
  each sampler's d2 distribution sits in an FES basin or slope.
- **Walker 2 dissociation tail**: walker 2 reached d1 = 14.4 Å,
  past the v0.6 1D barrier. Cross-check the 2D barrier height
  here against the 1D result.

## Cost (v0.7 W4c)

- 3 walkers × 30 ns each: ~$5 in Modal A100-80GB time
- Gaussian sum + analysis: local CPU, $0
- Cumulative v0.7 spend: ~$45 ($30 Hsp90 MD apo + $30 Hsp90 MD
  holo + $1.20 Hsp90 generative + $5 holo metad walker 4 + $5 2D
  metad - $20 deferred from W3d Phase 2 which never produced
  results due to disulfide bond constraints on the morph approach)
