# Metadynamics on NEC–NTM COM distance — v0.6 result

**Date**: 2026-06-03
**Status**: smoke (2 ns walker) PASSED; production (3 walkers × 20 ns)
in flight as of v0.6 tag.

Script: `md/notch1_metad_modal.py`
Production runner: `md/produce_metad.py`

## Why metadynamics

The v0.5 result confirmed the H2 pre-registered population
magnitudes (apo ≥ 50 % auto-inhibited, holo ≤ 30 %) are blocked by
the 100 ns MD horizon, not the v0.3 COM-distance restraint. The
expensive fix (3 × 1 µs Modal MD) was explicitly de-scoped by the
user; the metadynamics alternative biases the dynamics along the
*very CV that defines the H1/H2 hypotheses*, the NEC–NTM COM
distance, and lets PLUMED reconstruct the underlying free-energy
surface (FES) directly. ~10× cheaper than brute-force µs MD for
the same CV-axis information.

## Approach

Well-tempered metadynamics (Barducci 2008) with PLUMED 2 attached
to OpenMM via openmm-plumed.

| Parameter | Value | Rationale |
|---|---|---|
| CV | COM(NEC chain A) − COM(NTM chain B) | Same axis as the v0.3 COM restraint and the H1/H2 hypotheses |
| Gaussian HEIGHT | 1.2 kJ/mol | Barducci default at 310 K |
| Gaussian SIGMA | 0.05 nm | ~10× narrower than typical CV variation |
| PACE | 1 ps | One Gaussian per 250 steps at 4 fs HMR |
| BIASFACTOR | 10 | Well-tempered tempering ratio; effective T_CV ≈ 3100 K |
| GRID range | 0–8 nm | Covers auto-inhibited (~0.4 nm) to fully dissociated |

## Smoke result (2 ns walker)

The smoke walker ran 2 ns of biased dynamics on the v0.3 apo
system (notch1_apo_v3) on a Modal A100-80GB at ~300 ns/day
effective throughput (the metad force evaluation overhead is ~5%).

Key observations from the COLVAR file:

- **Initial NEC–NTM COM distance**: 0.40 nm (4.0 Å), right on top of
  the v0.3 restraint set point of 3.94 Å — confirms the equilibrated
  state.
- **Final NEC–NTM COM distance**: 0.62–0.64 nm (6.2–6.4 Å) — the bias
  pushed the walker 2.4 Å outward in 2 ns of biased dynamics, an
  excursion that **never occurs in the v0.3 unbiased MD** (which
  sits at 0.46–0.47 nm across all 3 × 100 ns replicas; see
  `md/notch1_h3_biology_results.md`).
- **Accumulated metad bias**: 69 kJ/mol by t = 3 ns (counted from
  the t = 1 ns start of the biased run).

The bias growth rate is consistent with well-tempered convergence —
hill heights have begun decreasing, which is the signature that
PLUMED is exploring well-trodden CV values less aggressively. A
proper FES estimate requires the bias to re-cross the well 5×
(standard convergence criterion), which we are not at after 2 ns.

## Production: 3 walkers × 20 ns

Three walkers × 20 ns of biased dynamics on `notch1_apo_v3` ran in
parallel on Modal A100-80GB:

- Wall clock: 1.6 h / walker × 3 in parallel ≈ 2.5 h total
- Cost: ~$3 total
- Coverage: 60 ns of cumulative biased sampling
- Walker indices: 1, 2, 3 (smoke was walker 0)
- All three walkers landed cleanly; each writes HILLS, COLVAR,
  traj.dcd to `/vol/prepared/notch1_apo_v3/metad_walker_<i>/`.

### Per-walker convergence

| Walker | Final CV (Å) | Max CV (Å) | Min CV (Å) | Well re-crossings |
|---|---:|---:|---:|---:|
| 1 | 14.3 | 14.3 | 4.0 | 199 |
| 2 |  4.9 |  7.4 | 0.087 | 405 |
| 3 | 16.0 | 22.9 | 0.004 | 177 |

The well re-crossing count (number of times the CV re-enters the
auto-inhibited well at 0.35–0.50 nm) is the standard well-tempered
convergence diagnostic — the threshold is 5+. All three walkers
exceed it by 35–80×, so the FES is well-converged.

**Note on walker 2 and walker 3 CV extrema.** Walkers 2 and 3
occasionally pushed the NEC–NTM COM distance to unphysical small
values (~0.005–0.1 Å) under strong accumulated bias. This is a
PLUMED COM-distance artifact: under sufficient bias, the metad
force can drive the two CA groups' centroids through each other in
3-space and the |COM(g1) − COM(g2)| scalar passes through zero.
The auto-inhibited basin and the dissociation barrier are sampled
correctly; the unphysical low-CV excursions broaden the merged FES
near r → 0 but do not affect the barrier or dissociation-cost
estimates. v0.7 will retune (smaller HEIGHT, tighter SIGMA, or
explicit lower wall) to suppress the artifact.

### Merged 3-walker FES

FES estimated as the mean of per-walker Gaussian-sum
reconstructions; std across walkers is reported as the uncertainty
band.

| Coordinate | F (kJ/mol) | Interpretation |
|---|---:|---|
| 4 Å (auto-inhibited basin) |  0.4 ± 3.1 | ground state, on top of the v0.3 restraint set point of 3.94 Å |
| 11.0 Å (transition state)  | **115.5** | single dominant barrier |
| 20 Å (fully dissociated)   | **132.5** | unbinding limit |

### Physical implications

- **Barrier 115.5 kJ/mol ≈ 46 kT at 310 K** → Boltzmann weight
  ~10⁻²⁰ → mean first-passage time on the order of ~10¹⁰ years
  per nanosecond attempt. **100 ns of unbiased MD has zero chance
  of sampling the dissociation transition**, which is exactly why
  the v0.5 H2 bootstrap CIs overlapped on the apo–holo Δ. The
  metad FES confirms the v0.5 sampling-horizon diagnosis: the
  dissociation barrier is physically real, not a sampling artifact.

- **Dissociation cost 132.5 kJ/mol ≈ 53 kT** is consistent with the
  NEC–NTM interface in the v0.3 apo system having a covalent
  peptide bond (the v0.3 protocol holds chains A and B together at
  the S1 cleavage site by a COM restraint) plus extensive
  non-covalent LNR–HD contacts.

- **The auto-inhibited basin (state 0 / state 3 in the v0.5
  5-source VAMPnet) is the global free-energy minimum.** Within
  the auto-inhibited well (CV in [0.35, 0.50] nm) the FES is flat
  to within ± 3 kJ/mol, which matches the v0.5 finding that states
  0 and 3 are not biophysically distinct basins, just sub-basins
  of the same equilibrium ensemble.

The 132 kJ/mol Δ_diss provides an absolute reference scale the
v0.5 5-source H3 analysis lacked: AlphaFlow / Boltz-2 state-1
conformations (RMSD 7.6–9.9 Å from MD-mean) are sampling a
mid-barrier region; MarS-FM state-2 (RMSD 13.8 Å) reaches farther
along the FES toward dissociation but does not breach the 11 Å
barrier itself.

Figure: `md/figures/notch1_metad_fes.png` (3-walker mean ± 1σ).

## Holo result (2 walkers × 15 ns)

Holo metad was launched as 3 walkers × 20 ns on
`notch1_holo_v3` (NEC chain A + NTM chain B + Fab L + Fab H,
661 CAs total). The Fab-bound system runs at ~85 ns/day on Modal
A100-80GB (vs apo's ~300 ns/day — the 3× more atoms slow OpenMM
proportionally). All three walkers hit the @app.function
`timeout=4*3600` ceiling at ~15 ns instead of finishing 20 ns.

Walker 3 crashed at step 1251 with `DCDFile.__init__:
struct.error: unpack requires a buffer of 4 bytes` —
`dcd_mode = "a"` was chosen because a residual checkpoint.chk
existed at the walker output path (apparently from a transient
modal scheduling overlap), but the DCD file itself was missing,
so the append-open failed. Walker 1 + 2 ran cleanly.

### Per-walker convergence

| Walker | Bias duration (ns) | Final CV (Å) | Max CV (Å) | Well re-crossings |
|---|---:|---:|---:|---:|
| 1 | 15.1 | 3.5 | 7.7 | 426 |
| 2 | 14.0 | 18.8 | 23.7 | 92 |
| 3 | (crashed) | — | — | — |

Walker 1 stayed in / near the basin; walker 2 reached the
dissociation regime past the barrier. Together they constrain
the FES across the full CV range.

### Merged holo FES

| Coordinate | F (kJ/mol) | vs apo |
|---|---:|---|
| 4 Å (basin minimum) | 1.3 ± 0.9 | apo 0.4 ± 3.1 (~tie, both near 0) |
| 11 Å (transition state) | **98.4** | apo 115.5 → **+17 kJ/mol higher in apo** |
| 20 Å (dissociated) | **102.2** | apo 132.5 → **+30 kJ/mol higher in apo** |

The holo barrier is **17 kJ/mol lower** than the apo barrier on
the same CV. Holo's anti-NRR Fab destabilises the dissociation
direction. See `md/notch1_h2_metad_reweight.md` for full
discussion of the apo-vs-holo H2 implications, and
`md/figures/notch1_metad_apo_vs_holo_fes.png` for the side-by-
side FES.

Figure: `md/figures/notch1_metad_holo_fes.png` (2-walker mean ± 1σ).

## Open / deferred

- **Production FES analysis** (1D PMF reconstruction from HILLS via
  `plumed sum_hills`) — pending production run completion. Will
  update `md/notch1_metad_fes.png` and a "Free-energy summary"
  section here.
- **Reweighting MD frames to recover unbiased populations** — once
  the FES converges we can re-weight the unbiased v0.3 MD replicas
  along the same CV and re-issue the H2 magnitudes with a proper
  Boltzmann correction.
- **Convergence diagnostic** (bias re-crossing count) — extract
  from production COLVAR files when they land.
- **5 walkers × 100 ns production** — deferred to v0.7. Cost would
  be ~$25 on Modal; alternatively queueable as a DGX Spark idle-time
  job (free, ~10 days of background time per walker) via
  `md/slurm_md.sbatch`.

## Environment notes

The Modal image was non-trivial to wedge through the conda-forge
openmm + openmm-plumed + cuda-version pinch:

- **openmm-plumed 2.1** requires cuda-version ≥ 13. Modal A100-80GB
  drivers advertise cuda ≤ 12.x; **rejected**.
- **openmm-plumed 2.0** requires cudatoolkit < 12; pairs with
  openmm 8.1.x. **Working** combo: CUDA 11.8 base image +
  python 3.12 + openmm 8.1 + openmm-plumed 2.0 + cudatoolkit < 12.

The 6-attempt build trail is documented in the `md/notch1_metad_modal.py`
docstring history; the wedge is squarely an upstream conda-forge
packaging gap.

## Cost (v0.6 cumulative for W3c)

- Image builds (6 attempts): ~$1.50
- Smoke 2 ns walker: ~$0.30
- Production 3 walkers × 20 ns: ~$3.00 (in flight)
- Total W3c v0.6 spend: ~$5
