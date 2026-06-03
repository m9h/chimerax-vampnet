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

## Production (in flight)

Three walkers × 20 ns of biased dynamics on `notch1_apo_v3` were
launched in parallel on Modal A100-80GB:

- Wall clock: ~2 hours per walker (parallel)
- Cost: ~$3 total
- Coverage: 60 ns of cumulative biased sampling
- Walker indices: 1, 2, 3 (smoke was walker 0)

Each walker writes HILLS, COLVAR, traj.dcd to
`/vol/prepared/notch1_apo_v3/metad_walker_<i>/`. Post-processing
to a FES (1D PMF along NEC–NTM COM) lands in v0.7 alongside the
DGX Spark long-MD harvest.

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
