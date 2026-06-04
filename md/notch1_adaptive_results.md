# Live MCP-driven adaptive sampling on Notch1 NRR — v0.6 W3d result

**Date**: 2026-06-03
**Status**: Phase 1 smoke (1 replica × 2 ns) PoC PASSED end-to-end.

Script: `examples/live_adaptive_sampling_notch1.py`
Result JSON: `md/notch1_adaptive_v0p6_results.json`

## What this PoC demonstrates

The v0.4 MCP adaptive-sampling loop demonstrated agent-driven
short-MD bursts on the generic chignolin case. v0.6 W3d applies
that loop to the actual Notch1 NRR system. The loop closure being
demonstrated:

1. **Agent (script)** posts to `vampnet_fit` on the v0.5 5-source
   ensemble; recovers baseline state populations (s0=46%, s1=18%,
   s2=18%, s3=19%).
2. **Agent** identifies states 1 and 2 as MD-inaccessible (per the
   v0.5 H3 finding) and decides to extend sampling.
3. **Agent** posts to `modal_md::fanout` to launch a fresh MD
   replica from the v0.3 apo equilibrated state.
4. **Agent** polls the Modal volume for the new replica's
   `traj.dcd`.
5. **Agent** pulls the DCD, featurizes the 174 NEC Cα atoms, and
   projects them onto the v0.5 VAMPnet to get per-state occupancy.

Phase 1 smoke parameters: 1 replica × 2 ns at replica index 10
(to avoid collision with the existing v0.3 0/1/2 replicas).

## Result

| Replica | n_frames | s0 | s1 | s2 | s3 |
|---|---:|---:|---:|---:|---:|
| 10 (Phase 1) | 100 | **100 %** | 0 % | 0 % | 0 % |
| (baseline v0.5 ensemble) | 2 269 | 46 % | 18 % | 18 % | 19 % |

The new 2 ns replica spent **100 % of its frames in state 0** (the
tightest MD-equilibrium sub-basin). No frames in state 1 or 2
(the structure-prediction-only and flow-matching-only basins).

## Interpretation

This is the expected result given v0.5 + v0.6 W3c:

- v0.5 H3 already established that states 1 and 2 are
  generative-only at 100 ns of MD per replica (zero MD frames in
  either state across 1 500 frames).
- v0.6 W3c metad measured the dissociation barrier separating the
  MD-equilibrium states from the generative basins at 115.5 kJ/mol
  → MFP ~10¹⁰ years per ns attempt → zero chance of any 2 ns
  burst crossing.

So the negative result ("Phase 1 random-seed MD does NOT reach
state 1 or 2 in 2 ns") is a direct confirmation of v0.6 W3c. The
v0.6 PoC value is in **demonstrating the loop closure works
end-to-end** on the real target system, not in producing new
science.

## What this enables for v0.7

Phase 2 of the adaptive-sampling loop would seed MD bursts from
the AlphaFlow state-1 mean structure (instead of from the apo
equilibrated state). The test then becomes: "if we *start* MD in
state 1, does it stay there or drain back to state 0/3 within
20 ns?" The answer connects to whether state 1 is a real
metastable basin or a transient generative-model artifact.

Phase 2 requires building a hybrid AlphaFlow-state-1-NEC +
apo-NTM all-atom PDB and re-equilibrating, which v0.6 deferred.
v0.7 should implement it via a Cα-morph + side-chain
re-minimization pipeline.

## Reproducibility

```sh
.venv/bin/python examples/live_adaptive_sampling_notch1.py \\
    --replicas 1 --ns 2 --start-replica 10
```

Cost: ~$0.20 in Modal A100 time. Wall clock: ~10 min (build
cache hit) + 30 min sim + 2 min pull + 30 s VAMPnet score = ~45
minutes total.

## Topology gotcha (v0.6 fix)

The Phase 1 smoke initially failed at the `mdtraj.load` step
because the script defaulted to the pre-prep
`notch1_modal/notch1_apo/equilibrated.pdb` as the topology, whose
atom count does not match the v0.3-prepped DCDs (different
solvation box). Fix: pull `prepared/notch1_apo_v3/equilibrated.pdb`
from the Modal volume on first use and cache it locally at
`/tmp/notch1_v3_topo/equilibrated_v3.pdb`. The script now does
this automatically.
