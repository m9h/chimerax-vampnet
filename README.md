# chimerax-vampnet

VAMPnets + Markov state modeling of protein conformational landscapes,
integrated into UCSF ChimeraX. Loads heterogeneous ensembles (MD,
AlphaFlow, BioEmu) on equal footing, trains a VAMPnet (Mardt et al.
2018) via [deeptime](https://deeptime-ml.github.io/), and surfaces
metastable states + transition rates as ChimeraX models, animations,
and structured CLI output.

Every command returns JSON-serializable data so an MCP-capable LLM
agent (Claude Desktop, Cursor, etc.) can drive an adaptive analysis
loop via the included HTTP bridge.

## Status — v0.3

**1184 LOC across 8 modules, 18/18 tests green; v0.3 adds COM-distance restraint for NRR membrane anchor.**

| Module | Lines | Status |
|---|---:|---|
| `src/cmd.py` | 234 | ChimeraX command registration (11 commands) |
| `src/featurize.py` | 226 | MD / AlphaFlow / BioEmu loaders + CA-distance + backbone-torsion features |
| `src/viz.py` | 204 | `color_by_state` (live recolor on coordset change) + `build_state_means` |
| `src/mcp_server.py` | 195 | HTTP/JSON bridge for LLM agents |
| `src/vampnet_core.py` | 151 | `fit` (deeptime VAMPNet + MLP lobe) + `save`/`load` |
| `src/animate.py` | 88 | Slow-mode animation between extreme metastable states |
| `src/msm.py` | 64 | MSM transition graph (nodes + edges) |
| `src/__init__.py` | 22 | BundleAPI subclass |

### Validation

- **Day-2 smoke test** (`tests/test_random_walk.py`): synthetic 4-state Markov chain → bundle featurize → deeptime VAMPNet wiring. ✅ Passes.
- **Tier-1 (chignolin CLN025)**: self-generated 1 µs trajectory at 340 K on Modal H100, 4 fs HMR, $37. Bundle recovers:
  - Slowest implied timescale **166 ns** (published reference: 100-500 ns)
  - Clean folded vs unfolded state separation (Trp9-Tyr1 5.7 vs 15.6 Å)
- **Tier-2 (Notch1 NRR apo + holo)**: apo + corrected-chain holo
  trajectories under two restraint protocols. v0.2 unrestrained MD
  (3×100 ns each, Modal A100-80GB): directional H2 **met** (+3.7 pp,
  apo more auto-inhibited), magnitudes not met (apo 21.6% vs target
  ≥50%, holo 17.9% vs target ≤30%) due to NRR-fragment dissociation
  to >100 Å. v0.3 NEC-NTM COM-distance restraint (3×100 ns each):
  same directional result (+3.8 pp), restraint cleanly eliminates
  dissociation (COM sep std 24 Å → 0.3 Å), but magnitudes **still**
  not met — isolates the 100 ns sampling horizon (not the restraint)
  as the magnitude blocker. Full diagnostics:
  `md/notch1_h2_results.md` (v0.2), `md/notch1_h2_v3_results.md` (v0.3).
- **Wider robustness (ATLAS sweep)**: 4 public ATLAS trajectories
  spanning all-α / mostly-β / mixed / large-α folds and 73-518
  residues, all converge to non-degenerate 4-state decompositions
  with 38-75 ns slow timescales. See `md/wider_atlas_results.md`.
- **LLM-agent adaptive sampling demo**: `examples/adaptive_sampling_demo.py`
  on the chignolin trajectory grows the slowest implied timescale
  from 94 → 201 ns (2.1× improvement) via rare-state-targeted
  re-sampling, exercising the MCP-stdio proxy end-to-end.

## Commands

```
vampnet load_ensemble  source  path  [format auto|alphaflow|bioemu|md]
vampnet fit             [n_states 4]  [lag 10]  [features ca_distances|torsions]  [epochs 200]
vampnet timescales      [taus 1,2,5,10,20,50,100]
vampnet states                       # color frames by state, live updates as you scrub
vampnet means                        # build per-state mean-structure models
vampnet animate         [mode 1] [n_frames 100]
vampnet network                      # transition matrix as a graph (nodes + edges)
vampnet save            path
vampnet load            path
vampnet mcp serve       [port 7345]  # expose bundle to MCP-capable LLM agents
vampnet mcp stop
```

Each command returns a JSON-serializable dict (also visible in the
ChimeraX log) that the MCP bridge proxies as a tool result.

## Install (development)

```bash
# From within ChimeraX:
toolshed install --reinstall /path/to/chimerax-vampnet

# Or build the wheel from the command line:
chimerax --nogui --exit --cmd "devel build /path/to/chimerax-vampnet"
```

The bundle's only external runtime dependency is `deeptime>=0.4`.
PyTorch is already shipped with ChimeraX's AlphaFold bundle so we
don't redeclare it.

## Quickstart: chignolin tutorial

The shipped trajectory of CLN025 at 340 K demonstrates the full
analysis pipeline. From a ChimeraX session:

```
open chignolin/equilibrated.pdb
vampnet load_ensemble md chignolin/replica_0/traj.dcd source md
vampnet fit nStates 2 lag 100 features ca_distances epochs 80
vampnet states                # colors structure folded vs unfolded
vampnet means                 # builds 2 mean structures
vampnet animate mode 1 nFrames 100
vampnet network               # transition rates
```

Or run the .cxc walkthrough directly: `open examples/chignolin_tutorial.cxc`

## HTTP / MCP bridge

`vampnet mcp serve` starts a stdlib HTTP server on the chosen port
(default 7345). Endpoints:

```
GET  /health                   # {"status": "ok", "model_loaded": bool}
GET  /tools                    # full MCP-compatible tool manifest
POST /tools/vampnet_fit        # body: {"n_states": 4, "lag": 10, ...}
POST /tools/vampnet_states     # ...
POST /tools/vampnet_animate    # ...
# etc
```

This is the substrate. A future v0.2 ships a stdio MCP proxy script so
Claude Desktop / Cursor / Continue can speak MCP natively.

## Test stack

```bash
python -m venv .venv && .venv/bin/pip install torch deeptime pytest
.venv/bin/python -m pytest tests/
```

All tests run **without** a live ChimeraX session via mock fixtures.
ChimeraX integration is exercised through the .cxc tutorial.

## MD pipeline (`md/`)

The bundle ships with a self-contained MD pipeline for generating
demo trajectories. Two backends:

- **Local GB10**: `md/Dockerfile` + `md/run_md.sh` builds an OpenMM
  CUDA aarch64 container (conda-forge `openmm=8.5.1`, 4 fs HMR,
  CUDA 12.9). 2950 ns/day on alanine dipeptide, 270 ns/day on
  chignolin.
- **Modal cloud**: `md/modal_md.py` provides `prep` + `fanout`
  entrypoints with H100 parallel replica spawn (`.spawn()` fire-
  and-forget). 3050 ns/day on chignolin, ~950 ns/day on Notch1 NRR
  apo, ~360 ns/day on Notch1 NRR holo at 4 fs.

See `md/README.md` for the GB10 workflow and `md/modal_md.py` for the
Modal cloud commands.

## Roadmap

**v0.2 (shipped):**
- ✅ MCP-stdio proxy for Claude Desktop / Cursor / Continue
- ✅ Implied-timescales convergence test in `vampnet timescales`
- ✅ Corrected-chain Notch1 holo prep + 3×100 ns A100-80GB MD
- ✅ H2 directional verdict (+3.7 pp, magnitudes pending)
- ✅ Wider ATLAS robustness sweep (4 proteins, 4 folds)
- ✅ LLM-agent adaptive-sampling demonstration (2.1× slow IT growth)

**v0.3 (shipped):**
- ✅ NEC-NTM COM-distance restraint (`CustomCentroidBondForce`,
  replaces per-atom anchor — see `md/notch1_h2_v3_results.md` and
  `md/prep.py:_add_com_distance_restraint`). Per-atom anchors NaN
  at HMR=4 fs at any useful k; COM-distance is invariant to whole-
  system drift and has no per-atom force singularities.
- ✅ Apo + holo MD under the COM restraint (3×100 ns each); H2
  directional replicates (+3.8 pp); restraint eliminates NRR
  dissociation; sampling horizon (not restraint) isolated as
  magnitude blocker.
- ✅ `vampnet load_ensemble … source marsfm` loader stub ready
  for MarS-FM (arXiv:2509.24779) checkpoint when released.

**v0.4:**
- MarS-FM ensemble integration on Notch1 NRR apo + holo (once the
  official checkpoint drops) — direct test of the magnitude question.
  Speedup ~600× vs explicit MD; per-replica cost ~$0.05 vs ~$70.
- Bootstrap uncertainty on H2 populations (deeptime MSM bootstrap).
- Real AlphaFlow + BioEmu inference on Modal — `md/alphaflow_modal.py`
  is the draft Modal app.
- Multi-source joint VAMPNet with per-source covariance weighting +
  stratified state-coverage report (test H3).
- Live MCP-driven adaptive sampling on Notch1: wire the demo loop to
  actual Modal MD launches seeded from `vampnet means` outputs.

**Stretch (v0.5+):**
- Gibbs energy landscape rendering inside ChimeraX
- MACE-OFF neural-network potentials in the MD backend
- DiffDock pose-stability teaching demo
- Umbrella sampling on NEC-NTM distance for a μs-equivalent PMF

## License

MIT — see `LICENSE`.

## Citation

If you use this bundle, please cite both the underlying VAMPnet
method and (if applicable) deeptime:

- Mardt, A., Pasquali, L., Wu, H. & Noé, F. *VAMPnets for deep learning of molecular kinetics.* Nat. Commun. **9**, 5 (2018).
- Hoffmann, M. et al. *Deeptime: a Python library for machine learning dynamical models from time series data.* Mach. Learn. Sci. Technol. **3**, 015009 (2021).
