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

## Status — v0.1 pre-alpha

**1184 LOC across 8 modules, 4/4 tests green.**

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
- **Tier-2 (Notch1 NRR apo + holo)**: ran on Modal H100 (~$65 spent). Apo pipeline ran end-to-end. Holo had a chain-selection bug (filtered wrong chains from 3L95) — re-prep needed for a real apo/holo comparison. Bundle's *analysis* pipeline is uninvolved in the bug; it's an MD-prep issue.

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

**v0.2:**
- Residue-number-based chain identification (survives PDBFixer renumbering)
- Positional restraints for membrane-anchored proteins (e.g., Notch1 NRR)
- MCP-stdio proxy script for Claude Desktop integration
- Implied-timescales convergence test integrated into `vampnet timescales`
- Direct AlphaFlow + BioEmu ensemble fetch from HuggingFace

**v0.3:**
- Multi-source joint VAMPNet (MD + AlphaFlow + BioEmu fused embedding)
- Gibbs energy landscape rendering in ChimeraX
- Adaptive sampling loop via MCP (LLM agent selects which states to extend MD on)

## License

MIT — see `LICENSE`.

## Citation

If you use this bundle, please cite both the underlying VAMPnet
method and (if applicable) deeptime:

- Mardt, A., Pasquali, L., Wu, H. & Noé, F. *VAMPnets for deep learning of molecular kinetics.* Nat. Commun. **9**, 5 (2018).
- Hoffmann, M. et al. *Deeptime: a Python library for machine learning dynamical models from time series data.* Mach. Learn. Sci. Technol. **3**, 015009 (2021).
