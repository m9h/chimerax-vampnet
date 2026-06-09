# 🔬 The analysis layer

Once you have ensembles, the ChimeraX bundle turns them into a landscape you can
see and steer. **8 modules, 1375 LOC, 49 tests.**

## Commands

```
vampnet load_ensemble  source  path  [format auto|alphaflow|bioemu|md|marsfm]
vampnet fit            [n_states 4] [lag 10] [features ca_distances|torsions] [epochs 200]
vampnet timescales     [taus 1,2,5,10,20,50,100]   # implied-timescale convergence
vampnet states                       # color frames by state, live as you scrub
vampnet means                        # build per-state mean-structure models
vampnet animate        [mode 1] [n_frames 100]      # slow-mode morph between extremes
vampnet network                      # transition matrix as a graph
vampnet save / load    path
vampnet mcp serve      [port 7345]   # expose the bundle to an MCP LLM agent
```

Every command returns a JSON-serializable dict, so an MCP-capable agent
(Claude Desktop, Cursor, …) can drive an **adaptive analysis loop** via the
included HTTP bridge.

## What the modules do

| Module | Role |
|---|---|
| `src/featurize.py` | MD / AlphaFlow / BioEmu loaders + Cα-distance & backbone-torsion features |
| `src/vampnet_core.py` | `fit` (deeptime VAMPNet + MLP lobe) + `save`/`load` |
| `src/viz.py` | `color_by_state` (live recolor on coordset change) + `build_state_means` |
| `src/msm.py` | MSM transition graph (nodes + edges) |
| `src/animate.py` | slow-mode animation between extreme metastable states |
| `src/mcp_server.py` | HTTP/JSON bridge for LLM agents |
| `src/cmd.py` | ChimeraX command registration |

## The `md/` directory

`md/` is the data-and-analysis layer that *feeds* the bundle. It is mapped in
detail in its own [`md/README.md`](https://github.com/m9h/chimerax-vampnet/blob/main/md/README.md),
grouped into five roles:

1. **Frontier-model adapters** (`*_modal.py`) — the [catalog](catalog.md).
2. **Classical MD pipeline** — `modal_md.py`, `prep.py`, `prep_membrane.py`,
   `produce.py`, `notch1_metad_modal.py`.
3. **Reusable analysis libraries** — `multisource_h3.py` (joint VAMPnet over N
   sources), `multichain.py`, `mcmc_diagnostics.py` (arviz ESS / R-hat),
   `diagnostics_sweep.py`.
4. **Per-system analysis scripts** — Notch1 / Hsp90 / β2AR feature extractors.
5. **Demos & validation tiers** — chignolin, alanine dipeptide, the ATLAS sweep.

## Install (development)

```bash
# From within ChimeraX:
toolshed install --reinstall /path/to/chimerax-vampnet

# Or build the wheel from the command line:
chimerax --nogui --exit --cmd "devel build /path/to/chimerax-vampnet"
```

The bundle's only external runtime dependency is `deeptime>=0.4`. PyTorch ships
with ChimeraX's AlphaFold bundle, so it isn't redeclared. The `md/` adapters
need a (free-to-start) [Modal](https://modal.com/) account; nothing in `md/` is
required to use the ChimeraX bundle on ensembles you already have.

```bash
# Test stack — 49 tests, no live ChimeraX needed:
python -m venv .venv && .venv/bin/pip install torch deeptime pytest
.venv/bin/python -m pytest tests/
```
