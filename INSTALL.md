# Installing chimerax-vampnet

`chimerax-vampnet` is a [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/)
bundle. v0.6 is the first release intended for end users; earlier
versions were dev-only.

## Prerequisites

- **UCSF ChimeraX 1.10 or newer.** Daily build is fine.
  Download: <https://www.cgl.ucsf.edu/chimerax/download.html>
- **Python ≥ 3.11** if you also want to run the headless `tests/` and
  `examples/` scripts outside ChimeraX (the bundle ships its own
  Python through ChimeraX).
- ~2 GB of disk for the bundle + its torch/deeptime/numpy
  dependencies (ChimeraX downloads them on first install).

## Install path A — ChimeraX Toolshed (recommended)

*Coming with the v0.6 Toolshed submission.* Once approved, in
ChimeraX run:

```
toolshed install chimerax-vampnet
```

## Install path B — local development install (current)

This is the path that works today. Clone the repo and `toolshed
install` from a local path:

```sh
git clone https://github.com/m9h/chimerax-vampnet.git
cd chimerax-vampnet
```

Then inside ChimeraX:

```
toolshed install --reinstall /full/path/to/chimerax-vampnet
```

The bundle should now be available; verify with:

```
help vampnet fit
```

You should see the synopsis with parameter ranges and an example.
If `help vampnet fit` reports an unknown command, the bundle did not
register — re-open ChimeraX and re-run `toolshed install`.

## Platform notes

### macOS

ChimeraX runs natively on Apple Silicon and Intel; the bundle's
dependencies (torch, deeptime, numpy) are auto-installed by
ChimeraX. No system-level Python is required.

### Linux

Same as macOS. If you hit a torch wheel issue, ensure ChimeraX's
embedded Python is being used (`chimerax --python-version`).

### Windows

The bundle works in ChimeraX on Windows but the MCP HTTP bridge
binds to `127.0.0.1` and Windows Defender may prompt for permission
the first time `vampnet mcp serve` is run.

## 5-minute quickstart

After install, run the chignolin tutorial (the bundle ships it):

```
open chignolin_tutorial.cxc
```

Inside the repo, the file is at `examples/chignolin_tutorial.cxc`.
It walks through `vampnet load_ensemble → fit → states → means →
animate` on the 10-residue chignolin folding peptide, takes ~2
minutes on a modern laptop, and produces a 2-state folded ↔
unfolded VAMPnet you can inspect interactively.

## MCP bridge bootstrap

The bundle exposes its commands as an HTTP/JSON tool server so
MCP-capable LLM agents (Claude Desktop, Cursor, Continue, etc.) can
drive a ChimeraX session.

Inside ChimeraX:

```
vampnet mcp serve
```

By default this listens on `127.0.0.1:7345`. To point a desktop
MCP client at it, configure your client's MCP server entry with a
stdio proxy command — the bundle ships one at
`src/mcp_stdio_proxy.py`. Example for Claude Desktop's
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chimerax-vampnet": {
      "command": "python",
      "args": ["/full/path/to/chimerax-vampnet/src/mcp_stdio_proxy.py"]
    }
  }
}
```

Restart Claude Desktop and the `vampnet_*` tools should appear in
the tool list. Stop the bridge with `vampnet mcp stop` in ChimeraX.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `help vampnet fit` says unknown command | bundle didn't register | re-open ChimeraX, re-run `toolshed install` |
| `vampnet fit` errors with "no ensembles loaded" | nothing loaded yet | `vampnet load_ensemble md_apo /path/to/your.dcd` first |
| `vampnet states` errors with "no MD structure" | only generative ensembles loaded; no DCD | open the MD topology PDB first via `open` |
| `vampnet mcp serve` says port in use | another bridge running | `vampnet mcp stop` or pick a different port |
| Import error for `torch` / `deeptime` | ChimeraX's first install didn't pull deps | restart ChimeraX, re-run `toolshed install --reinstall` |

## Running the test suite

The Python tests (without ChimeraX) can run in a regular venv:

```sh
python -m venv .venv && .venv/bin/pip install torch deeptime pytest mdtraj numpy
.venv/bin/python -m pytest tests/ -v
```

The end-to-end ChimeraX integration test
(`tests/test_chimerax_integration.py`) requires the `chimerax`
binary on `PATH` and the chignolin demo data; it skips gracefully
if either is missing.

## Where to next

- `examples/chignolin_tutorial.cxc` — interactive 2-state folding demo
- `examples/mardt2018_reproduction.py` — alanine dipeptide + pentapeptide reproduction
- `examples/adaptive_sampling_demo.py` — MCP-agent adaptive sampling loop
- `examples/live_adaptive_sampling.py` — real Modal MD launches from agent decisions
- `md/notch1_h3_results.md` — the v0.5 5-source headline result
