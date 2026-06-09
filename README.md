# 🧬 chimerax-vampnet

[![version](https://img.shields.io/badge/version-v0.10-1f6feb)](CITATION.cff)
[![docs](https://img.shields.io/badge/docs-live-2ea043)](http://center17.org/chimerax-vampnet/)
[![license](https://img.shields.io/badge/license-MIT-3fb950)](LICENSE)
[![UCSF ChimeraX](https://img.shields.io/badge/UCSF-ChimeraX-f6862d)](https://www.cgl.ucsf.edu/chimerax/)
[![python](https://img.shields.io/badge/python-3.9%E2%80%933.12-3776ab)](pyproject.toml)
[![tests](https://img.shields.io/badge/tests-49%20passing-2ea043)](tests/)
[![frontier models](https://img.shields.io/badge/frontier%20models-11-8957e5)](md/)
[![runs on Modal](https://img.shields.io/badge/runs%20on-Modal-7c3aed)](https://modal.com/)
[![method](https://img.shields.io/badge/method-VAMPnet%20%2B%20MSM-08979c)](https://deeptime-ml.github.io/)
![built for](https://img.shields.io/badge/built%20for-HTGAA%20%C2%B7%20Biopunk%20Lab-db61a2)
[![sister project](https://img.shields.io/badge/sister%20project-chimerax--origami-58a6ff)](https://github.com/m9h/chimerax-origami)

**A teaching resource for learning protein structure and dynamics by
driving frontier biomolecular models — built for the Biopunk Lab
HTGAA ("How To Grow Almost Anything") course.**

This repository is two things at once:

1. A **ChimeraX bundle** that loads heterogeneous structural ensembles
   (classical MD, AlphaFlow, BioEmu, Boltz-2, …) on equal footing,
   trains a VAMPnet (Mardt et al. 2018) via
   [deeptime](https://deeptime-ml.github.io/), and surfaces metastable
   states + transition rates as ChimeraX models, animations, and
   structured CLI output.
2. A **catalog of working cloud recipes** for invoking ~11 frontier
   protein models — the kind announced in papers and seminars but
   rarely runnable without a week of dependency archaeology. Each one
   is a single `modal run` away.

The goal is not a new scientific discovery. The goal is to let a
student **reproduce frontier-research workflows end to end** — pick a
protein, generate conformational ensembles from a dozen different
models, and *see* its energy landscape — and in doing so learn how
structure becomes dynamics, and how today's models do (and don't)
capture that.

---

## 🔭 The big idea: structure → ensemble → landscape

A crystal structure is **one** snapshot. A protein's *function* lives
in the **ensemble** of shapes it visits — and in the **landscape** of
barriers between them. This repo teaches that progression hands-on:

```
   sequence / PDB
        │
        ▼   (frontier model OR classical MD — the catalog below)
   conformational ensemble        ← "what shapes does it visit?"
        │
        ▼   (vampnet load_ensemble + fit, in ChimeraX)
   metastable states + rates      ← "how is the landscape organized?"
        │
        ▼   (vampnet states / means / animate / network)
   colored 3D models you can scrub, animate, and compare
```

The payoff is comparison: load an MD trajectory **and** an AlphaFlow
ensemble **and** a BioEmu ensemble into the *same* VAMPnet, and you can
literally watch which models explore the real conformational
equilibrium and which collapse onto the crystal prior. That comparison
is the central teaching moment — and a live research question.

---

## 🚀 The frontier-model catalog (`md/*_modal.py`)

> **🔗 Seminar connection.** Many of these models were first presented
> in the **Starkly Speaking** seminar series (UMA, Timewarp, OM-TPS,
> Plainer-EDM, Prose, StABlE all carry a talk date in their adapter
> docstring). The catalog turns each of those talks into something a
> student can *re-run*, not just read about.

Each adapter builds **its own** [Modal](https://modal.com/) cloud image
(self-contained — dependency collisions stay isolated to the one tool
that needs them) and emits a `.npz` of Cα coordinates that
`vampnet load_ensemble` reads directly. Every recipe encodes hard-won,
student-invisible knowledge: gated-HuggingFace access, CUDA/conda ABI
pinning, multi-GPU fanout, stdlib-only PDB parsing for stripped-down
cluster Pythons.

> **Status legend** — ✅ *verified* end-to-end on a real system in this
> project; 🧪 *scaffold* with first-invocation recipe drafted but not
> yet run to completion (honestly marked — finishing these is itself a
> great student exercise).

### Conformational-ensemble generators

| Model | What it gives you | Source / checkpoint | Status |
|---|---|---|---|
| **AlphaFlow / ESMFlow** | Flow-matching ensembles from sequence (Jing et al.) | `bjing-mit/alphaflow` | ✅ |
| **BioEmu** (v1.3.1) | Emulated equilibrium ensemble (Microsoft Research) | `microsoft/bioemu` | ✅ |
| **Boltz-2** | AF3-class diffusion structures, incl. membrane proteins | `boltz-community` | ✅ |
| **MarS-FM** | Flow-matching in MSM state space (Kapusniak et al.) | `valencelabs/mars-fm` (MD-CATH 450) | ✅ |
| **ESMFold2** | Multi-chain folded ensembles w/ per-seed sampling (Biohub) | `biohub/ESMFold2` | ✅ |
| **Prose** | Transferable all-atom normalizing flow for peptides | `transferable-samplers/prose-280M` | 🧪 |

### Force fields & physics-informed methods

| Model | What it gives you | Source / checkpoint | Status |
|---|---|---|---|
| **UMA** (Meta FAIR) | Universal ML force field — drives *real MD*, not just snapshots | `facebook/UMA` (`uma-s-1p2`) | ✅ |
| **Timewarp** | Normalizing-flow MCMC proposal; *trajectory-aware* (rates!) | `microsoft/timewarp` | ✅ MH on dipeptides¹ |
| **Plainer EDM (ScoreMD)** | Fokker-Planck-consistent diffusion: sampler *and* integrator | `noegroup/ScoreMD` | 🧪 |
| **OM-TPS** | Zero-shot transition-path sampling from any score model | `ASK-Berkeley/OM-TPS` | 🧪 |
| **StABlE** | Stability-aware (physics-informed) NNIP *training* objective | `ASK-Berkeley/StABlE-Training` | 🧪 |

### Classical baseline & enhanced sampling

| Tool | What it gives you | Adapter |
|---|---|---|
| **OpenMM MD** | Ground-truth trajectories; soluble *and* POPC-membrane prep | `md/modal_md.py`, `md/prep.py`, `md/prep_membrane.py` |
| **Metadynamics (PLUMED)** | 1D / 2D free-energy surfaces along chosen CVs | `md/notch1_metad_modal.py` |

¹ Timewarp's Metropolis-Hastings loop runs end-to-end on alanine
dipeptide; its 0-acceptance edge case (perfect state-dict load, but a
downstream coordinate/topology mismatch) is documented in-repo as a
worked example of the friction real frontier tools carry. An
importance-sampling "rescue" path and arviz MCMC diagnostics ship
alongside it (`md/mcmc_diagnostics.py`).

**Typical invocation** (every adapter exposes a Modal `local_entrypoint`):

```bash
# Generate a 200-frame BioEmu ensemble for a sequence, save Cα npz:
modal run md/bioemu_modal.py --sequence MKT...QHL --name my_protein --out my_protein_bioemu200.npz

# Then, inside ChimeraX:
vampnet load_ensemble bioemu my_protein_bioemu200.npz format bioemu
```

Gated models (UMA, some HF checkpoints) reuse one Modal
`huggingface-secret`; you accept the model's terms once on its HF page.
See each adapter's module docstring for the exact image recipe, the
checkpoint provenance, and the date it was last verified.

---

## 🔬 The analysis layer: ChimeraX VAMPnet bundle

8 modules, 1375 LOC, 49 tests. Once you have ensembles, the bundle
turns them into a landscape you can see and steer:

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
(Claude Desktop, Cursor, …) can drive an **adaptive analysis loop** via
the included HTTP bridge — the same human/LLM-in-the-loop pattern the
sister project (below) uses for design.

---

## 🎓 A worked student path

1. **Pick a target.** Start small — alanine dipeptide or chignolin —
   then graduate to a real receptor.
2. **Generate ensembles three ways:** classical MD (`md/modal_md.py`),
   one flow model (AlphaFlow), one emulator (BioEmu).
3. **Load all three into one VAMPnet** and `fit n_states 4`.
4. **Compare.** Use `vampnet states` and `vampnet network`: does the
   generative model populate the same basins MD does, or collapse to
   one? (Spoiler from this project's runs: on Hsp90 NTD the generative
   models collapse to the crystal prior while MD discovers a *closed-lid
   cryptic state* — a real, re-runnable finding.)
5. **Perturb and re-watch.** Add an antibody or ligand context and see
   the stationary populations shift.

### Validation systems already wired

| System | Why it's interesting | What's in the repo |
|---|---|---|
| **Alanine dipeptide** | Canonical 5-basin Ramachandran toy | MD + Timewarp ensembles |
| **Chignolin (CLN025)** | Fast-folder w/ DESRES Anton gold standard | 1 µs self-generated MD, folded/unfolded split |
| **Notch1 NRR** | Membrane-receptor conformational *switch* (apo vs antibody-bound) | MD + 5 generative sources, multi-chain ESMFold2, metadynamics |
| **Hsp90α NTD** | Chaperone w/ a cryptic drug pocket | MD + 5 sources; cryptic-state discovery |
| **β2-adrenergic receptor** | Gold-standard Class-A GPCR, drug target | 5 generative ensembles + GPCR-specific structural analysis |

---

## 🔗 Sister project: `chimerax-origami` — same idea, in DNA

This bundle has a deliberate mirror image,
[`chimerax-origami`](../dna-origami), built for the **DNA-origami**
half of the same HTGAA curriculum. The two share **one abstraction —
the contact map — and one thesis: folding/assembly reliability is the
minimization of landscape *frustration*.**

| | chimerax-vampnet | chimerax-origami |
|---|---|---|
| Object | protein conformational landscape | DNA-origami assembly landscape |
| Input | MD / AlphaFlow / BioEmu ensembles | cadnano / scadnano / oxDNA designs |
| Shared substrate | Cα–Cα **contact map** | base-pair **contact map** |
| "Frustration" | metastable kinetic traps | off-target hybridization |
| Inverse design | adaptive sampling | Pareto scaffold selection |
| Perturbation | antibody shifts state populations | lipid envelope shifts in-vivo fate |
| Driver | MCP LLM agent in the loop | MCP LLM agent in the loop |

The connection is concrete, not just analogy: the Notch1 NRR is a
*membrane* receptor, and the origami side wraps nanostructures in a
*lipid envelope* (Perrault & Shih, virus-inspired encapsulation) — both
projects independently land on the **membrane interface** as the
boundary condition, and a membrane-binding event as the control knob
that re-weights an ensemble. Stack them and you get a closed
design–build–test–learn loop: origami builds the enveloped chassis,
VAMPnet characterizes the protein cargo's dynamics, and an MCP agent
drives both. See [`dna-origami/CONNECTIONS.md`](../dna-origami/CONNECTIONS.md)
for the full treatment.

For students, the lesson is transferable: **structure → contact map →
landscape → de-frustration** is the same workflow whether you're
folding a protein or assembling a nanostructure.

---

## ⚙️ Install (development)

```bash
# From within ChimeraX:
toolshed install --reinstall /path/to/chimerax-vampnet

# Or build the wheel from the command line:
chimerax --nogui --exit --cmd "devel build /path/to/chimerax-vampnet"
```

The bundle's only external runtime dependency is `deeptime>=0.4`.
PyTorch ships with ChimeraX's AlphaFold bundle, so we don't redeclare
it. The `md/` adapters need a (free-to-start) [Modal](https://modal.com/)
account; nothing in `md/` is required to use the ChimeraX bundle on
ensembles you already have.

### Quickstart: chignolin tutorial

```
open /data/datasets/chimerax-vampnet/chignolin_modal/chignolin/equilibrated.pdb
vampnet load_ensemble md /data/datasets/chimerax-vampnet/chignolin_modal/chignolin/replica_0/traj.dcd source md
vampnet fit nStates 2 lag 100 features ca_distances epochs 80
vampnet states                # colors structure folded vs unfolded
vampnet means                 # builds 2 mean structures
vampnet animate mode 1 nFrames 100
vampnet network               # transition rates
```

Or run the walkthrough directly: `open examples/chignolin_tutorial.cxc`

### Test stack

```bash
python -m venv .venv && .venv/bin/pip install torch deeptime pytest
.venv/bin/python -m pytest tests/          # 49 tests, no live ChimeraX needed
```

ChimeraX integration is exercised through the `.cxc` tutorials; all
unit tests run against mock fixtures.

---

## ⚠️ What works, and what's still rough (read this)

Frontier tooling is *frontier* — honest friction is part of the
lesson. Current state:

- **Verified end-to-end** (✅ above): AlphaFlow, BioEmu, Boltz-2,
  MarS-FM, ESMFold2, UMA, and OpenMM MD all produce real ensembles on
  real systems in this repo.
- **Partially working:** Timewarp samples but hits a 0-acceptance edge
  case (documented as a worked debugging example, with an
  importance-sampling rescue and MCMC diagnostics).
- **Scaffolded (🧪):** Prose, Plainer-EDM/ScoreMD, OM-TPS, and StABlE
  have drafted image recipes and first-invocation TODOs but haven't
  been run to completion — finishing one is a self-contained project.
- **Membrane MD** (β2AR) prep exists; the production run hit a
  numerical instability at NVT equilibration and is deferred — a
  realistic example of where membrane systems bite.

When something fails, the module docstring records *why* and what was
tried. That trail is deliberately preserved as teaching material.

---

## 📄 License & citation

MIT — see `LICENSE`. If you use this bundle, please cite the underlying
method:

- Mardt, A., Pasquali, L., Wu, H. & Noé, F. *VAMPnets for deep learning
  of molecular kinetics.* Nat. Commun. **9**, 5 (2018).
- Hoffmann, M. et al. *Deeptime: a Python library for machine learning
  dynamical models from time series data.* Mach. Learn. Sci. Technol.
  **3**, 015009 (2021).

Each `md/*_modal.py` adapter's docstring cites the specific frontier
model it wraps (paper, arXiv, code, checkpoint) — start there to credit
the model authors.
