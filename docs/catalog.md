# 🚀 The frontier-model catalog

Each adapter lives in `md/*_modal.py`, builds its **own**
[Modal](https://modal.com/) cloud image (self-contained — dependency collisions
stay isolated to the one tool that needs them), and emits a `.npz` of Cα
coordinates that `vampnet load_ensemble` reads directly. Every recipe encodes
hard-won, student-invisible knowledge: gated-HuggingFace access, CUDA/conda ABI
pinning, multi-GPU fanout, stdlib-only PDB parsing for stripped-down cluster
Pythons.

!!! tip "Seminar connection"
    Many of these models were first presented in the **Starkly Speaking**
    seminar series (UMA, Timewarp, OM-TPS, Plainer-EDM, Prose, StABlE all carry
    a talk date in their adapter docstring). The catalog turns each of those
    talks into something a student can *re-run*, not just read about.

**Status legend** — :white_check_mark: *verified* end-to-end on a real system ·
:test_tube: *scaffold* with a first-invocation recipe drafted but not yet run to
completion (finishing one is a great student exercise).

## Conformational-ensemble generators

| Model | What it gives you | Source / checkpoint | Status |
|---|---|---|:--:|
| **AlphaFlow / ESMFlow** | Flow-matching ensembles from sequence (Jing et al.) | `bjing-mit/alphaflow` | :white_check_mark: |
| **BioEmu** (v1.3.1) | Emulated equilibrium ensemble (Microsoft Research) | `microsoft/bioemu` | :white_check_mark: |
| **Boltz-2** | AF3-class diffusion structures, incl. membrane proteins | `boltz-community` | :white_check_mark: |
| **MarS-FM** | Flow-matching in MSM state space (Kapusniak et al.) | `valencelabs/mars-fm` (MD-CATH 450) | :white_check_mark: |
| **ESMFold2** | Multi-chain folded ensembles w/ per-seed sampling (Biohub) | `biohub/ESMFold2` | :white_check_mark: |
| **Prose** | Transferable all-atom normalizing flow for peptides | `transferable-samplers/prose-280M` | :test_tube: |

## Force fields & physics-informed methods

| Model | What it gives you | Source / checkpoint | Status |
|---|---|---|:--:|
| **UMA** (Meta FAIR) | Universal ML force field — drives *real MD*, not just snapshots | `facebook/UMA` (`uma-s-1p2`) | :white_check_mark: |
| **Timewarp** | Normalizing-flow MCMC proposal; *trajectory-aware* (rates!) | `microsoft/timewarp` | :white_check_mark:[^tw] |
| **Plainer EDM (ScoreMD)** | Fokker-Planck-consistent diffusion: sampler *and* integrator | `noegroup/ScoreMD` | :test_tube: |
| **OM-TPS** | Zero-shot transition-path sampling from any score model | `ASK-Berkeley/OM-TPS` | :test_tube: |
| **StABlE** | Stability-aware (physics-informed) NNIP *training* objective | `ASK-Berkeley/StABlE-Training` | :test_tube: |

## Classical baseline & enhanced sampling

| Tool | What it gives you | Adapter |
|---|---|---|
| **OpenMM MD** | Ground-truth trajectories; soluble *and* POPC-membrane prep | `md/modal_md.py`, `md/prep.py`, `md/prep_membrane.py` |
| **Metadynamics (PLUMED)** | 1D / 2D free-energy surfaces along chosen CVs | `md/notch1_metad_modal.py` |

[^tw]:
    Timewarp's Metropolis-Hastings loop runs end-to-end on alanine dipeptide;
    its 0-acceptance edge case (perfect state-dict load, but a downstream
    coordinate/topology mismatch) is documented in-repo as a worked example of
    the friction real frontier tools carry. An importance-sampling "rescue" path
    and arviz MCMC diagnostics ship alongside it (`md/mcmc_diagnostics.py`).

## Running one

Every adapter exposes a Modal `local_entrypoint`:

```bash
# Generate a 200-frame BioEmu ensemble for a sequence, save a Cα npz:
modal run md/bioemu_modal.py --sequence MKT...QHL --name my_protein --out my_protein_bioemu200.npz

# Then, inside ChimeraX:
vampnet load_ensemble bioemu my_protein_bioemu200.npz format bioemu
```

Gated models (UMA, some HF checkpoints) reuse one Modal `huggingface-secret`;
you accept the model's terms once on its HF page. See each adapter's module
docstring for the exact image recipe, the checkpoint provenance, and the date it
was last verified.

!!! warning "Frontier tooling is *frontier*"
    Honest friction is part of the lesson. Verified adapters produce real
    ensembles on real systems; scaffolds have drafted recipes and
    first-invocation TODOs. When something fails, the module docstring records
    *why* and what was tried — that trail is deliberately preserved as teaching
    material.
