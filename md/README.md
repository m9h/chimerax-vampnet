# `md/` — ensemble generation & landscape analysis

This directory is the **data-and-analysis layer** that feeds the
ChimeraX `vampnet` bundle. It does three jobs:

1. **Generate conformational ensembles** — from ~11 frontier models
   (the catalog in the [root README](../README.md#the-frontier-model-catalog))
   *and* from classical OpenMM MD.
2. **Run enhanced sampling** — metadynamics free-energy surfaces.
3. **Analyze landscapes** — multi-source joint VAMPnets, MCMC
   diagnostics, per-system structural features.

Everything emits a `.npz` of Cα coordinates that
`vampnet load_ensemble` reads, or a `*_results.{md,json}` writeup.

> It's a big directory (54 scripts). The map below is the way in —
> read it before opening files. Generated `.npz` ensembles are
> git-ignored (they live on `/data`); `*_results.*` writeups are
> committed as the research log.

---

## 1. Frontier-model adapters (`*_modal.py`)

Each builds its own [Modal](https://modal.com/) image and is invoked
`modal run md/<file> --help`. Full provenance (paper, checkpoint, last-
verified date) is in each file's module docstring. Status mirrors the
root README: ✅ verified end-to-end · 🧪 scaffold (recipe drafted).

| File | Model | Role | Status |
|---|---|---|:--:|
| `alphaflow_modal.py` | AlphaFlow / ESMFlow | sequence → flow-matched ensemble | ✅ |
| `bioemu_modal.py` | BioEmu | emulated equilibrium ensemble | ✅ |
| `boltz_modal.py` | Boltz-2 | AF3-class diffusion structures | ✅ |
| `marsfm_modal.py` | MarS-FM | flow-matching in MSM space | ✅ |
| `esmfold2_modal.py` | ESMFold2 | multi-chain folded ensembles | ✅ |
| `uma_modal.py` | UMA (Meta FAIR) | universal ML force field → real MD | ✅ |
| `timewarp_modal.py` | Timewarp | trajectory-aware flow MCMC proposal | ✅¹ |
| `prose_modal.py` | Prose | transferable peptide normalizing flow | 🧪 |
| `edm_plainer_modal.py` | Plainer EDM / ScoreMD | FP-consistent diffusion | 🧪 |
| `om_tps_modal.py` | OM-TPS | zero-shot transition-path sampling | 🧪 |
| `stable_nnip_modal.py` | StABlE | stability-aware NNIP *training* | 🧪 |

¹ Timewarp samples end-to-end on alanine dipeptide but hits a
0-acceptance edge case; see its docstring + `mcmc_diagnostics.py`.

## 2. Classical MD pipeline

The OpenMM ground-truth path — both a local GB10 container and Modal.

| File | Role |
|---|---|
| `modal_md.py` | Modal `prep` + `fanout` entrypoints (parallel replicas) |
| `prep.py` | soluble-protein system prep (water box, ions, COM restraint) |
| `prep_membrane.py` | POPC-bilayer prep for membrane proteins (β2AR) |
| `produce.py` / `produce_metad.py` | run an MD / metadynamics replica |
| `notch1_metad_modal.py` | 1D/2D metadynamics fanout (PLUMED) |
| `Dockerfile`, `run_md.sh`, `slurm_md.sbatch` | local GB10 + Slurm runners |

See **§ Quickstart** below for the canonical tier-1/tier-2 run.

## 3. Reusable analysis libraries

System-agnostic — import or run these on any ensemble.

| File | Role |
|---|---|
| `multisource_h3.py` | joint VAMPnet over N sources; per-state per-source breakdown |
| `multichain.py` | chain-aware merge/chunk of multi-chain ensembles |
| `mcmc_diagnostics.py` | arviz ESS / R-hat / autocorrelation for sampler output |
| `diagnostics_sweep.py` | one-line-per-file diagnostic sweep across `*.npz` |
| `extract_ca_modal.py` | pull Cα arrays out of full-atom trajectories |
| `filter_chains.py` | subset a PDB to named chains (stdlib-only) |

## 4. Per-system analysis scripts

One-off scripts that produce the committed `*_results.*` writeups for a
specific target. Grouped by system:

- **Notch1 NRR** — `notch1_h2_modal.py`, `notch1_h3_multisource.py`,
  `notch1_h3_biology.py`, `notch1_h3_bootstrap.py`,
  `notch1_h3_2d_landscape.py`, `notch1_v3_quickcheck.py`,
  `build_state1_hybrid.py`, `figure_notch1_apo.py`
- **Hsp90 NTD** — `hsp90_ntd_h3_biology.py`, `hsp90_cryptic_pocket.py`
- **β2AR** — `b2ar_setup.py`
- **Metadynamics post-processing** — `metad_fes_postprocess.py`,
  `metad_2d_fes_postprocess.py`, `metad_apo_holo_compare.py`,
  `metad_h2_reweight.py`

## 5. Demos & validation tiers

- `tier1_chignolin.py`, `tier1_vampnet.py`, `alanine_dipeptide.py`,
  `analyze_ala.py` — smallest validation systems
- `tier2_notch1.py` — the Notch1 NRR tier
- `atlas_demo.py`, `atlas_fetch.py`, `wider_atlas_demo.py` — ATLAS
  public-trajectory robustness sweep
- `joint_md_af_demo.py`, `synthetic_alphaflow.py` — MD+AlphaFlow
  joint-VAMPnet teaching demos

## Where results land

Committed research-log outputs share a naming convention:

- `*_results.md` — human-readable interpretation (tables + biology)
- `*_results.json` — machine-readable backing data
- `*_fes_data.json` — metadynamics free-energy grids (consumed by the
  `metad_*` post-processors above)

Generated `.npz` ensembles and `.dcd`/`.chk` trajectories are
git-ignored; they live on `/data`. `zenodo_prepare.py` tarballs the
committed arrays + writeups for a Zenodo deposit (release packaging).

---

## Quickstart

All MD scripts run inside the `openmm:gb10` container (conda-forge
OpenMM + CUDA 12). Build it once:

```bash
docker build -t openmm:gb10 .
```

**Tier-1 — alanine dipeptide** (50 ns implicit solvent, ~1 h on GB10):

```bash
./run_md.sh python alanine_dipeptide.py /data/ala
# → recovers the 5 canonical Ramachandran basins (αR, β, PII, αL, γ)
```

**Tier-2 — Notch1 NRR apo** (PDB 3I08; needs an NTM-tail anchor or the
fragment dissociates — see `prep.py:_add_com_distance_restraint`):

```bash
python filter_chains.py 3i08.pdb 3i08_apo.pdb A B          # NEC=A, NTM=B
./run_md.sh python prep.py 3i08_apo.pdb /data/notch1_apo --anchor-chain-tail B:5
for r in 0 1 2; do
  ./run_md.sh python produce.py /data/notch1_apo --replica $r --steps 25000000
done
```

Load the result into the bundle:

```
open /data/notch1_apo/equilibrated.pdb
vampnet load_ensemble #1 /data/notch1_apo/replica_0/traj.dcd source md
vampnet fit nStates 4 lag 20 features ca_distances
```

For the **Modal cloud** path (parallel replica spawn on A100/H100) use
`modal_md.py` instead of the local container; see its module docstring.
