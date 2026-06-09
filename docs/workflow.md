# 🎓 A worked student path

The whole project is one progression a student walks: **a sequence becomes an
ensemble becomes a landscape.** Here is that path, end to end.

## 1. Pick a target

Start small, then graduate to a real receptor:

- **Alanine dipeptide** — the canonical 5-basin Ramachandran toy.
- **Chignolin (CLN025)** — a fast-folder with a DESRES Anton gold standard.
- **A real receptor** — Notch1 NRR, Hsp90α NTD, or the β2-adrenergic receptor
  (see [Validation systems](systems.md)).

## 2. Generate ensembles three ways

The teaching value is in the *contrast* between methods, so generate the same
target's ensemble from sources that work differently:

```bash
# classical molecular dynamics (ground truth)
modal run md/modal_md.py --pdb target.pdb --name target ...

# a flow-matching generative model
modal run md/alphaflow_modal.py --sequence <SEQ> --name target --out target_af200.npz

# an equilibrium emulator
modal run md/bioemu_modal.py --sequence <SEQ> --name target --out target_bioemu200.npz
```

See the [catalog](catalog.md) for every available generator.

## 3. Load all three into one VAMPnet

Inside ChimeraX, load every ensemble into the *same* model and fit a shared
landscape:

```
vampnet load_ensemble md      target_md.dcd       source md
vampnet load_ensemble alphaflow target_af200.npz   format alphaflow
vampnet load_ensemble bioemu  target_bioemu200.npz format bioemu
vampnet fit nStates 4 lag 20 features ca_distances
```

## 4. Compare — the central teaching moment

```
vampnet states     # color frames by metastable state, live as you scrub
vampnet network    # transition rates between states
```

Now ask: **does the generative model populate the same basins MD does, or
collapse to one?**

!!! example "A real, re-runnable finding"
    On **Hsp90 NTD**, the generative models collapse to the crystal prior while
    MD discovers a *closed-lid cryptic state* the generative models miss — a
    genuine result you can reproduce with this pipeline. On **Notch1**, by
    contrast, generative ensembles *add* coverage that 100 ns of MD can't reach.
    Same workflow, opposite lessons.

## 5. Perturb and re-watch

Add an antibody or ligand context and re-fit: the stationary populations shift.
This is how a *binding event* re-weights a conformational ensemble — the same
abstraction that, on the [DNA-origami side](connections.md), a lipid envelope
applies to a nanostructure's *fate* landscape.

## The progression, in one diagram

```
   sequence / PDB
        │
        ▼   frontier model OR classical MD
   conformational ensemble
        │
        ▼   vampnet load_ensemble + fit
   metastable states + transition rates
        │
        ▼   vampnet states / means / animate / network
   colored 3D models — scrub, animate, compare, perturb
```

Continue to the [analysis layer](analysis.md) for the full command reference.
