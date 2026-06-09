# 🔗 Connections

## Sister project: `chimerax-origami`

chimerax-vampnet has a deliberate mirror image —
[`chimerax-origami`](https://github.com/m9h/chimerax-origami) — built for the
**DNA-origami** half of the same HTGAA curriculum. The two share **one
abstraction — the contact map — and one thesis: folding/assembly reliability is
the minimization of landscape *frustration*.**

| | chimerax-vampnet | chimerax-origami |
|---|---|---|
| Object | protein conformational landscape | DNA-origami assembly landscape |
| Input | MD / AlphaFlow / BioEmu ensembles | cadnano / scadnano / oxDNA designs |
| Shared substrate | Cα–Cα **contact map** | base-pair **contact map** |
| "Frustration" | metastable kinetic traps | off-target hybridization |
| Inverse design | adaptive sampling | Pareto scaffold selection |
| Perturbation | antibody shifts state populations | lipid envelope shifts in-vivo fate |
| Driver | MCP LLM agent in the loop | MCP LLM agent in the loop |

The connection is concrete, not just analogy: the Notch1 NRR is a *membrane*
receptor, and the origami side wraps nanostructures in a *lipid envelope*
(Perrault & Shih, virus-inspired encapsulation) — both projects independently
land on the **membrane interface** as the boundary condition, and a
membrane-binding event as the control knob that re-weights an ensemble. Stack
them and you get a closed design–build–test–learn loop: origami builds the
enveloped chassis, VAMPnet characterizes the protein cargo's dynamics, and an
MCP agent drives both.

!!! abstract "The transferable lesson"
    **structure → contact map → landscape → de-frustration** is the same
    workflow whether you're folding a protein or assembling a nanostructure.

## Toward `biopunk-biophysics`

The eventual plan is to **merge the two bundles into a single meta-project**
(working name *biopunk-biophysics*). `chimerax-origami` already mirrors this
bundle's module layout 1:1 (`contactmap.py` ↔ `featurize.py`, `score.py` ↔
`vampnet_core.py`, …), so the merged repo can be as simple as `vampnet/` +
`origami/` + a shared `core/` for the contact-map abstraction, under one README.

## Seminar series

Several models in the [catalog](catalog.md) — UMA, Timewarp, OM-TPS,
Plainer-EDM, Prose, StABlE — were first presented in the **Starkly Speaking**
seminar series. Each adapter's module docstring carries the talk date and the
paper, so a student can pair the recorded talk with a runnable recipe.

## References

- Mardt, Pasquali, Wu & Noé. *VAMPnets for deep learning of molecular kinetics.*
  Nat. Commun. **9**, 5 (2018).
- Hoffmann et al. *Deeptime: a Python library for machine learning dynamical
  models from time series data.* Mach. Learn. Sci. Technol. **3**, 015009 (2021).
- Perrault & Shih. *Virus-Inspired Membrane Encapsulation of DNA Nanostructures
  To Achieve In Vivo Stability.* ACS Nano (2014).
- Shirt-Ediss, Torelli, Navarro & Krasnogor. *Optimising DNA origami assembly by
  reducing off-target interactions.* Nat. Commun. (2026).

Each `md/*_modal.py` adapter's docstring cites the specific frontier model it
wraps (paper, arXiv, code, checkpoint) — start there to credit the model authors.
