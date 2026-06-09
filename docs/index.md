# 🧬 chimerax-vampnet

**A teaching resource for learning protein structure and dynamics by driving
frontier biomolecular models — built for the Biopunk Lab HTGAA ("How To Grow
Almost Anything") course.**

---

This project is two things at once:

1. A **ChimeraX bundle** that loads heterogeneous structural ensembles
   (classical MD, AlphaFlow, BioEmu, Boltz-2, …) on equal footing, trains a
   VAMPnet ([Mardt et al. 2018](https://www.nature.com/articles/s41467-017-02388-1))
   via [deeptime](https://deeptime-ml.github.io/), and surfaces metastable
   states + transition rates as ChimeraX models, animations, and structured
   CLI output.
2. A **catalog of working cloud recipes** for invoking ~11 frontier protein
   models — the kind announced in papers and seminars but rarely runnable
   without a week of dependency archaeology. Each one is a single `modal run`
   away.

!!! quote "The point"
    The goal is **not** a new scientific discovery. It is to let a student
    **reproduce frontier-research workflows end to end** — pick a protein,
    generate conformational ensembles from a dozen different models, and *see*
    its energy landscape — and in doing so learn how structure becomes
    dynamics, and how today's models do (and don't) capture that.

## Where to go next

<div class="grid cards" markdown>

-   :rocket: **[Frontier-model catalog](catalog.md)**

    The 11 frontier models you can run, what each gives you, and which are
    verified end-to-end vs. scaffolded.

-   :mortar_board: **[Student workflow](workflow.md)**

    The `structure → ensemble → landscape` path, walked step by step on a
    real protein.

-   :microscope: **[Analysis layer](analysis.md)**

    The ChimeraX `vampnet` commands that turn ensembles into a landscape you
    can scrub, animate, and compare.

-   :link: **[Connections](connections.md)**

    The DNA-origami sister project and the shared *contact-map / frustration*
    idea that ties protein folding to nanostructure assembly.

</div>

## The big idea: structure → ensemble → landscape

A crystal structure is **one** snapshot. A protein's *function* lives in the
**ensemble** of shapes it visits — and in the **landscape** of barriers
between them. This project teaches that progression hands-on:

```
   sequence / PDB
        │
        ▼   (frontier model OR classical MD — the catalog)
   conformational ensemble        ← "what shapes does it visit?"
        │
        ▼   (vampnet load_ensemble + fit, in ChimeraX)
   metastable states + rates      ← "how is the landscape organized?"
        │
        ▼   (vampnet states / means / animate / network)
   colored 3D models you can scrub, animate, and compare
```

The payoff is **comparison**: load an MD trajectory *and* an AlphaFlow ensemble
*and* a BioEmu ensemble into the same VAMPnet, and you can literally watch which
models explore the real conformational equilibrium and which collapse onto the
crystal prior. That comparison is the central teaching moment — and a live
research question.

!!! note "Status"
    Current version **v0.10**. 7 of 11 frontier adapters are verified
    end-to-end; the rest are honestly-marked scaffolds. See the
    [catalog](catalog.md) for the per-model status.
