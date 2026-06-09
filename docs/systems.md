# 🧪 Validation systems

The pipeline is wired and exercised on five systems of increasing difficulty.
Each is a self-contained teaching example.

| System | Why it's interesting | What's in the repo |
|---|---|---|
| **Alanine dipeptide** | Canonical 5-basin Ramachandran toy | MD + Timewarp ensembles |
| **Chignolin (CLN025)** | Fast-folder w/ DESRES Anton gold standard | 1 µs self-generated MD, folded/unfolded split |
| **Notch1 NRR** | Membrane-receptor conformational *switch* (apo vs antibody-bound) | MD + 5 generative sources, multi-chain ESMFold2, metadynamics |
| **Hsp90α NTD** | Chaperone w/ a cryptic drug pocket | MD + 5 sources; cryptic-state discovery |
| **β2-adrenergic receptor** | Gold-standard Class-A GPCR, drug target | 5 generative ensembles + GPCR-specific structural analysis |

## What each system teaches

### Alanine dipeptide & chignolin — calibration
The smallest systems exist to prove the pipeline recovers *known* answers: the
five Ramachandran basins for the dipeptide, and a clean folded/unfolded
separation for chignolin (with a slowest implied timescale in the published
100–500 ns range).

### Notch1 NRR — generative models *add* coverage
A membrane-receptor switch where 100 ns of MD can't reach every relevant state.
Here the generative ensembles **extend** sampling beyond what MD alone sees, and
metadynamics maps the apo-vs-antibody free-energy difference.

### Hsp90α NTD — generative models *collapse*
The inverse lesson. The generative models collapse onto the most-populated
crystal conformation while MD broadly samples states they miss — including a
**closed-lid cryptic pocket** crystals don't show. The same multi-source
workflow yields the opposite conclusion from Notch1, which is exactly the point:
*the answer is system-dependent, and you have to look.*

### β2-adrenergic receptor — the membrane stress test
A clinically critical Class-A GPCR (the target of asthma β-agonists and
heart-failure β-blockers). It forces the pipeline onto a lipid bilayer and asks
whether soluble-trained generative models produce anything sensible for a
membrane protein. Five generative ensembles plus GPCR-specific structural
markers (TM6 outward movement, ionic lock, NPxxY motif) are wired; the membrane
MD itself is an [open thread](https://github.com/m9h/chimerax-vampnet) (a
numerical instability at equilibration — a realistic example of where membrane
systems bite).

!!! info "The cross-system finding"
    Taken together these systems probe one question — *do generative ensembles
    reflect the conformational equilibrium, or just the sequence prior?* — and
    the honest answer differs by system. That tension is the research seed; for
    a student it is a concrete, reproducible way to learn what these models can
    and cannot do.
