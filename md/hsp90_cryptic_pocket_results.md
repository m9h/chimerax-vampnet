# Hsp90 NTD cryptic pocket analysis — v0.7 W3

**Date**: 2026-06-05
Script: `md/hsp90_cryptic_pocket.py`
Backing data: `md/hsp90_cryptic_pocket_results.json`
Figure: `md/figures/hsp90_cryptic_pocket.png`

## Approach

CA-based geometric proxies for pocket opening, computed per frame
across the v0.7 6-source ensemble:

- **lid-floor COM**: COM(lid loop, residues 100-115) - COM(pocket
  floor, residues 50-60). Large = lid lifted off the floor.
- **Asn51–Thr109**: direct Cα-Cα distance between Asn51 (β-sheet
  binding residue) and Thr109 (Helix 2 / lid loop). Direct measure
  of the binding-pocket gate.
- **lid-Rg**: radius of gyration of the lid loop alone (residues
  100-115). Higher = more extended lid conformation.
- **Asn51–Thr115**: a second gate distance from Asn51 to the end
  of the lid loop (Thr115).

## Per-(state × source) means (Å)

| state | source | n | lid-floor | **Asn51-Thr109** | lid-Rg | Asn51-Thr115 |
|---|---|---:|---:|---:|---:|---:|
| 0 | MD_apo  |  5 534 | 15.6 ± 0.6 | **10.3 ± 1.3** | 7.9 ± 0.3 | 15.5 ± 1.1 |
| 0 | MD_holo | 24 622 | 15.9 ± 0.8 | **16.6 ± 0.9** | 7.9 ± 0.2 | 14.4 ± 0.9 |
| 1 | MD_apo  | 25 012 | 16.3 ± 0.6 | **10.1 ± 0.9** | 7.4 ± 0.2 | 16.6 ± 0.7 |
| 1 | MD_holo |     86 | 15.4 ± 0.3 | **15.3 ± 0.4** | 7.8 ± 0.1 | 13.8 ± 0.4 |
| 1 | MarS-FM |    200 | 18.4 ± 5.8 | **16.7 ± 7.0** | 7.8 ± 1.1 | 18.9 ± 6.3 |
| 1 | BioEmu  |    188 | 17.3 ± 1.1 | **16.1 ± 1.2** | 7.5 ± 0.7 | 17.1 ± 3.4 |
| 1 | Boltz-2 |    200 | 17.1 ± 0.2 | **18.0 ± 0.5** | 7.4 ± 0.1 | 16.4 ± 0.3 |
| 1 | AlphaFlow |  200 | 17.8 ± 0.9 | **16.8 ± 1.0** | 7.5 ± 0.6 | 17.7 ± 2.1 |
| 2 | MD_apo  | 14 454 | 15.9 ± 0.6 | **9.3 ± 0.6**  | 7.6 ± 0.2 | 16.6 ± 1.0 |
| 3 | MD_holo | 20 292 | 15.3 ± 0.6 | **15.4 ± 0.7** | 7.9 ± 0.3 | 14.3 ± 1.4 |

## Key finding: **bimodal pocket opening on the Asn51-Thr109 axis**

The Asn51-Thr109 distance is the cleanest discriminator. Two
distinct populations emerge:

- **CLOSED pocket** (~9-10 Å Asn51-Thr109):
  - MD_apo across all 3 of its states (0, 1, 2) — 45 000 frames
  - The pocket gate is essentially clamped shut at ~10 Å
- **OPEN pocket** (~15-18 Å Asn51-Thr109):
  - MD_holo across all its states (0, 1-tiny, 3) — 45 000 frames
  - All 4 generative samplers (200 + 188 + 200 + 200 = 788 frames),
    all in state 1
  - Pocket gate ~5-8 Å wider than apo MD

## Interpretation

This is the **inverse** of the standard "cryptic pocket" definition:

- A classical cryptic pocket is one that's *hidden* in apo (closed,
  not visible from the apo crystal) but *visible* in ligand-bound.
  Generative models that learn from the ligand-bound crystal *would*
  predict the cryptic pocket.
- For Hsp90 NTD here, the apo MD discovers a **hidden CLOSED state**
  the crystals don't show. The 1YER apo crystal (and 1YET holo
  crystal) both have the lid in an OPEN position; apo MD relaxes
  into a closed-lid conformation within 300 ns. Generative samplers,
  trained on PDB crystal forms, all reproduce the OPEN-lid state
  matching the crystal.

In short: **MD samples a cryptic CLOSED state; generative models
reproduce the crystal OPEN state.** The cryptic state is the
MD-only state, exactly the inverse of the Notch1 H3 result.

This is consistent with the v0.7 sequence-prior-collapse finding:
generative models (AlphaFlow, Boltz-2, BioEmu) reproduce the
"what does the sequence prior predict?" answer, which corresponds
to the dominant PDB crystal forms. They don't recover the
conformational selection effects that emerge from explicit
solvent + force field dynamics.

The MD_holo MD (starting from 1YET conformation with GDM stripped
by PDBFixer) stays in the open-lid conformation because the lid
was structured around the ligand in the crystal and hasn't yet
relaxed in 300 ns. So MD_holo here is best read as "starting from
holo crystal, lid-open trajectory" rather than a true holo
ensemble. (A proper holo MD would parameterize GDM via Amber GAFF
+ OpenFF and bind it explicitly; v0.7.x or later work.)

## Boltz-2 outlier

Boltz-2 on this gate distance is again the lowest-variance source:
Asn51-Thr109 = 18.0 ± 0.5 Å across 200 samples. AlphaFlow and
BioEmu have similar means but with 1-3 Å std. MarS-FM has 7.0 Å
std — the only generative sampler with meaningful pocket-opening
diversity. Replicates the systematic Boltz-2 conservatism finding
on a totally different geometric metric.

## What this means for drug design

Hsp90 NTD is a clinical drug target. Several FDA-pursued
inhibitors (geldanamycin derivatives, NVP-AUY922, PU-H71) bind in
the open-lid conformation captured by all the crystal structures.

A drug design pipeline using **only generative ensembles**
(AlphaFlow / Boltz-2 / BioEmu) would *miss* the apo-MD-discovered
CLOSED state entirely. The CLOSED state may or may not be
physiologically relevant — but if it is (e.g., as an autoinhibited
state regulating ATP binding), the generative-only approach would
not know about it.

A drug design pipeline using **only MD** at the 300-ns horizon
would *miss* the open-lid state entirely from apo trajectories;
it would only see open-lid if seeded from a holo crystal.

The v0.7 multi-source pipeline catches both: the open-lid state
from generative samplers + holo-seeded MD, and the closed-lid
state from apo MD. This is exactly the v0.5 H3 multi-source value
proposition, demonstrated on a system with direct drug-design
relevance.

## Cost

~$0 (CPU analysis on existing CA-only npz). The 90 788-frame
6-source VAMPnet refit takes ~1 min on a laptop.

## See also

- `md/hsp90_ntd_h3_results.md` — the 5-source apo + 6-source
  apo+holo joint H3 setup that this analysis projects onto.
- `md/figures/hsp90_cryptic_pocket.png` — 4-panel scatter of
  pocket-opening metrics per (state × source).
- `md/figures/hsp90_ntd_apo_holo_joint_h3.png` — the 6-source
  state breakdown.

## Open / deferred

- **Heavy-atom SASA of pocket-lining residues** via
  mdtraj.shrake_rupley — more rigorous than CA distances; needs
  per-frame heavy-atom coords (MD: have DCD; generative: have
  `coords` key). Queued for v0.7.1 if a quantitative pocket
  volume is needed.
- **fpocket** on representative frames per (state × source) — for
  true pocket geometry + sphericity. fpocket is an external tool
  not currently in the bundle stack.
- **Explicit-ligand holo MD** — re-parameterize geldanamycin via
  GAFF / OpenFF, bind explicitly, run 300 ns. Would give a true
  holo ensemble vs the current "stripped-heterogen holo-shape"
  approximation.
