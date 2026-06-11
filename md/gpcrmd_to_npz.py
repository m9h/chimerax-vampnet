"""Convert a GPCRmd-downloaded MD simulation into the CA-coordinate npz
the chimerax-vampnet H3 multisource pipeline consumes.

GPCRmd (https://www.gpcrmd.org) distributes, per simulation, a topology
file (typically `.pdb`) plus one or more trajectory files (`.xtc` / `.dcd`).
**Downloading requires a free GPCRmd login** — once the files are local,
this script extracts protein CA coordinates into `<out>.npz` with key
`coords_ca`, shape `(n_frames, n_ca, 3)`, in **Angstroms** — matching the
generative-source npz (`b2ar_2rh1_*200.npz`) so the joint multisource
VAMPnet pools MD + generative ensembles in one consistent feature space.
(mdtraj stores coordinates in nm; this script multiplies by 10.)

Usage
-----
    python md/gpcrmd_to_npz.py \
        --top  gpcrmd_b2ar/<sim>.pdb \
        --traj gpcrmd_b2ar/<sim>.xtc \
        --out  b2ar_2rh1_gpcrmd_md.npz \
        [--sel "name CA and protein"] [--stride 1] [--max-frames 0]

`--traj` may be repeated for multi-part trajectories; they are
concatenated in the order given. The default output name
`b2ar_2rh1_gpcrmd_md.npz` is exactly what md/multisource_h3.py looks for
as the β2AR MD source (system `b2ar_2rh1`), so no further wiring is
needed once it lands in the repo root.

Aligning the residue count
--------------------------
The β2AR generative sources have **282** CAs. If this script reports a
different `n_ca`, the MD selection doesn't match that 282-residue space
(common causes: extra chains like a nanobody/G-protein in active-state
sims, or different modeled termini). Use the printed per-chain CA counts
and resSeq range to refine `--sel` — e.g. `--sel "name CA and chainid 0"`
to keep only the receptor chain. The joint VAMPnet needs n_ca to match
across all sources.
"""

import argparse
from pathlib import Path

import numpy as np

NM_TO_ANG = 10.0
DEFAULT_OUT = "b2ar_2rh1_gpcrmd_md.npz"
DEFAULT_SEL = "name CA and protein"


def _one_letter(residues) -> str:
    """One-letter sequence from an iterable of mdtraj residues
    (non-standard residues -> 'X')."""
    return "".join((r.code or "X") for r in residues)


def convert(top: str, trajs: list, sel: str, stride: int, max_frames: int):
    import mdtraj as md  # lazy: only needed at runtime, keeps import light

    parts = []
    seq = None
    chain_counts = None
    resseq_lo = resseq_hi = None

    for t in trajs:
        traj = md.load(t, top=top, stride=max(1, stride))
        idx = traj.topology.select(sel)
        if len(idx) == 0:
            raise SystemExit(
                f"selection '{sel}' matched 0 atoms in {t} — check --sel")
        sub = traj.atom_slice(idx)
        parts.append(np.asarray(sub.xyz, dtype=np.float32) * NM_TO_ANG)

        if seq is None:  # diagnostics from the first trajectory
            residues = [traj.topology.atom(i).residue for i in idx]
            seq = _one_letter(residues)
            rs = [r.resSeq for r in residues]
            resseq_lo, resseq_hi = min(rs), max(rs)
            chain_counts = {}
            for a in (traj.topology.atom(i) for i in idx):
                cid = a.residue.chain.index
                chain_counts[cid] = chain_counts.get(cid, 0) + 1

    coords = np.concatenate(parts, axis=0)
    if max_frames and coords.shape[0] > max_frames:
        keep = np.linspace(0, coords.shape[0] - 1, max_frames).astype(int)
        coords = coords[keep]
    return coords, seq, resseq_lo, resseq_hi, chain_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True, help="topology file (.pdb/.psf/.prmtop)")
    ap.add_argument("--traj", required=True, action="append",
                    help="trajectory file (.xtc/.dcd); repeat for multi-part")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output npz (default: {DEFAULT_OUT})")
    ap.add_argument("--sel", default=DEFAULT_SEL,
                    help=f"mdtraj atom selection (default: {DEFAULT_SEL!r})")
    ap.add_argument("--stride", type=int, default=1, help="frame stride")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap to this many frames (uniform subsample; 0 = all)")
    ap.add_argument("--expected-n-ca", type=int, default=282,
                    help="warn if the CA count differs (β2AR generative = 282)")
    args = ap.parse_args()

    coords, seq, lo, hi, chain_counts = convert(
        args.top, args.traj, args.sel, args.stride, args.max_frames)
    n_frames, n_ca, _ = coords.shape

    np.savez_compressed(args.out, coords_ca=coords.astype(np.float32),
                        seqres=np.array(seq))

    print(f"[gpcrmd] wrote {args.out}")
    print(f"  frames : {n_frames}")
    print(f"  n_ca   : {n_ca}  (resSeq {lo}–{hi})")
    print(f"  chains : " + ", ".join(f"chain{c}={n}" for c, n in sorted(chain_counts.items())))
    print(f"  units  : Angstrom (coord span "
          f"{coords.max(axis=(0, 1)).max() - coords.min(axis=(0, 1)).min():.1f} Å)")
    if n_ca != args.expected_n_ca:
        print(f"  [warn] n_ca={n_ca} != expected {args.expected_n_ca}. The joint "
              f"VAMPnet needs a matching selection across all sources — refine "
              f"--sel (e.g. keep only the receptor chain) so n_ca == "
              f"{args.expected_n_ca}.")
    else:
        print(f"  [ok] n_ca matches the β2AR generative sources "
              f"({args.expected_n_ca}); md/multisource_h3.py will pick this up.")


if __name__ == "__main__":
    main()
