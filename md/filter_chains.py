"""Tiny helper to extract a chain subset from a PDB before passing to prep.py.

Apo Notch1 NRR (3I08) -- chain A (NEC, residues 1449-1622) + chain B
(NTM, 1670-1729) are the NRR:

  python filter_chains.py 3i08.pdb 3i08_apo.pdb A B

Holo Notch1 NRR + anti-NRR Fab (3L95) -- the ASU is a 2:2 NRR:Fab dimer.
The NRR is in chains X (residues 1447-1727) and Y (1461-1726); the Fab
heavy chains are B/H and light chains are A/L. Keep one NRR copy + one
Fab pair:

  python filter_chains.py 3l95.pdb 3l95_holo.pdb X H L

NOT `A B H L` (those are all Fab; the original holo run drained ~$140
of Modal MD on two Fab antibodies floating in water).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    keep = set(sys.argv[3:])
    n_in = n_kept = 0
    with src.open() as fi, dst.open("w") as fo:
        for line in fi:
            if line.startswith(("ATOM", "HETATM", "TER")):
                n_in += 1
                chain = line[21:22]
                if chain not in keep:
                    continue
                n_kept += 1
                fo.write(line)
            elif line.startswith(("HEADER", "TITLE", "REMARK", "CRYST1", "ENDMDL", "END", "MODEL")):
                fo.write(line)
    fo = open(dst, "a")
    fo.write("END\n")
    fo.close()
    print(f"[filter] {src} -> {dst}: kept {n_kept}/{n_in} ATOM/HETATM/TER lines, chains={sorted(keep)}")


if __name__ == "__main__":
    main()
