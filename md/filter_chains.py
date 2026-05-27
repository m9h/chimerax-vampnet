"""Tiny helper to extract a chain subset from a PDB before passing to prep.py.

  python filter_chains.py 3i08.pdb 3i08_apo.pdb A B
  python filter_chains.py 3l95.pdb 3l95_holo.pdb A B H L
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
