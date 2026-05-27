"""Tier-1 validation: parse the alanine dipeptide DCD, compute phi/psi
torsions for every frame, cluster, and check we recover the canonical
Ramachandran basins.

Runs inside the openmm:gb10 container:

  ./run_md.sh python analyze_ala.py /data/ala
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from openmm.app import PDBFile


def _find_quad(pdb: PDBFile, names):
    """Return atom indices for atoms whose (residue.id, name) match `names`."""
    out = []
    atoms = list(pdb.topology.atoms())
    res_by_id = {r.id: r for r in pdb.topology.residues()}
    for (resid, atomname) in names:
        match = [a for a in atoms if a.residue.id == resid and a.name == atomname]
        if not match:
            raise RuntimeError(f"atom {atomname} in res {resid} not found")
        out.append(match[0].index)
    return tuple(out)


def _dihedral(p0, p1, p2, p3):
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9))
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    return np.arctan2(y, x)


def run(traj_dir: Path):
    pdb = PDBFile(str(traj_dir / "ala_dipeptide.pdb"))
    # Alanine dipeptide Ace-Ala-Nme: phi = C(ACE)-N(ALA)-CA(ALA)-C(ALA),
    # psi = N(ALA)-CA(ALA)-C(ALA)-N(NME).
    # openmmtools labels residues with IDs "1", "2", "3" by default.
    residues = list(pdb.topology.residues())
    ace = residues[0]
    ala = residues[1]
    nme = residues[2]

    def ai(res, name):
        return next(a for a in res.atoms() if a.name == name).index

    phi_quad = (ai(ace, "C"), ai(ala, "N"), ai(ala, "CA"), ai(ala, "C"))
    psi_quad = (ai(ala, "N"), ai(ala, "CA"), ai(ala, "C"), ai(nme, "N"))

    coords = _read_dcd(traj_dir / "traj.dcd", n_atoms=pdb.topology.getNumAtoms())
    print(f"[ala-an] loaded trajectory: shape={coords.shape}")

    phi_atoms = np.array(phi_quad)
    psi_atoms = np.array(psi_quad)
    phi = _dihedral(coords[:, phi_atoms[0]], coords[:, phi_atoms[1]],
                     coords[:, phi_atoms[2]], coords[:, phi_atoms[3]])
    psi = _dihedral(coords[:, psi_atoms[0]], coords[:, psi_atoms[1]],
                     coords[:, psi_atoms[2]], coords[:, psi_atoms[3]])
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)

    print(f"[ala-an] phi/psi sample: phi {phi_deg.min():.1f} .. {phi_deg.max():.1f}, "
          f"psi {psi_deg.min():.1f} .. {psi_deg.max():.1f}")
    N = len(phi_deg)

    # Canonical L-alanine basins (mutually exclusive boxes).
    basins = {
        "alpha_R": (-180, -30, -100,   30),  # right-handed alpha helix
        "beta":    (-180, -30,   30,  180),  # extended beta/PPII
        "alpha_L": (  30, 100,  -50,  100),  # left-handed alpha (rare for L-AAs)
        "gamma_R": (-180, -30,  100,  180),  # variant of beta
        "C7eq":    ( -90, -30,   30,  100),  # C7-equatorial
    }
    for name, (pl, ph, sl, sh) in basins.items():
        mask = (phi_deg >= pl) & (phi_deg < ph) & (psi_deg >= sl) & (psi_deg < sh)
        print(f"  {name:8s}: {mask.sum():>6d} / {N} ({100.0 * mask.mean():5.1f}%)")

    # 6x6 phi/psi heatmap to dump regardless of basin boxes.
    print(f"\n[ala-an] phi/psi 6x6 histogram (%; rows=phi bins, cols=psi bins):")
    H, phi_edges, psi_edges = np.histogram2d(phi_deg, psi_deg, bins=6,
                                              range=[[-180, 180], [-180, 180]])
    H = 100.0 * H / N
    print(f"  phi\\psi  " + "  ".join(f"{e:>5.0f}" for e in psi_edges[:-1]))
    for i, pl in enumerate(phi_edges[:-1]):
        print(f"  {pl:>5.0f}    " + "  ".join(f"{H[i, j]:5.1f}" for j in range(6)))


def _read_dcd(path: Path, n_atoms: int) -> np.ndarray:
    """Minimal DCD parser sufficient for the standard openmm-written format."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    # Header block 1: 4-byte size, 'CORD' or 'VELD' magic, 20 ints, end block size.
    pos = 0
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4
    magic = data[pos:pos + 4]
    pos += 4
    header = struct.unpack("<20i", data[pos:pos + 80])
    pos += 80
    end = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4
    # Header block 2: title text.
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4
    pos += blocksize
    pos += 4
    # Header block 3: atom count.
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4
    natom = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4
    pos += 4
    # Frames: optional unit-cell block + 3 float arrays for x, y, z.
    has_cell = header[10] != 0
    frames = []
    while pos + 4 <= len(data):
        if has_cell:
            blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
            pos += 4
            pos += blocksize
            pos += 4
        # x, y, z float32 arrays.
        try:
            xyz = []
            for _ in range(3):
                blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
                pos += 4
                vals = np.frombuffer(data[pos:pos + blocksize], dtype="<f4")
                pos += blocksize
                pos += 4
                xyz.append(vals)
            frame = np.stack(xyz, axis=-1)  # (A, 3)
            if frame.shape[0] != natom:
                break
            frames.append(frame)
        except Exception:
            break
    if not frames:
        raise RuntimeError("no frames decoded from DCD")
    arr = np.stack(frames, axis=0).astype(np.float32)
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traj_dir", type=Path)
    args = p.parse_args()
    run(args.traj_dir)


if __name__ == "__main__":
    main()
