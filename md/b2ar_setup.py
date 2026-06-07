"""β2AR membrane MD setup for v0.8 W1.

Pre-processes the two β2AR PDBs (2RH1 inactive + 3P0G active-like)
into clean MD-ready inputs by stripping crystallization scaffolds:

- **2RH1**: drop BRIL/T4L lysozyme fusion at residues 1002-1161
  (replaces ICL3 in the crystal). PDBFixer's findMissingResidues
  picks up the resulting ~32-residue ICL3 gap and re-models it as
  a coil; this is acceptable for MD because ICL3 is intrinsically
  flexible in solution. Also drop one stray residue 415 (a
  crystallization additive labeled as a residue in the PDB).
- **3P0G**: drop chain B (Nb80 nanobody, 121 residues) and the
  P0G agonist (PDBFixer removeHeterogens handles the agonist; we
  drop chain B explicitly here before sending to PDBFixer).

Then drives both through `modal_md::prep --membrane` and the
fanout to produce 3 × 300 ns MD per system.

  $ .venv/bin/python md/b2ar_setup.py prep    # both systems
  $ .venv/bin/python md/b2ar_setup.py produce  # 3 replicas each
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mdtraj as md

ROOT = Path(__file__).resolve().parent.parent
RAW = Path("/tmp/b2ar_pdbs")
CLEAN = Path("/tmp/b2ar_clean")
CLEAN.mkdir(parents=True, exist_ok=True)


def _strip_2rh1():
    """Drop BRIL fusion (1002-1161) + stray 415 + non-protein."""
    print("=== 2RH1 (inactive) ===")
    t = md.load(str(RAW / "2rh1.pdb"))
    # Keep only β2AR backbone residues 29-230 and 263-342 on chain A.
    keep_indices = []
    for atom in t.topology.atoms:
        c = atom.residue.chain.chain_id
        r = atom.residue.resSeq
        # Keep β2AR residues on chain A, drop BRIL (1000s) + stray.
        if c == "A" and ((29 <= r <= 230) or (263 <= r <= 342)):
            keep_indices.append(atom.index)
    sub = t.atom_slice(keep_indices)
    out = CLEAN / "2rh1_clean.pdb"
    sub.save_pdb(str(out))
    print(f"  wrote {out}: {sub.n_atoms} atoms, {sub.n_residues} residues "
          f"(BRIL + stray + non-protein dropped)")
    return out


def _strip_3p0g():
    """Drop chain B (Nb80 nanobody) + P0G agonist."""
    print("=== 3P0G (active-like) ===")
    t = md.load(str(RAW / "3p0g.pdb"))
    # Keep only chain A protein residues (PDBFixer.removeHeterogens
    # will strip P0G later).
    keep_indices = []
    for atom in t.topology.atoms:
        c = atom.residue.chain.chain_id
        rname = atom.residue.name
        if c == "A" and rname not in {"P0G", "HOH", "WAT"}:
            keep_indices.append(atom.index)
    sub = t.atom_slice(keep_indices)
    out = CLEAN / "3p0g_clean.pdb"
    sub.save_pdb(str(out))
    print(f"  wrote {out}: {sub.n_atoms} atoms, {sub.n_residues} residues "
          f"(chain B Nb80 + P0G dropped)")
    return out


def _modal_prep_membrane(system: str, pdb_path: Path, detach: bool = True):
    """Invoke modal_md::prep with --membrane on the cleaned PDB."""
    cmd = ["modal", "run"]
    if detach:
        cmd.append("--detach")
    cmd += [str(ROOT / "md" / "modal_md.py::prep"),
            "--system", system,
            "--pdb", str(pdb_path),
            "--membrane"]
    print(f"  launching: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    for line in r.stdout.splitlines()[-8:]:
        print(f"    {line}")
    if r.returncode != 0:
        for line in r.stderr.splitlines()[-5:]:
            print(f"    [stderr] {line}")
    return r.returncode == 0


def _modal_fanout(system: str, n_replicas: int = 3, ns: float = 300.0,
                    detach: bool = True):
    cmd = ["modal", "run"]
    if detach:
        cmd.append("--detach")
    cmd += [str(ROOT / "md" / "modal_md.py::fanout"),
            "--system", system,
            "--replicas", str(n_replicas),
            "--ns", str(ns)]
    print(f"  launching: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    for line in r.stdout.splitlines()[-8:]:
        print(f"    {line}")
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["clean", "prep", "produce", "smoke"])
    p.add_argument("--ns", type=float, default=300.0)
    p.add_argument("--replicas", type=int, default=3)
    args = p.parse_args()

    if args.action == "clean":
        _strip_2rh1()
        _strip_3p0g()
        return

    if args.action == "smoke":
        # Smoke: clean both, prep 2RH1 only (faster), kick off short
        # 50 ns single replica to validate the membrane stack.
        _strip_2rh1()
        _modal_prep_membrane("b2ar_2rh1_v0p8_smoke",
                              CLEAN / "2rh1_clean.pdb")
        print("\n[next] when prep lands, run:")
        print("  modal run --detach md/modal_md.py::produce "
              "--system b2ar_2rh1_v0p8_smoke --replica 0 --ns 50")
        return

    if args.action == "prep":
        pdb_2rh1 = _strip_2rh1()
        pdb_3p0g = _strip_3p0g()
        print("\n--- launching prep for inactive (2RH1) ---")
        _modal_prep_membrane("b2ar_2rh1_v0p8", pdb_2rh1)
        print("\n--- launching prep for active-like (3P0G) ---")
        _modal_prep_membrane("b2ar_3p0g_v0p8", pdb_3p0g)
        return

    if args.action == "produce":
        print(f"--- launching production: {args.replicas} × {args.ns} ns each ---")
        for sys in ["b2ar_2rh1_v0p8", "b2ar_3p0g_v0p8"]:
            print(f"\n[{sys}]")
            _modal_fanout(sys, n_replicas=args.replicas, ns=args.ns)
        return


if __name__ == "__main__":
    main()
