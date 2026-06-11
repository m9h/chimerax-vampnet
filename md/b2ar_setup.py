"""β2AR membrane MD setup for v0.8 W1.

Seeds from **OPM (Orientations of Proteins in Membranes)** coordinates,
not raw RCSB. This is the fix for the v0.8 W1 NaN-at-NVT: OpenMM's
`Modeller.addMembrane` *requires* the protein pre-oriented with the
membrane normal along z and centered on the bilayer. Raw RCSB structures
are in the crystal frame, so addMembrane inserts the POPC bilayer in the
wrong plane, the lipids clash through the 7TM bundle, and the system NaNs
at the first NVT step. OPM serves the same structures rotated into the
membrane frame (membrane core z≈[-15,15], centered at 0) plus DUM
boundary-marker atoms — exactly what addMembrane needs.

Pre-processes the two β2AR structures into clean, membrane-oriented,
MD-ready inputs by stripping non-receptor atoms (note OPM re-letters
chains, so the receptor chain differs per structure):

- **2RH1** (OPM chain **C**, the first of two identical copies): keep
  receptor residues 29-230 + 263-342 (= the 282-CA span that matches the
  β2AR generative sources). Drops the T4L fusion (1002-1161, replaces
  ICL3 — PDBFixer re-models the resulting flexible ICL3 gap as coil), the
  duplicate copy (chain D), the DUM markers, and crystallization
  additives. ICL3 is intrinsically flexible in solution, so coil
  re-modeling is acceptable for MD.
- **3P0G** (OPM chain **A**, the receptor): keep chain-A standard
  residues only — drops chain B (Nb80 nanobody), the P0G agonist, DUM
  markers, and additives.

Then drives both through `modal_md::prep --membrane` and the
fanout to produce 3 × 300 ns MD per system.

  $ .venv/bin/python md/b2ar_setup.py prep    # both systems
  $ .venv/bin/python md/b2ar_setup.py produce  # 3 replicas each
"""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

import mdtraj as md

ROOT = Path(__file__).resolve().parent.parent
RAW = Path("/tmp/b2ar_pdbs")
CLEAN = Path("/tmp/b2ar_clean")
CLEAN.mkdir(parents=True, exist_ok=True)

OPM_BASE = "https://opm-assets.storage.googleapis.com/pdb"

# Standard amino-acid residue names (+ common protonation/disulfide
# variants). Filtering to these drops OPM DUM markers, ligands, and
# crystallization additives while preserving the (oriented) receptor.
STD_AA = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
    "THR TRP TYR VAL HID HIE HIP CYX".split())


def _fetch_opm(pdbid: str) -> Path:
    """Download OPM membrane-oriented coordinates for `pdbid` into RAW.
    OPM rotates the structure so the membrane normal is z and the bilayer
    is centered at z=0 — the orientation Modeller.addMembrane assumes.
    Cached on disk after the first fetch."""
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{pdbid}_opm.pdb"
    if out.exists() and out.stat().st_size > 10000:
        return out
    url = f"{OPM_BASE}/{pdbid}.pdb"
    print(f"[opm] fetching membrane-oriented {pdbid} <- {url}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "chimerax-vampnet/0.10"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    out.write_bytes(data)
    print(f"[opm]   wrote {out} ({len(data) // 1024} KB)")
    return out


def _strip_2rh1():
    """OPM-oriented 2RH1 -> receptor only (chain C). Keeps residues
    29-230 + 263-342 (the 282-CA span matching the β2AR generative
    sources); drops the T4L fusion (1002-1161), the duplicate copy
    (chain D), DUM markers, and additives."""
    print("=== 2RH1 (inactive) ===")
    t = md.load(str(_fetch_opm("2rh1")))
    keep_indices = []
    for atom in t.topology.atoms:
        c = atom.residue.chain.chain_id
        r = atom.residue.resSeq
        # OPM re-letters the receptor to chain C; STD_AA drops DUM +
        # additives; the ranges drop the T4L-occupied ICL3 region.
        if c == "C" and atom.residue.name in STD_AA \
                and ((29 <= r <= 230) or (263 <= r <= 342)):
            keep_indices.append(atom.index)
    sub = t.atom_slice(keep_indices)
    out = CLEAN / "2rh1_clean.pdb"
    sub.save_pdb(str(out))
    print(f"  wrote {out}: {sub.n_atoms} atoms, {sub.n_residues} residues "
          f"(membrane-oriented; T4L + dup chain + DUM dropped)")
    return out


def _strip_3p0g():
    """OPM-oriented 3P0G -> receptor only (chain A). Drops chain B
    (Nb80 nanobody), the P0G agonist, DUM markers, and additives."""
    print("=== 3P0G (active-like) ===")
    t = md.load(str(_fetch_opm("3p0g")))
    keep_indices = []
    for atom in t.topology.atoms:
        # OPM keeps the receptor on chain A; Nb80 is on chain B (excluded
        # by the chain test), and STD_AA drops the P0G agonist, DUM
        # markers, and crystallization additives.
        if atom.residue.chain.chain_id == "A" \
                and atom.residue.name in STD_AA:
            keep_indices.append(atom.index)
    sub = t.atom_slice(keep_indices)
    out = CLEAN / "3p0g_clean.pdb"
    sub.save_pdb(str(out))
    print(f"  wrote {out}: {sub.n_atoms} atoms, {sub.n_residues} residues "
          f"(membrane-oriented; Nb80 + P0G + DUM dropped)")
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
