"""Build a hybrid AF-state-1 NEC + apo NTM all-atom PDB for the
v0.7 W3d Phase 2 adaptive sampling experiment.

The question: if MD is *seeded* in the AlphaFlow state-1 conformation,
does it stay there or relax back to MD-equilibrium states 0/3?

Approach:
  1. Load v0.3 apo equilibrated.pdb (NEC chain A + NTM chain B +
     solvent + ions, already PDBFixed and equilibrated).
  2. Kabsch-align the AF-state-1 mean Cα structure onto apo chain A
     Cα atoms (so the AF target is in the right coordinate frame).
  3. For each NEC residue, translate ALL atoms by the per-residue
     Cα displacement: target_Cα - current_Cα. This preserves
     intra-residue side-chain geometry but moves residues to the
     AF-state-1 backbone.
  4. NTM chain B + waters + ions untouched.
  5. Write hybrid PDB.

After this, run modal_md.py::prep on the hybrid PDB. PDBFixer will
strip the dirty waters (some may now overlap moved chain A atoms)
and re-solvate; OpenMM's minimization will fix peptide bond
distortions from the rigid-residue displacement.

Output: /tmp/notch1_state1seed.pdb (then upload to Modal)
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

APO_PDB = "/tmp/notch1_v3_topo/equilibrated_v3.pdb"
AF_MEAN_NPZ = ROOT / "md" / "notch1_af_state1_mean.npz"
OUT_PDB = "/tmp/notch1_state1seed.pdb"


def main():
    print("=" * 60)
    print("W3d Phase 2: build hybrid AF-state-1 NEC + apo NTM PDB")
    print("=" * 60)

    # Load AF mean (in A, 174 CAs).
    af_data = np.load(AF_MEAN_NPZ)
    af_mean_A = af_data["coords_ca"].astype(np.float32)  # Å
    print(f"AF mean Cα: {af_mean_A.shape} (Å)")

    # Load apo equilibrated.pdb.
    print(f"loading {APO_PDB}…")
    apo = md.load(APO_PDB)
    print(f"  atoms {apo.n_atoms}, residues {apo.n_residues}")

    # Chain A CA indices.
    chain_a = [c for c in apo.topology.chains if c.chain_id == "A"][0]
    chain_a_ca_idx = [a.index for a in chain_a.atoms if a.name == "CA"]
    assert len(chain_a_ca_idx) == 174, (
        f"expected 174 CAs on chain A, got {len(chain_a_ca_idx)}"
    )
    apo_ca_A = apo.xyz[0, chain_a_ca_idx, :] * 10.0  # nm → Å
    print(f"apo chain A Cα: {apo_ca_A.shape}")

    # Kabsch-align AF mean ONTO apo chain A Cα.
    af_centered = af_mean_A - af_mean_A.mean(0)
    apo_centered = apo_ca_A - apo_ca_A.mean(0)
    cov = af_centered.T @ apo_centered
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    af_aligned = af_centered @ rot + apo_ca_A.mean(0)
    rmsd_initial = float(np.sqrt(((af_aligned - apo_ca_A) ** 2).sum(-1).mean()))
    print(f"AF mean → apo chain A Cα-RMSD after Kabsch: {rmsd_initial:.2f} Å")

    # Per-residue Cα displacement: target - current.
    # Build a residue-index → Cα displacement map for chain A.
    new_xyz = apo.xyz[0].copy()  # nm
    ca_displacements_A = []
    for res_idx_in_chain, res in enumerate(chain_a.residues):
        # Find this residue's Cα atom.
        ca_atoms = [a for a in res.atoms if a.name == "CA"]
        if not ca_atoms:
            continue
        ca_atom = ca_atoms[0]
        current_ca_A = new_xyz[ca_atom.index] * 10.0  # nm → Å
        target_ca_A = af_aligned[res_idx_in_chain]
        disp_A = target_ca_A - current_ca_A
        disp_nm = disp_A / 10.0
        ca_displacements_A.append(np.linalg.norm(disp_A))
        # Translate ALL atoms of this residue.
        for atom in res.atoms:
            new_xyz[atom.index] += disp_nm
    print(f"per-residue Cα displacement: mean={np.mean(ca_displacements_A):.2f} Å, "
          f"max={np.max(ca_displacements_A):.2f} Å")

    # Construct new Trajectory with the same topology.
    new_traj = md.Trajectory(
        xyz=new_xyz[None, :, :],
        topology=apo.topology,
        unitcell_lengths=apo.unitcell_lengths,
        unitcell_angles=apo.unitcell_angles,
    )

    # Save.
    new_traj.save_pdb(OUT_PDB)
    print(f"\nwrote {OUT_PDB}")

    # Verify by re-loading.
    check = md.load(OUT_PDB)
    check_ca_A = check.xyz[0, chain_a_ca_idx, :] * 10.0
    rmsd_check = float(np.sqrt(((check_ca_A - af_aligned) ** 2).sum(-1).mean()))
    print(f"verification: hybrid chain A Cα RMSD vs AF-mean target = "
          f"{rmsd_check:.2f} Å (should be ~0)")


if __name__ == "__main__":
    main()
