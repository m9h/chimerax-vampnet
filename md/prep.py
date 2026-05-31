"""System preparation for Notch1 NRR MD (and other multi-chain proteins).

Pipeline:
  1. PDBFixer cleanup (missing residues, missing atoms, hydrogens).
  2. Solvate in TIP3P with >= 10 A padding.
  3. Neutralize with NaCl to ~150 mM.
  4. Energy-minimize.
  5. NVT equilibration at 310 K (500 ps).
  6. NPT equilibration at 1 atm (500 ps).

Output: a .xml system file + a .pdb file containing the equilibrated
starting structure for production MD. Drives off a single CLI:

  python prep.py 3i08.pdb prepared_3i08/

The prepared output is what produce.py reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openmm import app, unit, CustomCentroidBondForce, LangevinIntegrator, MonteCarloBarostat, Platform, XmlSerializer
from openmm import Vec3
from pdbfixer import PDBFixer


def _chain_ca_indices(topology, chain_id):
    idx = []
    for chain in topology.chains():
        if chain.id != chain_id:
            continue
        for res in chain.residues():
            for atom in res.atoms():
                if atom.name == "CA":
                    idx.append(atom.index)
    return idx


def _compute_com_distance_nm(positions, indices_a, indices_b):
    """Plain-Python COM-to-COM distance in nm."""
    import math
    sxa = sya = sza = sxb = syb = szb = 0.0
    for i in indices_a:
        p = positions[i]
        try:
            p = p.value_in_unit(unit.nanometer)
        except AttributeError:
            pass
        sxa += p.x; sya += p.y; sza += p.z
    for i in indices_b:
        p = positions[i]
        try:
            p = p.value_in_unit(unit.nanometer)
        except AttributeError:
            pass
        sxb += p.x; syb += p.y; szb += p.z
    na, nb = len(indices_a), len(indices_b)
    dx = sxa/na - sxb/nb; dy = sya/na - syb/nb; dz = sza/na - szb/nb
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _add_com_distance_restraint(system, topology, positions, chain_a, chain_b,
                                  k_kj_per_nm2: float = 100.0):
    """Harmonic restraint on |COM(chain_a) - COM(chain_b)| around the
    initial distance, computed from positions.

    Compensates for the missing transmembrane anchor on Notch1 NRR by
    preventing NEC (chain_a) from dissociating from NTM (chain_b),
    while leaving thermal motion of individual residues untouched.
    Invariant to whole-system drift (per-atom positional restraints at
    sufficient k to constrain drift NaN at HMR=4 fs; this formulation
    avoids that pathology because there are no per-atom force
    singularities). Returns (r0_nm, n_atoms_a, n_atoms_b)."""
    a_idx = _chain_ca_indices(topology, chain_a)
    b_idx = _chain_ca_indices(topology, chain_b)
    if not a_idx or not b_idx:
        raise ValueError(f"no CAs for chain {chain_a!r} or {chain_b!r}")
    r0 = _compute_com_distance_nm(positions, a_idx, b_idx)
    force = CustomCentroidBondForce(2, "0.5*k*(distance(g1, g2) - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    force.addGroup(a_idx)
    force.addGroup(b_idx)
    force.addBond([0, 1], [
        k_kj_per_nm2 * unit.kilojoule_per_mole / unit.nanometer ** 2,
        r0 * unit.nanometer,
    ])
    system.addForce(force)
    return r0, len(a_idx), len(b_idx)


def prepare_system(pdb_in: Path, out_dir: Path, padding_nm: float = 1.0,
                   ionic_strength_M: float = 0.15, temperature_K: float = 310.0,
                   ph: float = 7.4, ff: str = "amber14", dt_fs: float = 4.0,
                   hmr_amu: float = 4.0,
                   com_restrain: tuple | None = None,
                   com_restrain_k: float = 100.0,
                   max_internal_gap: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep] fixing {pdb_in} -> {out_dir}/fixed.pdb")
    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    # Drop (a) terminal missing residues (would build artificial chain ends)
    # and (b) internal gaps longer than max_internal_gap (de novo modeling of
    # long disordered loops at protein interfaces clashes with bound partners
    # and blows up NVT). The Notch1 NRR's S1 cleavage loop is the canonical
    # case: residues 1622-1670 are unresolved in 3L95 because they're the
    # disordered S1 cleavage site; modeling 48 de novo residues against the
    # bound anti-NRR Fab caused NVT to NaN.
    keys = list(fixer.missingResidues.keys())
    for key in keys:
        chain = list(fixer.topology.chains())[key[0]]
        n_res = sum(1 for _ in chain.residues())
        gap_len = len(fixer.missingResidues[key])
        if key[1] == 0 or key[1] == n_res:
            del fixer.missingResidues[key]
        elif gap_len > max_internal_gap:
            print(f"[prep]   dropping internal {gap_len}-residue gap in chain "
                  f"{chain.id} at insertion {key[1]}; will leave as chain break")
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(out_dir / "fixed.pdb", "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

    print(f"[prep] loading force field ({ff})")
    if ff == "amber14":
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    else:
        raise ValueError(f"unknown forcefield: {ff}")

    print(f"[prep] solvating with TIP3P (padding {padding_nm} nm, ionic strength {ionic_strength_M} M)")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(
        forcefield,
        padding=padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_M * unit.molar,
        neutralize=True,
    )
    with open(out_dir / "solvated.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    print("[prep] creating OpenMM system")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=hmr_amu * unit.amu,  # HMR; with 4 amu safe to integrate at 4 fs
    )

    # COM-distance restraint (if any) is added AFTER NPT below, using the
    # equilibrated COM separation as the reference. We replaced the
    # per-atom positional anchor with a COM-COM restraint because the
    # per-atom version at k stiff enough to constrain whole-protein drift
    # NaN's at HMR=4 fs (force-singularity pathology); the COM-COM
    # restraint is invariant to whole-system drift and avoids per-atom
    # force singularities.

    print("[prep] energy minimization")
    integrator = LangevinIntegrator(temperature_K * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond)
    platform = _select_platform()
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=10000)

    eq_steps = int(500.0 / dt_fs * 1000)  # 500 ps regardless of dt
    print(f"[prep] NVT equilibration (500 ps, dt={dt_fs} fs, {eq_steps} steps)")
    simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    simulation.step(eq_steps)

    print(f"[prep] NPT equilibration (500 ps, 1 atm, {eq_steps} steps)")
    barostat = MonteCarloBarostat(1.0 * unit.atmosphere, temperature_K * unit.kelvin)
    barostat_idx = system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(eq_steps)

    # Capture post-NPT state for saving.
    state_eq = simulation.context.getState(
        getPositions=True, getVelocities=True)

    if com_restrain:
        chain_a, chain_b = com_restrain
        print(f"[prep] removing barostat (production will be NVT)")
        system.removeForce(barostat_idx)
        # Save the restraint spec to disk; produce.py re-applies the
        # CustomCentroidBondForce at run start using loaded positions.
        # We identify chains by their iteration INDEX in the topology
        # rather than .id, because PDBFixer/Modeller chain IDs may be
        # renamed in the equilibrated PDB output (input X K H L -> A B C D)
        # but the iteration order is preserved.
        mt_chains = list(modeller.topology.chains())
        try:
            a_index = next(i for i, c in enumerate(mt_chains) if c.id == chain_a)
            b_index = next(i for i, c in enumerate(mt_chains) if c.id == chain_b)
        except StopIteration:
            raise ValueError(
                f"--com-restrain chain ids {chain_a!r}/{chain_b!r} not found "
                f"in topology (available: {[c.id for c in mt_chains[:6]]}...)")
        a_atoms = [a.index for a in mt_chains[a_index].atoms() if a.name == "CA"]
        b_atoms = [a.index for a in mt_chains[b_index].atoms() if a.name == "CA"]
        r0 = _compute_com_distance_nm(state_eq.getPositions(), a_atoms, b_atoms)

        import json
        spec = {
            "type": "com_distance",
            "chain_a_index": a_index,
            "chain_b_index": b_index,
            "k_kj_per_nm2": com_restrain_k,
        }
        with open(out_dir / "anchor_specs.json", "w") as f:
            json.dump(spec, f)
        print(f"[prep] saved COM-distance restraint spec: "
              f"chain idx {a_index} ({chain_a!r}, {len(a_atoms)} CAs) <-> "
              f"chain idx {b_index} ({chain_b!r}, {len(b_atoms)} CAs), "
              f"r0={r0*10:.2f} A, k={com_restrain_k} kJ/mol/nm^2")

    print(f"[prep] writing equilibrated coordinates -> {out_dir}/equilibrated.pdb")
    with open(out_dir / "equilibrated.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, state_eq.getPositions(), f)

    print(f"[prep] serializing system -> {out_dir}/system.xml")
    with open(out_dir / "system.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))
    with open(out_dir / "integrator.xml", "w") as f:
        f.write(XmlSerializer.serialize(integrator))
    with open(out_dir / "state.xml", "w") as f:
        f.write(XmlSerializer.serialize(state_eq))

    print(f"[prep] done. produce.py can now consume {out_dir}")


def _select_platform():
    try:
        return Platform.getPlatformByName("CUDA")
    except Exception:
        try:
            return Platform.getPlatformByName("OpenCL")
        except Exception:
            return Platform.getPlatformByName("Reference")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdb_in", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--padding-nm", type=float, default=1.0)
    p.add_argument("--ionic-strength", type=float, default=0.15)
    p.add_argument("--temperature", type=float, default=310.0)
    p.add_argument("--ph", type=float, default=7.4)
    p.add_argument("--dt-fs", type=float, default=4.0,
                   help="Integrator timestep in fs. 4 fs requires HMR=4 amu.")
    p.add_argument("--hmr-amu", type=float, default=4.0,
                   help="Hydrogen mass for HMR. 4 amu allows safe 4 fs integration.")
    p.add_argument("--com-restrain", default=None, metavar="CHAIN_A:CHAIN_B",
                   help="Apply a harmonic COM-COM distance restraint between "
                        "two input-PDB chains, around their equilibrated "
                        "distance. Compensates for the missing transmembrane "
                        "anchor on partial structures by preventing the two "
                        "chains from dissociating, without per-atom force "
                        "singularities. Example for Notch1 NRR: --com-restrain A:B "
                        "(restrains NEC<->NTM COM distance).")
    p.add_argument("--com-restrain-k", type=float, default=100.0,
                   help="Force constant for --com-restrain, kJ/mol/nm^2 "
                        "(default 100 = ~0.24 kcal/mol/A^2). At 310K, RMS "
                        "displacement around r0 is ~1.6 A.")
    p.add_argument("--max-internal-gap", type=int, default=10,
                   help="Skip de novo modeling of internal missing-residue gaps "
                        "longer than this. Long disordered loops at interfaces "
                        "clash with bound partners and blow up NVT (Notch1 NRR's "
                        "48-residue S1 cleavage gap in 3L95 is the canonical case).")
    args = p.parse_args()

    com_restrain = None
    if args.com_restrain:
        if ":" not in args.com_restrain:
            p.error(f"--com-restrain expects CHAIN_A:CHAIN_B, got {args.com_restrain!r}")
        a, b = args.com_restrain.split(":", 1)
        com_restrain = (a, b)

    prepare_system(args.pdb_in, args.out_dir,
                   padding_nm=args.padding_nm,
                   ionic_strength_M=args.ionic_strength,
                   temperature_K=args.temperature,
                   ph=args.ph,
                   dt_fs=args.dt_fs,
                   hmr_amu=args.hmr_amu,
                   com_restrain=com_restrain,
                   com_restrain_k=args.com_restrain_k,
                   max_internal_gap=args.max_internal_gap)


if __name__ == "__main__":
    main()
