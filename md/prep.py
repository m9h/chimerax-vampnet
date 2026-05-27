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

from openmm import app, unit, LangevinIntegrator, MonteCarloBarostat, Platform, XmlSerializer
from openmm import Vec3
from pdbfixer import PDBFixer


def prepare_system(pdb_in: Path, out_dir: Path, padding_nm: float = 1.0,
                   ionic_strength_M: float = 0.15, temperature_K: float = 310.0,
                   ph: float = 7.4, ff: str = "amber14", dt_fs: float = 4.0,
                   hmr_amu: float = 4.0):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep] fixing {pdb_in} -> {out_dir}/fixed.pdb")
    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    # Drop terminal missing residues to avoid building artificial chain ends.
    keys = list(fixer.missingResidues.keys())
    for key in keys:
        chain = list(fixer.topology.chains())[key[0]]
        n_res = sum(1 for _ in chain.residues())
        if key[1] == 0 or key[1] == n_res:
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
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(eq_steps)

    print(f"[prep] writing equilibrated coordinates -> {out_dir}/equilibrated.pdb")
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(out_dir / "equilibrated.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), f)

    print(f"[prep] serializing system -> {out_dir}/system.xml")
    with open(out_dir / "system.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))
    with open(out_dir / "integrator.xml", "w") as f:
        f.write(XmlSerializer.serialize(integrator))
    with open(out_dir / "state.xml", "w") as f:
        f.write(XmlSerializer.serialize(state))

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
    args = p.parse_args()
    prepare_system(args.pdb_in, args.out_dir,
                   padding_nm=args.padding_nm,
                   ionic_strength_M=args.ionic_strength,
                   temperature_K=args.temperature,
                   ph=args.ph,
                   dt_fs=args.dt_fs,
                   hmr_amu=args.hmr_amu)


if __name__ == "__main__":
    main()
