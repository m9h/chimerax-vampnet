"""Membrane protein prep for chimerax-vampnet v0.8 (β2AR pilot).

Mirrors `md/prep.py` (soluble-protein pipeline) but inserts the
protein into a POPC bilayer via OpenMM `Modeller.addMembrane` and
uses `MonteCarloMembraneBarostat` for semi-isotropic pressure
coupling during NPT equilibration.

Pipeline:
  1. PDBFixer cleanup (missing residues / atoms / hydrogens; drop
     internal gaps > max_internal_gap to avoid modeling long
     disordered loops into membrane).
  2. Insert into POPC bilayer with TIP3P solvent + 0.15 M NaCl
     above/below the bilayer (Modeller.addMembrane handles all of
     this in one call).
  3. Amber14 force field + Amber14/lipid17 for POPC (the
     amber14/lipid17.xml file ships with OpenMM).
  4. Energy minimization.
  5. NVT equilibration with Cα restraints (500 ps) — restraints
     keep the protein from drifting during initial lipid pack-down.
  6. NPT equilibration with semi-isotropic barostat (500 ps).
  7. Release Cα restraints + final brief NVT (200 ps) — let side
     chains relax.

Output identical layout to `prep.py`: system.xml + integrator.xml +
state.xml + equilibrated.pdb + (no anchor_specs.json since no
COM restraint is meaningful for a monomeric membrane protein).

  python prep_membrane.py 2rh1.pdb prepared_2rh1_v0.8/

Designed for v0.8 W1. Defaults match Notch1/Hsp90 prep where
sensible (310 K, 4 fs HMR, Amber14, 0.15 M NaCl).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openmm import (app, unit, LangevinIntegrator,
                     MonteCarloMembraneBarostat, Platform, XmlSerializer,
                     CustomExternalForce, Vec3)
from pdbfixer import PDBFixer


def _select_platform():
    try:
        return Platform.getPlatformByName("CUDA")
    except Exception:
        try:
            return Platform.getPlatformByName("OpenCL")
        except Exception:
            return Platform.getPlatformByName("Reference")


def _add_ca_restraint(system, topology, positions, k_kj_per_nm2=1000.0):
    """Per-Cα harmonic restraint to its initial position; used during
    membrane pack-down to keep the protein anchored while POPC
    relaxes around it. Released before production."""
    force = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    n_ca = 0
    for atom in topology.atoms():
        if atom.name == "CA":
            pos = positions[atom.index]
            try:
                p = pos.value_in_unit(unit.nanometer)
            except AttributeError:
                p = pos
            force.addParticle(atom.index, [
                k_kj_per_nm2 * unit.kilojoule_per_mole / unit.nanometer ** 2,
                p[0] * unit.nanometer if not isinstance(p, Vec3) else p.x * unit.nanometer,
                p[1] * unit.nanometer if not isinstance(p, Vec3) else p.y * unit.nanometer,
                p[2] * unit.nanometer if not isinstance(p, Vec3) else p.z * unit.nanometer,
            ])
            n_ca += 1
    force_idx = system.addForce(force)
    return force_idx, n_ca


def prepare_membrane_system(pdb_in: Path, out_dir: Path,
                              ionic_strength_M: float = 0.15,
                              temperature_K: float = 310.0,
                              ph: float = 7.4, ff: str = "amber14",
                              dt_fs: float = 4.0, hmr_amu: float = 4.0,
                              lipid_type: str = "POPC",
                              z_padding_nm: float = 1.5,
                              xy_padding_nm: float = 0.5,
                              max_internal_gap: int = 10,
                              ca_restraint_k: float = 1000.0):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep_mem] fixing {pdb_in} -> {out_dir}/fixed.pdb")
    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    keys = list(fixer.missingResidues.keys())
    for key in keys:
        chain = list(fixer.topology.chains())[key[0]]
        n_res = sum(1 for _ in chain.residues())
        gap_len = len(fixer.missingResidues[key])
        if key[1] == 0 or key[1] == n_res:
            del fixer.missingResidues[key]
        elif gap_len > max_internal_gap:
            print(f"[prep_mem]   dropping internal {gap_len}-residue gap in "
                  f"chain {chain.id} at insertion {key[1]}; leave as break")
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(out_dir / "fixed.pdb", "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

    print(f"[prep_mem] loading force field ({ff}; lipids in amber14-all)")
    if ff == "amber14":
        # amber14-all.xml already bundles lipid17; loading
        # amber14/lipid17.xml separately raises "DLPC template already
        # exists" (duplicate registration). Just amber14-all.xml +
        # TIP3P for the surrounding water is enough.
        forcefield = app.ForceField(
            "amber14-all.xml",
            "amber14/tip3p.xml",
        )
    else:
        raise ValueError(f"unknown forcefield: {ff}")

    print(f"[prep_mem] embedding in {lipid_type} bilayer + TIP3P + "
          f"{ionic_strength_M} M NaCl (z_pad {z_padding_nm} nm, "
          f"xy_pad {xy_padding_nm} nm)")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    # OpenMM's Modeller.addMembrane aligns the protein along the bilayer
    # normal (z by default), inserts the chosen lipid type, fills xy
    # padding, and adds TIP3P + ions above/below. Single call does
    # everything that would otherwise need CHARMM-GUI.
    modeller.addMembrane(
        forcefield,
        lipidType=lipid_type,
        membraneCenterZ=0 * unit.nanometer,
        minimumPadding=xy_padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_M * unit.molar,
        neutralize=True,
    )
    with open(out_dir / "solvated.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    print("[prep_mem] creating OpenMM system")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=hmr_amu * unit.amu,
    )

    # Add Cα restraints for the pack-down phase.
    print(f"[prep_mem] applying Cα restraints (k = {ca_restraint_k} kJ/mol/nm²)")
    restraint_force_idx, n_ca = _add_ca_restraint(
        system, modeller.topology, modeller.positions,
        k_kj_per_nm2=ca_restraint_k,
    )
    print(f"[prep_mem]   restrained {n_ca} Cα atoms")

    print("[prep_mem] energy minimization")
    # Stage 1: dt=1 fs initial integrator for the brittle pack-down
    # phase. Membrane systems freshly inserted into POPC have large
    # lipid-protein clashes that even thorough minimization can't fully
    # quench; running at the production dt=4 fs immediately NaNs. The
    # standard cure is a graded timestep: 100 ps at 1 fs → 500 ps at
    # 2 fs → 200 ps at the production dt.
    integrator = LangevinIntegrator(temperature_K * unit.kelvin,
                                       1.0 / unit.picosecond,
                                       1.0 * unit.femtosecond)
    platform = _select_platform()
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=10000)

    nvt1_steps = 100 * 1000  # 100 ps at 1 fs
    print(f"[prep_mem] Stage 1 NVT with Cα restraints "
          f"(100 ps, dt=1 fs, {nvt1_steps} steps) — gentle lipid pack-down")
    simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)
    simulation.step(nvt1_steps)

    # Stage 2: dt=2 fs NPT with the membrane barostat.
    print(f"[prep_mem] Stage 2 NPT @ dt=2 fs, "
          f"MonteCarloMembraneBarostat (500 ps, semi-isotropic)")
    integrator_2 = LangevinIntegrator(temperature_K * unit.kelvin,
                                        1.0 / unit.picosecond,
                                        2.0 * unit.femtosecond)
    barostat = MonteCarloMembraneBarostat(
        1.0 * unit.atmosphere,
        0.0 * unit.bar * unit.nanometer,
        temperature_K * unit.kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
    )
    barostat_idx = system.addForce(barostat)
    state_after_s1 = simulation.context.getState(
        getPositions=True, getVelocities=True,
    )
    simulation = app.Simulation(modeller.topology, system, integrator_2, platform)
    simulation.context.setState(state_after_s1)
    npt_steps_2fs = int(500.0 / 2.0 * 1000)  # 500 ps at 2 fs
    simulation.step(npt_steps_2fs)

    # Stage 3: dt=4 fs (production timestep) NVT to relax side chains
    # after restraints are released. Drop the barostat (production
    # is NVT to match the v0.3 Notch1 protocol) and the restraints.
    print(f"[prep_mem] Stage 3 NVT @ dt={dt_fs} fs without restraints "
          f"(200 ps) — side-chain relaxation at production timestep")
    # Drop barostat (whose index may have shifted).
    barostat_idx_now = next(
        i for i, f in enumerate(system.getForces())
        if isinstance(f, MonteCarloMembraneBarostat)
    )
    system.removeForce(barostat_idx_now)
    # Drop Cα restraints (index may have shifted too).
    restraint_idx_now = next(
        (i for i, f in enumerate(system.getForces())
         if isinstance(f, CustomExternalForce)),
        None,
    )
    if restraint_idx_now is not None:
        system.removeForce(restraint_idx_now)
    integrator_3 = LangevinIntegrator(temperature_K * unit.kelvin,
                                        1.0 / unit.picosecond,
                                        dt_fs * unit.femtosecond)
    state_after_s2 = simulation.context.getState(
        getPositions=True, getVelocities=True,
    )
    simulation = app.Simulation(modeller.topology, system, integrator_3, platform)
    simulation.context.setState(state_after_s2)
    relax_steps = int(200.0 / dt_fs * 1000)
    simulation.step(relax_steps)
    integrator = integrator_3  # serialize the production-timestep one

    state_eq = simulation.context.getState(
        getPositions=True, getVelocities=True,
    )

    print(f"[prep_mem] writing equilibrated coordinates -> {out_dir}/equilibrated.pdb")
    with open(out_dir / "equilibrated.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, state_eq.getPositions(), f)

    print(f"[prep_mem] serializing system -> {out_dir}/system.xml")
    with open(out_dir / "system.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))
    with open(out_dir / "integrator.xml", "w") as f:
        f.write(XmlSerializer.serialize(integrator))
    with open(out_dir / "state.xml", "w") as f:
        f.write(XmlSerializer.serialize(state_eq))

    print(f"[prep_mem] done. produce.py can now consume {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdb_in", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--ionic-strength", type=float, default=0.15)
    p.add_argument("--temperature", type=float, default=310.0)
    p.add_argument("--ph", type=float, default=7.4)
    p.add_argument("--dt-fs", type=float, default=4.0)
    p.add_argument("--hmr-amu", type=float, default=4.0)
    p.add_argument("--lipid-type", type=str, default="POPC")
    p.add_argument("--z-padding-nm", type=float, default=1.5)
    p.add_argument("--xy-padding-nm", type=float, default=0.5)
    p.add_argument("--max-internal-gap", type=int, default=10)
    p.add_argument("--ca-restraint-k", type=float, default=1000.0)
    args = p.parse_args()
    prepare_membrane_system(
        args.pdb_in, args.out_dir,
        ionic_strength_M=args.ionic_strength,
        temperature_K=args.temperature, ph=args.ph,
        dt_fs=args.dt_fs, hmr_amu=args.hmr_amu,
        lipid_type=args.lipid_type,
        z_padding_nm=args.z_padding_nm,
        xy_padding_nm=args.xy_padding_nm,
        max_internal_gap=args.max_internal_gap,
        ca_restraint_k=args.ca_restraint_k,
    )


if __name__ == "__main__":
    main()
