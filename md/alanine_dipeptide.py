"""Tier-1 validation: alanine dipeptide MD, ~50 ns implicit solvent.

The standard MSM hello-world. Should run in ~1 hr on a GB10. Uses
openmmtools' canonical AlanineDipeptideImplicit test system so we
inherit a clean Topology + Amber14 force field assignment with all
backbone bonds intact:

  python alanine_dipeptide.py out_ala/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openmm import app, unit, LangevinIntegrator, Platform


def run(out_dir: Path, ns: float = 50.0, temperature_K: float = 300.0):
    out_dir.mkdir(parents=True, exist_ok=True)

    # openmmtools ships the canonical alanine dipeptide as a ready-made
    # System+Topology+Positions triple. No PDB parsing headaches.
    from openmmtools.testsystems import AlanineDipeptideImplicit
    testsystem = AlanineDipeptideImplicit(constraints=app.HBonds)
    system = testsystem.system
    topology = testsystem.topology
    positions = testsystem.positions

    # Persist the starting structure for the bundle to load.
    with open(out_dir / "ala_dipeptide.pdb", "w") as f:
        app.PDBFile.writeFile(topology, positions, f)

    integrator = LangevinIntegrator(temperature_K * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)

    platform = _select_platform()
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)

    steps = int(ns * 500_000)  # 2 fs/step -> 500,000 steps per ns
    save_every = 1000  # 2 ps -> 25,000 frames in 50 ns
    sim.reporters.append(app.DCDReporter(str(out_dir / "traj.dcd"), save_every))
    sim.reporters.append(app.StateDataReporter(
        str(out_dir / "log.csv"), save_every,
        step=True, time=True, temperature=True, speed=True, totalSteps=steps,
    ))
    sim.reporters.append(app.StateDataReporter(
        sys.stdout, save_every * 50,
        step=True, time=True, temperature=True, speed=True, totalSteps=steps,
    ))
    print(f"[ala] running {ns} ns ({steps} steps) on {platform.getName()} -> {out_dir/'traj.dcd'}")
    sim.step(steps)
    print("[ala] done")


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
    p.add_argument("out_dir", type=Path)
    p.add_argument("--ns", type=float, default=50.0)
    p.add_argument("--temperature", type=float, default=300.0)
    args = p.parse_args()
    run(args.out_dir, ns=args.ns, temperature_K=args.temperature)


if __name__ == "__main__":
    main()
