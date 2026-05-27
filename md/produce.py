"""Production MD off a prep.py-prepared system.

Pipeline:
  - load system.xml + state.xml
  - run for N steps at the chosen timestep
  - write a DCD trajectory + a periodic checkpoint

Designed to be invoked once per replica:

  python produce.py prepared_3i08/  --steps 50000000  --replica 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from openmm import app, unit, Platform, XmlSerializer


def produce(prepared_dir: Path, replica: int, steps: int = 50_000_000,
            report_interval: int = 5000, dcd_interval: int = 5000):
    prepared_dir = Path(prepared_dir)
    out_dir = prepared_dir / f"replica_{replica}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[produce] loading prepared system from {prepared_dir}")
    with open(prepared_dir / "system.xml") as f:
        system = XmlSerializer.deserialize(f.read())
    with open(prepared_dir / "integrator.xml") as f:
        integrator = XmlSerializer.deserialize(f.read())

    pdb = app.PDBFile(str(prepared_dir / "equilibrated.pdb"))
    platform = _select_platform()
    simulation = app.Simulation(pdb.topology, system, integrator, platform)

    chk_path = out_dir / "checkpoint.chk"
    if chk_path.exists():
        print(f"[produce] resuming from {chk_path}")
        simulation.loadCheckpoint(str(chk_path))
        already = simulation.currentStep
        steps_remaining = max(0, steps - already)
        # DCD must be appended to, not overwritten.
        dcd_mode = "a"
        print(f"[produce] resumed at step {already}; {steps_remaining} steps remain")
    else:
        with open(prepared_dir / "state.xml") as f:
            state = XmlSerializer.deserialize(f.read())
        try:
            integrator.setRandomNumberSeed(int(time.time()) + replica * 7919)
        except Exception:
            pass
        simulation.context.setState(state)
        steps_remaining = steps
        dcd_mode = "w"
        print(f"[produce] fresh run; {steps_remaining} steps total")

    dcd_path = out_dir / "traj.dcd"
    log_path = out_dir / "log.csv"

    simulation.reporters.append(app.DCDReporter(str(dcd_path), dcd_interval, append=(dcd_mode == "a")))
    simulation.reporters.append(app.CheckpointReporter(str(chk_path), report_interval * 10))
    simulation.reporters.append(app.StateDataReporter(
        str(log_path), report_interval,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, density=True, progress=True, totalSteps=steps,
        elapsedTime=True, speed=True, remainingTime=True, append=(dcd_mode == "a"),
    ))
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, report_interval * 10,
        step=True, time=True, temperature=True, density=True, speed=True,
        remainingTime=True, totalSteps=steps,
    ))

    if steps_remaining > 0:
        print(f"[produce] running {steps_remaining} steps -> {dcd_path}")
        simulation.step(steps_remaining)
    print("[produce] done")


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
    p.add_argument("prepared_dir", type=Path)
    p.add_argument("--replica", type=int, default=0)
    p.add_argument("--steps", type=int, default=50_000_000, help="default = 100 ns at 2 fs")
    p.add_argument("--report-interval", type=int, default=5000)
    p.add_argument("--dcd-interval", type=int, default=5000)
    args = p.parse_args()
    produce(args.prepared_dir, args.replica, args.steps,
            args.report_interval, args.dcd_interval)


if __name__ == "__main__":
    main()
