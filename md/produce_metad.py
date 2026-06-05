"""Production MD with PLUMED well-tempered metadynamics bias.

Mirrors `produce.py` but:
  1. Skips the v0.3 COM-distance restraint (we are MEASURING the free
     energy along that CV, not constraining it; if anchor_specs.json
     is present we IGNORE it).
  2. Loads a PLUMED input from --plumed and attaches a PlumedForce.
     PLUMED handles deposition + bias accumulation + HILLS file
     writing; we only have to thread the file path through.

  python produce_metad.py prepared/<system>/ --walker 0 \\
      --steps 25_000_000 --plumed /path/to/plumed.dat \\
      --report-interval 5000 --dcd-interval 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from openmm import app, unit, Platform, XmlSerializer
from openmmplumed import PlumedForce


def produce_metad(prepared_dir: Path, walker: int, steps: int,
                   plumed_input: str, report_interval: int,
                   dcd_interval: int):
    prepared_dir = Path(prepared_dir)
    out_dir = prepared_dir / f"metad_walker_{walker}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[metad] loading prepared system from {prepared_dir}")
    with open(prepared_dir / "system.xml") as f:
        system = XmlSerializer.deserialize(f.read())
    with open(prepared_dir / "integrator.xml") as f:
        integrator = XmlSerializer.deserialize(f.read())

    pdb = app.PDBFile(str(prepared_dir / "equilibrated.pdb"))

    # PLUMED needs to write its HILLS / COLVAR files into the
    # walker's own output dir so multi-walker runs don't collide.
    # cd into out_dir before constructing Simulation so PLUMED's
    # relative output paths resolve there.
    import os
    os.chdir(out_dir)

    # PlumedForce takes the raw script text, not a file path.
    plumed_script = Path(plumed_input).read_text()
    system.addForce(PlumedForce(plumed_script))
    print(f"[metad] attached PlumedForce ({len(plumed_script)} chars)")

    platform = _select_platform()
    simulation = app.Simulation(pdb.topology, system, integrator, platform)

    chk_path = out_dir / "checkpoint.chk"
    # Stale-checkpoint guard (v0.7): a checkpoint.chk can exist on the
    # volume from a prior aborted run while the actual DCD does NOT
    # exist (or is empty), because Modal commits the volume on function
    # exit and OpenMM's CheckpointReporter fires on the first step but
    # DCDReporter only opens the file at the first DCD interval. If we
    # find a chk but no DCD, the chk is stale and a resume would crash
    # at DCDReporter __init__ with "struct.error: unpack requires a
    # buffer of 4 bytes" (the v0.6 holo walker-3 crash). Wipe it.
    dcd_check = out_dir / "traj.dcd"
    if chk_path.exists() and (not dcd_check.exists() or dcd_check.stat().st_size < 1024):
        print(f"[metad] stale checkpoint at {chk_path} (no DCD or empty); deleting")
        chk_path.unlink()
    if chk_path.exists():
        print(f"[metad] resuming from {chk_path}")
        simulation.loadCheckpoint(str(chk_path))
        already = simulation.currentStep
        steps_remaining = max(0, steps - already)
        dcd_mode = "a"
    else:
        with open(prepared_dir / "state.xml") as f:
            state = XmlSerializer.deserialize(f.read())
        try:
            integrator.setRandomNumberSeed(int(time.time()) + walker * 7919)
        except Exception:
            pass
        simulation.context.setState(state)
        steps_remaining = steps
        dcd_mode = "w"

    dcd_path = out_dir / "traj.dcd"
    log_path = out_dir / "log.csv"

    simulation.reporters.append(app.DCDReporter(
        str(dcd_path), dcd_interval, append=(dcd_mode == "a")))
    simulation.reporters.append(app.CheckpointReporter(
        str(chk_path), report_interval * 10))
    simulation.reporters.append(app.StateDataReporter(
        str(log_path), report_interval,
        step=True, time=True, potentialEnergy=True, temperature=True,
        progress=True, totalSteps=steps, elapsedTime=True, speed=True,
        remainingTime=True, append=(dcd_mode == "a"),
    ))
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, report_interval * 10,
        step=True, time=True, temperature=True, speed=True,
        remainingTime=True, totalSteps=steps,
    ))

    if steps_remaining > 0:
        print(f"[metad] running {steps_remaining} steps -> {dcd_path}")
        simulation.step(steps_remaining)
    print("[metad] done")


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
    p.add_argument("--walker", type=int, default=0)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--plumed", type=str, required=True,
                    help="path to PLUMED input file")
    p.add_argument("--report-interval", type=int, default=5000)
    p.add_argument("--dcd-interval", type=int, default=5000)
    args = p.parse_args()
    produce_metad(args.prepared_dir, args.walker, args.steps,
                   args.plumed, args.report_interval, args.dcd_interval)


if __name__ == "__main__":
    main()
