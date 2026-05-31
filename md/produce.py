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

from openmm import app, unit, CustomCentroidBondForce, Platform, XmlSerializer


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


def _apply_com_distance_restraint(system, topology, positions,
                                    chain_a_index: int, chain_b_index: int,
                                    k_kj_per_nm2: float = 100.0):
    """Re-apply CustomCentroidBondForce restraint on |COM(g1) - COM(g2)|
    using the current loaded positions to fix r0. Identifies the two
    chains by INDEX in topology iteration order (robust to PDBFixer
    renames between prep and produce)."""
    chains = list(topology.chains())
    a_idx = [a.index for a in chains[chain_a_index].atoms() if a.name == "CA"]
    b_idx = [a.index for a in chains[chain_b_index].atoms() if a.name == "CA"]
    if not a_idx or not b_idx:
        raise ValueError(f"no CAs at chain indices {chain_a_index}/{chain_b_index}")
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

    # If prep saved an anchor_specs.json, re-apply the COM-distance
    # restraint fresh using positions from state.xml. r0 is recomputed
    # from these exact positions so the initial restraint force is 0.
    anchor_specs_path = prepared_dir / "anchor_specs.json"
    if anchor_specs_path.exists():
        import json
        spec = json.loads(anchor_specs_path.read_text())
        if spec.get("type") != "com_distance":
            raise ValueError(f"unknown anchor spec type: {spec.get('type')!r}")
        with open(prepared_dir / "state.xml") as f:
            _state_for_anchor = XmlSerializer.deserialize(f.read())
        r0, na, nb = _apply_com_distance_restraint(
            system, pdb.topology, _state_for_anchor.getPositions(),
            int(spec["chain_a_index"]), int(spec["chain_b_index"]),
            k_kj_per_nm2=float(spec.get("k_kj_per_nm2", 100.0)))
        print(f"[produce] re-applied COM-distance restraint: "
              f"chain idx {spec['chain_a_index']}({na} CAs) <-> "
              f"chain idx {spec['chain_b_index']}({nb} CAs), "
              f"r0={r0*10:.2f} A, k={spec.get('k_kj_per_nm2', 100.0)} kJ/mol/nm^2")

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
