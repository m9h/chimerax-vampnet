"""Well-tempered metadynamics on the Notch1 NEC--NTM COM distance.

The H1/H2 collective variable (CV) that defines our auto-inhibited
state is the COM distance between NEC (chain A, residues 1-174)
and NTM (chain B, residues 1-60). v0.3 confirmed that the 100 ns
MD horizon is too short to recover the pre-registered H2
magnitudes (apo >= 50% auto-inhibited, holo <= 30%); v0.5
confirmed the bottleneck is sampling time, not the v0.3 COM
restraint protocol.

Metadynamics on this CV is the natural cheap alternative to
brute-force ~us MD: PLUMED deposits Gaussian bias kernels along
the CV at periodic intervals, flattening the underlying free
energy surface (FES) and forcing the walker to sample regions
that would otherwise have prohibitive Boltzmann weight. Well-
tempered metadynamics (Barducci 2008) gradually decreases the
deposition height so the bias converges to a finite estimate
of the *unbiased* FES.

For a 234-CA system in TIP3P at 310 K with 4 fs HMR timestep,
a single A100-80GB walker runs ~10 ns/h. Five walkers x 100 ns
each = $25 in Modal compute and gives a converged 1D FES along
the NEC-NTM COM distance -- the exact axis our H1/H2 hypotheses
are framed in.

Self-contained environment recipe -- each Modal adapter in md/
builds its own image rather than sharing a base, so dep
collisions stay isolated to the tool that needs them.

  Base image:   nvidia/cuda:12.6.3-runtime-ubuntu24.04 (same as
                modal_md.py, so the prepared/<system>/ artifacts
                produced by `modal run modal_md.py::prep` are
                directly reusable)
  Env manager:  micromamba creating /opt/conda/envs/md
  Pip recipe:   openmm 8.5.1 + openmm-plumed (conda-forge) so the
                CUDA platform reuses the same wheels as modal_md.py
  Volume:       chimerax-vampnet-md (shared with modal_md.py;
                reads prepared/<system>/ artifacts written by the
                same prep step)
  GPU pin:      A100-80GB
  Tested:       2026-06-03 -- pending smoke

Usage:
  modal run md/notch1_metad_modal.py::run --system notch1_apo_v3 \\
      --walker 0 --ns 5      # smoke run
  modal run md/notch1_metad_modal.py::fanout --system notch1_apo_v3 \\
      --walkers 5 --ns 100   # production
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-metad"
VOLUME_NAME = "chimerax-vampnet-md"

# Reuse modal_md.py's CUDA + micromamba base so prepared/ artifacts
# are directly compatible. The only addition is openmm-plumed.
image = (
    # CUDA 11.8 base because openmm-plumed 2.0 (the latest version
    # compatible with CUDA-12 conda-forge openmm) was actually built
    # against CUDA 11. The A100 driver is backward-compatible with
    # CUDA 11 PTX, so a CUDA-11 stack runs fine on Modal A100-80GB.
    # openmm-plumed 2.1 requires CUDA 13 which the A100 driver does
    # not yet support, hence the 11-vs-13 pin pinch.
    modal.Image.from_registry("nvidia/cuda:11.8.0-runtime-ubuntu22.04",
                                add_python="3.12")
    .apt_install("bzip2", "ca-certificates", "curl")
    .run_commands(
        "mkdir -p /opt/conda/bin",
        "curl -sLo /tmp/mm.tar.bz2 "
        "https://micro.mamba.pm/api/micromamba/linux-64/latest",
        "tar -xvjf /tmp/mm.tar.bz2 -C /opt/conda bin/micromamba",
        "rm /tmp/mm.tar.bz2",
    )
    .env({"PATH": "/opt/conda/bin:/opt/conda/envs/md/bin:/usr/bin",
          "MAMBA_ROOT_PREFIX": "/opt/conda"})
    .run_commands(
        # openmm + openmm-plumed in the same env so PlumedForce is
        # importable from the same Python that loads system.xml.
        # cuda-version pin removed: openmm-plumed 2.1 requires cuda-12
        # *13* on conda-forge as of 2026-06, so we let conda pick the
        # matching openmm + cuda-toolkit set instead of forcing 12.9
        # (the Modal base image's runtime CUDA is 12.6 but PlumedForce
        # only needs the CUDA toolkit and the openmm wheel's runtime).
        # openmm-plumed 2.0 was built against CUDA 11. A100 drivers
        # are backward-compatible so CUDA-11 PTX runs fine on Modal.
        "/opt/conda/bin/micromamba create -y -n md -c conda-forge "
        "python=3.12 'openmm=8.1' 'openmm-plumed=2.0' "
        "openmmtools pdbfixer mdtraj numpy scipy "
        "'cudatoolkit>=11.2,<12' "
        "&& /opt/conda/bin/micromamba clean -a -y",
        "/opt/conda/envs/md/bin/python -c "
        "'from openmmplumed import PlumedForce; print(\"openmm-plumed OK\")'",
    )
    .add_local_dir(str(Path(__file__).parent), "/workspace", copy=True)
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_MOUNT = "/vol"
PYTHON = "/opt/conda/envs/md/bin/python"


def _plumed_script(nec_ca_indices: list[int], ntm_ca_indices: list[int],
                    height_kjmol: float, sigma_nm: float, pace: int,
                    biasfactor: float, temperature_K: float,
                    colvar_stride: int, hills_stride: int) -> str:
    """Compose a PLUMED input file biasing the NEC-NTM COM distance.

    Atom indices are 1-based per PLUMED convention; the caller passes
    0-based indices and we convert.
    """
    nec_str = ",".join(str(i + 1) for i in nec_ca_indices)
    ntm_str = ",".join(str(i + 1) for i in ntm_ca_indices)
    return f"""
nec: COM ATOMS={nec_str}
ntm: COM ATOMS={ntm_str}
d: DISTANCE ATOMS=nec,ntm

metad: METAD ARG=d ...
  PACE={pace}
  HEIGHT={height_kjmol}
  SIGMA={sigma_nm}
  BIASFACTOR={biasfactor}
  TEMP={temperature_K}
  FILE=HILLS
  GRID_MIN=0.0
  GRID_MAX=8.0
  GRID_BIN=400
...

PRINT ARG=d,metad.bias FILE=COLVAR STRIDE={colvar_stride}
FLUSH STRIDE={hills_stride}
""".strip()


@app.function(gpu="A100-80GB", timeout=4 * 3600,
               volumes={VOL_MOUNT: vol})
def run_remote(system: str, walker: int, ns: float = 100.0,
                dt_fs: float = 4.0,
                height_kjmol: float = 1.2,
                sigma_nm: float = 0.05,
                pace_ps: float = 1.0,
                biasfactor: float = 10.0,
                temperature_K: float = 310.0,
                colvar_stride_ps: float = 1.0,
                chain_nec: str = "A", chain_ntm: str = "B"):
    """Run one metadynamics walker on the prepared system at
    prepared/<system>/. Writes results to
    prepared/<system>/metad_walker_<walker>/.

    The PLUMED input uses well-tempered metadynamics on the COM
    distance between chain_nec and chain_ntm CAs. Default kernel:
    HEIGHT=1.2 kJ/mol, SIGMA=0.05 nm, PACE=1 ps, BIASFACTOR=10
    (Barducci 2008 defaults at 310 K for a 1-nm-scale CV).
    """
    import subprocess
    import sys
    out_dir = Path(VOL_MOUNT) / "prepared" / system
    walker_dir = out_dir / f"metad_walker_{walker}"
    walker_dir.mkdir(parents=True, exist_ok=True)

    # Build the PLUMED input from the equilibrated PDB's atom indexing.
    import mdtraj as md
    eq = md.load_pdb(str(out_dir / "equilibrated.pdb"))
    nec_ca = [a.index for a in eq.topology.atoms
               if a.name == "CA" and a.residue.chain.chain_id == chain_nec]
    ntm_ca = [a.index for a in eq.topology.atoms
               if a.name == "CA" and a.residue.chain.chain_id == chain_ntm]
    if not nec_ca or not ntm_ca:
        raise RuntimeError(
            f"could not find CA atoms on chains {chain_nec} / {chain_ntm}; "
            f"available chains: {sorted({c.chain_id for c in eq.topology.chains})}"
        )

    steps_per_ps = 1000.0 / dt_fs
    pace_steps = int(pace_ps * steps_per_ps)
    colvar_stride_steps = int(colvar_stride_ps * steps_per_ps)
    plumed_text = _plumed_script(
        nec_ca, ntm_ca,
        height_kjmol=height_kjmol, sigma_nm=sigma_nm, pace=pace_steps,
        biasfactor=biasfactor, temperature_K=temperature_K,
        colvar_stride=colvar_stride_steps, hills_stride=pace_steps,
    )
    plumed_path = walker_dir / "plumed.dat"
    plumed_path.write_text(plumed_text)
    print(f"[metad] wrote {plumed_path} (NEC={len(nec_ca)} CAs, "
          f"NTM={len(ntm_ca)} CAs)")

    steps = int(ns * 1000 * steps_per_ps)
    cmd = [PYTHON, "/workspace/produce_metad.py", str(out_dir),
           "--walker", str(walker),
           "--steps", str(steps),
           "--plumed", str(plumed_path),
           "--report-interval", str(max(1000, colvar_stride_steps * 10)),
           "--dcd-interval", str(colvar_stride_steps * 5)]
    print(f"[modal.metad] {' '.join(cmd)}")
    sys.stdout.flush()
    subprocess.run(cmd, check=True)
    vol.commit()
    return {"system": system, "walker": walker, "ns": ns,
            "out_dir": str(walker_dir)}


@app.local_entrypoint()
def run(system: str, walker: int = 0, ns: float = 5.0,
         dt_fs: float = 4.0, sigma_nm: float = 0.05,
         height_kjmol: float = 1.2, biasfactor: float = 10.0,
         pace_ps: float = 1.0, temperature: float = 310.0):
    """Synchronously run one walker. Use ns=5 for smoke, 100 for production."""
    r = run_remote.remote(system, walker, ns=ns, dt_fs=dt_fs,
                            sigma_nm=sigma_nm, height_kjmol=height_kjmol,
                            biasfactor=biasfactor, pace_ps=pace_ps,
                            temperature_K=temperature)
    print(f"[local] metad walker done: {r}")


@app.local_entrypoint()
def fanout(system: str, walkers: int = 5, ns: float = 100.0,
            dt_fs: float = 4.0, sigma_nm: float = 0.05,
            height_kjmol: float = 1.2, biasfactor: float = 10.0,
            pace_ps: float = 1.0, temperature: float = 310.0,
            start_walker: int = 0):
    """Launch N independent metad walkers in parallel via spawn().

    Each walker writes its own HILLS file under
    prepared/<system>/metad_walker_<i>/. Reweight and merge offline
    via md/metad_fes_postprocess.py (to be written when results land).
    """
    handles = []
    for w in range(start_walker, start_walker + walkers):
        h = run_remote.spawn(system, w, ns=ns, dt_fs=dt_fs,
                              sigma_nm=sigma_nm, height_kjmol=height_kjmol,
                              biasfactor=biasfactor, pace_ps=pace_ps,
                              temperature_K=temperature)
        print(f"[local] spawned walker {w}  fc_id={h.object_id}")
        handles.append((w, h))
    print(f"[local] {len(handles)} metad walkers in flight. "
          f"Use `modal app logs <app_id>` to track or `modal volume ls "
          f"{VOLUME_NAME} prepared/{system}/` to see HILLS files land.")
