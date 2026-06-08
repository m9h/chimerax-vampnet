"""UMA (Meta FAIR's Universal Models for Atoms) as an MD force field on Modal.

Paper: Wood et al. 2025, "UMA: A Family of Universal Models for Atoms"
(arXiv:2506.23971). Weights at https://huggingface.co/facebook/UMA. Code in
https://github.com/facebookresearch/fairchem (Apache 2.0). Presented at
Starkly Speaking 2025-06-30.

Why it earns a slot in the v0.9+ pipeline:

  UMA is the single largest-trained universal MLFF as of mid-2025: 500 M
  unique 3D structures spanning molecules, materials, and catalysts. The
  Mixture-of-Linear-Experts architecture keeps inference fast (a 1.4 B-param
  model uses ~50 M active params per structure). UMA is energy-conservative
  by construction, so it can drive *MD* and not just predict equilibrium
  properties — unlike all our other generative adapters.

  Direct role in this project:
    - Replace AMBER14SB+TIP3P in md/produce.py with a UMA OpenMM/ASE calculator.
      Initial benchmark: chignolin folding (where DESRES Anton is the gold
      standard, and we have a v0.1 1 μs reference trajectory).
    - Drive OM-TPS (see md/om_tps_modal.py) — UMA's score function is the
      paper's preferred backbone for action-minimisation TPS.
    - For membrane systems (β2AR W1, deferred at NVT), UMA may avoid the
      lipid-protein clash NaN failures we hit with classical force fields.

  Output schema differs from the generative sources: UMA produces an MD
  trajectory, not iid samples. The schema mirrors md/produce.py's DCD-to-npz
  conversion in md/extract_ca_modal.py:
    coords        : (n_frames, n_atoms, 3) Å
    coords_ca     : (n_frames, n_ca, 3) Å
    seqres        : str
    chain_id      : (n_ca,)
    dt_fs         : float
    save_every_fs : float
    energy        : (n_frames,) kJ/mol  (UMA-predicted)

  Base image:   modal.Image.debian_slim(python_version=3.11)
  Pip extras:   torch (cu124), fairchem-core, ase, biopython, mdtraj.
                (Earlier scaffold over-bundled mace-torch/nequip/openmm-ml;
                stripped after 2026-06-08 smoke 1 image-build failure —
                fairchem-core ships its own ASE-compatible calculator.)
  Checkpoint:   uma-s-1p2 (small, default) or uma-m-1p1 (medium) — pulled
                from HuggingFace on first call via fairchem.core's
                pretrained_mlip module.
  Task tag:     "omol" for molecules / proteins (vs "omat" materials, "oc20"
                catalysis). UMA is a single network conditioned on a task
                token at inference.
  GPU pin:      A100-80GB for production; H100 ideal if available.
  Tested:       2026-06-08 — scaffold rev 2 after first-invocation
                ImportError on OCPCalculator (deprecated in v2.x). Correct
                API per upstream README: pretrained_mlip.get_predict_unit
                + FAIRChemCalculator with task_name="omol".

  modal run md/uma_modal.py::produce \\
      --pdb md/3i08_apo.pdb --name notch1_apo_uma \\
      --ns 10.0 --temperature 310.0 \\
      --out notch1_apo_uma.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-uma"
HF_REPO = "facebook/UMA"

image = (
    # fairchem-core main pins torch~=2.8.0 + numpy>=2.0,<2.5 + ase>=3.26.0;
    # we follow upstream pins exactly to avoid resolver fights.
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        extra_index_url="https://data.pyg.org/whl/torch-2.8.0+cu126.html",
    )
    .pip_install(
        # Install from git main: PyPI 2.20.0's module layout doesn't expose
        # pretrained_mlip / FAIRChemCalculator at fairchem.core.calculate.
        # When the next PyPI release lands, pin a version instead.
        "fairchem-core @ git+https://github.com/facebookresearch/fairchem.git#subdirectory=packages/fairchem-core",
        "ase>=3.26.0",
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "gemmi",
    )
)

VOLUME_NAME = "chimerax-vampnet-md"
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_MOUNT = "/vol"

# facebook/UMA is a gated HF repo (one-click access request, then a personal
# token). Reuses the project-wide "huggingface-secret" Modal secret created
# 2026-05-22; the same secret unlocks any HF-gated adapter (AF3 etc.).
HF_SECRET = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="A100-80GB", timeout=24 * 3600, volumes={VOL_MOUNT: vol},
              secrets=[HF_SECRET])
def produce_remote(pdb_bytes: bytes, name: str, ns: float = 10.0,
                    temperature_K: float = 310.0, dt_fs: float = 0.5,
                    save_every_fs: float = 200.0,
                    uma_variant: str = "uma-s-1p2",
                    task_name: str = "omol") -> bytes:
    """Run UMA-driven MD on a prepped PDB.

    dt_fs default is 0.5 fs (NNP-MD usually needs sub-fs timesteps for
    energy stability; we drop from the 4 fs HMR-MD we use with AMBER).

    uma_variant: HuggingFace tag for fairchem.core.pretrained_mlip; per the
    2026-06 release these are "uma-s-1p2" (small) and "uma-m-1p1" (medium).
    task_name: "omol" for molecules/proteins, "omat" for materials, "oc20"
    for catalysis. UMA is task-conditioned at inference.
    """
    import io
    import os
    import time

    import numpy as np
    import torch
    from ase.io import read as ase_read
    from ase.md.langevin import Langevin
    from ase import units as ase_units
    # Deep import path — the top-level fairchem.core __init__ re-export
    # is post-2.20.0, but the .calculate submodule has these in older
    # versions too. Belt and suspenders.
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    pdb_path = Path("/tmp/system.pdb")
    pdb_path.write_bytes(pdb_bytes)
    atoms = ase_read(str(pdb_path))
    n_atoms = len(atoms)

    print(f"[uma] {name}: {n_atoms} atoms, ns={ns}, dt={dt_fs} fs, "
          f"T={temperature_K} K, variant={uma_variant}, task={task_name}")

    # Cache UMA weights on the persistent volume.
    cache_dir = Path(VOL_MOUNT) / "uma_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    seed = int(np.random.randint(0, np.iinfo(np.int32).max, dtype=np.int64))
    predictor = pretrained_mlip.get_predict_unit(uma_variant, device="cuda",
                                                  seed=seed)
    calc = FAIRChemCalculator(predictor, task_name=task_name)
    atoms.calc = calc

    n_steps = int(ns * 1000 * 1000 / dt_fs)   # ns * 1e6 fs / dt_fs
    save_stride = max(1, int(save_every_fs / dt_fs))
    n_save = n_steps // save_stride + 1
    print(f"[uma] {n_steps} integration steps, save every {save_stride} "
          f"({n_save} frames)")

    dyn = Langevin(atoms, timestep=dt_fs * ase_units.fs,
                    temperature_K=temperature_K, friction=0.01 / ase_units.fs)

    coords_traj = np.zeros((n_save, n_atoms, 3), dtype=np.float32)
    energies = np.zeros(n_save, dtype=np.float32)
    coords_traj[0] = atoms.get_positions()
    energies[0] = float(atoms.get_potential_energy())

    t0 = time.time()
    for k in range(1, n_save):
        dyn.run(save_stride)
        coords_traj[k] = atoms.get_positions()
        energies[k] = float(atoms.get_potential_energy())
        if k % max(1, n_save // 20) == 0:
            ps_done = k * save_stride * dt_fs / 1000.0
            ns_per_day = (ps_done * 86400) / max(1e-9, (time.time() - t0)) / 1000.0
            print(f"[uma] {k}/{n_save} saves ({ps_done:.2f} ps), "
                  f"E={energies[k]:.1f} kJ/mol, {ns_per_day:.2f} ns/day")

    # CA mask via ASE atom-type filter — UMA uses element symbols; CA is
    # carbon with the PDB "CA" name, so we re-read the PDB to get the
    # ATOM-record name list (ASE strips this).
    ca_indices = _ca_indices_from_pdb(pdb_path.read_text())
    ca_mask = np.zeros(n_atoms, dtype=bool)
    ca_mask[ca_indices] = True
    coords_ca = coords_traj[:, ca_mask, :]
    print(f"[uma] coords {coords_traj.shape}, coords_ca {coords_ca.shape}, "
          f"final E={energies[-1]:.1f} kJ/mol")

    seqres = _seqres_from_pdb(pdb_path.read_text())
    chain_id = np.zeros(int(ca_mask.sum()), dtype=np.int64)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=coords_traj,
        coords_ca=coords_ca,
        seqres=np.array(seqres),
        chain_id=chain_id,
        energy=energies,
        dt_fs=np.array(dt_fs, dtype=np.float32),
        save_every_fs=np.array(save_every_fs, dtype=np.float32),
        temperature_K=np.array(temperature_K, dtype=np.float32),
        uma_variant=np.array(uma_variant),
    )
    vol.commit()
    return buf.getvalue()


def _ca_indices_from_pdb(text: str):
    """Indices of CA atoms in ATOM-record order — matches ASE's read order
    for a PDB without HETATM/altloc complications."""
    idx = []
    counter = 0
    for line in text.splitlines():
        if line.startswith("ATOM"):
            if line[12:16].strip() == "CA":
                idx.append(counter)
            counter += 1
    return idx


def _seqres_from_pdb(text: str):
    """One-letter sequence from CA records — same hack as featurize.py."""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    out = []
    for line in text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res = line[17:20].strip()
            out.append(three_to_one.get(res, "X"))
    return "".join(out)


@app.function(gpu="A100-80GB", timeout=600, volumes={VOL_MOUNT: vol},
              secrets=[HF_SECRET])
def smoke_remote() -> dict:
    """End-to-end pipeline smoke using ase.build.molecule('H2O') — no PDB
    parsing involved. Exactly the README example, adapted to return a
    summary instead of leaving atoms in-memory. Use this to confirm an
    image rebuild hasn't broken the UMA → ASE → Langevin path."""
    import os
    import time

    import numpy as np
    from ase import units as ase_units
    from ase.build import molecule
    from ase.md.langevin import Langevin
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    cache_dir = Path(VOL_MOUNT) / "uma_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    seed = int(np.random.randint(0, np.iinfo(np.int32).max, dtype=np.int64))
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda",
                                                  seed=seed)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms = molecule("H2O")
    atoms.calc = calc

    e0 = float(atoms.get_potential_energy())
    dyn = Langevin(atoms, timestep=0.1 * ase_units.fs,
                    temperature_K=400, friction=0.001 / ase_units.fs)
    t0 = time.time()
    dyn.run(steps=1000)
    elapsed = time.time() - t0
    e1 = float(atoms.get_potential_energy())
    vol.commit()
    summary = {
        "ok": True,
        "n_atoms": int(len(atoms)),
        "e0_eV": e0, "e1_eV": e1,
        "elapsed_s": float(elapsed),
        "steps": 1000,
        "ns_per_day_equiv": 1000 * 0.1 / 1000 * 86400 / max(1e-9, elapsed) / 1000,
    }
    print(f"[uma-smoke] {summary}")
    return summary


@app.local_entrypoint()
def smoke():
    """Run the README-style H2O smoke; prints a JSON summary."""
    print("[local] launching UMA H2O smoke (no PDB parsing)")
    summary = smoke_remote.remote()
    print(f"[local] smoke result: {summary}")


@app.local_entrypoint()
def produce(pdb: str, name: str = "system", ns: float = 10.0,
            temperature: float = 310.0, dt_fs: float = 0.5,
            save_every_fs: float = 200.0, uma_variant: str = "uma-s-1p2",
            task_name: str = "omol", out: str = ""):
    """Run UMA-driven MD on a prepped PDB and dump CA + all-atom trajectories.

    Costs roughly 5-10x classical AMBER MD on the same GPU per integration
    step, but UMA's potential-energy surface is closer to DFT than to AMBER —
    relevant for ligand binding, metalloproteins, membrane-embedded systems
    where AMBER force fields struggle.
    """
    pdb_bytes = Path(pdb).read_bytes()
    print(f"[local] UMA-MD on {name}: {len(pdb_bytes)} bytes PDB, "
          f"{ns} ns @ {dt_fs} fs, variant {uma_variant}")
    data = produce_remote.remote(pdb_bytes, name,
                                  ns=ns, temperature_K=temperature,
                                  dt_fs=dt_fs, save_every_fs=save_every_fs,
                                  uma_variant=uma_variant,
                                  task_name=task_name)
    out_path = Path(out) if out else Path(f"{name}_uma.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load as 'md' source in vampnet for H3 ingestion: "
          f"vampnet load_ensemble {name}_uma {out_path} format md")
