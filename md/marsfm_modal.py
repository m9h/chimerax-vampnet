"""MarS-FM single-protein inference adapter on Modal.

Wraps Kapusniak et al. 2025 (arXiv:2509.24779; github.com/valence-labs/mars-fm)
for inference on a single user-supplied PDB. Their `scripts/generate.py`
is designed for bulk inference over an MD-CATH split; this adapter:

  1. Pulls the HuggingFace checkpoint `valencelabs/mars-fm` (MD-CATH 450
     weights) into the Modal image cache.
  2. Takes a local equilibrated PDB (our chimerax-vampnet prep output).
  3. Extracts the sequence + builds the atom14 numpy array MarS-FM
     expects under data_dir.
  4. Writes a one-row CSV split with the protein's name + seqres.
  5. Invokes `python -m scripts.generate ...` with --calls_mars and
     --max_mars_samples to control ensemble size.
  6. Returns the generated PDB + XTC as bytes; locally we save a
     compressed npz that `vampnet load_ensemble ... source marsfm`
     consumes directly.

  modal run md/marsfm_modal.py::sample --pdb md/3i08_apo.pdb \\
      --name notch1_apo_3i08 --n-samples 2000 --out marsfm_apo_3i08.npz

Per-protein cost on H100: ~30 sec for 500 conformations of a
160-residue protein per the MarS-FM paper. Notch1 NRR apo at 234 CAs
expected to be ~50-90 sec / 500 samples. 2000 samples ~= $0.20.

This adapter is the v0.4 acceleration target promised in the paper
roadmap. It does NOT replace classical MD as the H2 reference --- it
provides the missing magnitude estimator that 100 ns MD cannot reach.

For NRR-containing holo complexes (NRR + Fab), MarS-FM is being asked
to extrapolate beyond MD-CATH's single-domain training distribution.
Treat holo MarS-FM output as exploratory; the apo arm is in-distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-marsfm"
HF_REPO = "valencelabs/mars-fm"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        "numpy<2",
        "scipy",
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "pandas",
        "einops",
        "tqdm",
        "ml-collections",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        # Clone the MarS-FM repo and install its requirements.
        "cd /opt && git clone https://github.com/valence-labs/mars-fm.git",
        # Their conda env file requires conda, so we install via pip from
        # the requirements they list (best-effort; the conda env may have
        # exact pins not reflected here -- adjust if inference fails).
        "cd /opt/mars-fm && (pip install -e . --no-deps || true)",
    )
)

app = modal.App(APP_NAME, image=image)


# Map of 20-letter amino acid codes used by the standard atom14 ordering.
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _pdb_to_atom14_and_seqres(pdb_bytes: bytes, chain_id: str | None = None):
    """Parse a PDB into the atom14 representation MarS-FM expects.

    Returns (atom14 ndarray shape (n_residues, 14, 3), seqres str). If
    chain_id is None, takes the first protein chain only (MarS-FM is a
    single-chain model)."""
    import io
    import numpy as np
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    # atom14 ordering used by AlphaFold (and inherited by MarS-FM via
    # OpenFold-style encoders). Index 0-13 within each residue.
    ATOM14_ORDER = {
        "ALA": ["N", "CA", "C", "O", "CB"],
        "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "CYS": ["N", "CA", "C", "O", "CB", "SG"],
        "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "GLY": ["N", "CA", "C", "O"],
        "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
        "SER": ["N", "CA", "C", "O", "CB", "OG"],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    }

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", io.BytesIO(pdb_bytes))
    model = next(structure.get_models())

    if chain_id is None:
        # First protein chain only.
        for chain in model:
            if any(is_aa(r, standard=True) for r in chain):
                target_chain = chain
                break
        else:
            raise ValueError("no protein chain found")
    else:
        target_chain = model[chain_id]

    residues = [r for r in target_chain if is_aa(r, standard=True)
                and r.get_resname() in THREE_TO_ONE]
    seqres = "".join(THREE_TO_ONE[r.get_resname()] for r in residues)

    atom14 = np.zeros((len(residues), 14, 3), dtype=np.float32)
    for i, res in enumerate(residues):
        names = ATOM14_ORDER.get(res.get_resname(), [])
        for j, name in enumerate(names):
            if name in res:
                atom14[i, j] = res[name].get_coord()
    return atom14, seqres


@app.function(gpu="H100", timeout=3600)
def sample_remote(pdb_bytes: bytes, name: str, n_samples: int = 2000,
                   chain_id: str | None = None):
    """Run MarS-FM inference on a single PDB. Returns the generated
    ensemble as numpy bytes."""
    import io
    import subprocess
    import sys
    import tempfile

    import numpy as np
    from huggingface_hub import snapshot_download

    print(f"[marsfm] downloading checkpoint {HF_REPO}")
    ckpt_dir = snapshot_download(repo_id=HF_REPO, cache_dir="/root/.cache/hf")
    # Pick the first .ckpt found in the snapshot.
    ckpts = list(Path(ckpt_dir).rglob("*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"no .ckpt in {ckpt_dir}")
    mars_ckpt = str(ckpts[0])
    print(f"[marsfm] checkpoint at {mars_ckpt}")

    print(f"[marsfm] parsing PDB ({len(pdb_bytes)} bytes), name={name}")
    atom14, seqres = _pdb_to_atom14_and_seqres(pdb_bytes, chain_id=chain_id)
    print(f"[marsfm] atom14 shape {atom14.shape}, seqres len {len(seqres)}")
    if len(seqres) > 500:
        print(f"[marsfm] WARNING: seqres len {len(seqres)} > 500 (MD-Cath training cap); "
              f"out-of-distribution behavior likely")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        data_dir = tmp / "data"
        data_dir.mkdir()
        splits_dir = tmp / "splits"
        splits_dir.mkdir()
        out_dir = tmp / "out"
        out_dir.mkdir()

        # Save atom14 input.
        np.save(data_dir / f"{name}.npy", atom14)

        # One-row CSV split.
        split_csv = splits_dir / "single.csv"
        split_csv.write_text(f"name,seqres\n{name},{seqres}\n")

        cmd = [
            sys.executable, "-m", "scripts.generate",
            "--mars_ckpt", mars_ckpt,
            "--data_dir", str(data_dir),
            "--split", str(split_csv),
            "--out_dir", str(out_dir),
            "--pdb_id", name,
            "--max_mars_samples", str(n_samples),
        ]
        print(f"[marsfm] {' '.join(cmd)}")
        sys.stdout.flush()
        subprocess.run(cmd, cwd="/opt/mars-fm", check=True)

        # MarS-FM writes <name>.pdb + <name>.xtc to out_dir. Load both
        # into a (N, n_atoms, 3) array and stash CA-only too for cheap
        # downstream consumption.
        import mdtraj as md
        traj = md.load(str(out_dir / f"{name}.xtc"),
                        top=str(out_dir / f"{name}.pdb"))
        coords_all = (traj.xyz * 10.0).astype(np.float32)  # nm -> A
        ca_indices = [a.index for a in traj.topology.atoms if a.name == "CA"]
        coords_ca = coords_all[:, ca_indices, :]
        print(f"[marsfm] generated {coords_all.shape[0]} frames; "
              f"all-atom shape {coords_all.shape}, CA-only {coords_ca.shape}")

        buf = io.BytesIO()
        np.savez_compressed(buf, coords=coords_all, coords_ca=coords_ca,
                            seqres=np.array(seqres))
        return buf.getvalue()


@app.local_entrypoint()
def sample(pdb: str, name: str = "protein", n_samples: int = 2000,
           chain_id: str = "", out: str = ""):
    """Generate a MarS-FM ensemble for the given PDB and save locally."""
    pdb_bytes = Path(pdb).read_bytes()
    chain = chain_id or None
    print(f"[local] uploading {len(pdb_bytes)} bytes for {name}; "
          f"requesting {n_samples} samples")
    data = sample_remote.remote(pdb_bytes, name, n_samples=n_samples,
                                  chain_id=chain)
    out_path = Path(out) if out else Path(f"{name}_marsfm.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load with: vampnet load_ensemble {name}_marsfm "
          f"{out_path} format marsfm")
