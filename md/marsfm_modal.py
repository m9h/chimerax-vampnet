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
    # NGC PyTorch container has tuned CUDA + cuDNN + torch + NCCL +
    # apex; use that as the base for any GPU-PyTorch workload rather
    # than rolling our own torch wheel on debian.
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.04-py3",
                                add_python=None)
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        # From the MarS-FM conda environment (mars.yaml). NGC PyTorch
        # image already has torch, numpy, scipy, pandas, tqdm.
        "pytorch-lightning>=2.4",
        "torchdiffeq",
        "deeptime",
        "dm-tree",
        "fair-esm",
        "biopython",
        "mdtraj",
        "wandb",
        "matplotlib",
        "statsmodels",
        "huggingface_hub",
        "ml-collections",
        "einops",
    )
    .run_commands(
        # Clone upstream as a placeholder; we then overwrite the source
        # tree with our local fork (which contains the multi-chain
        # Phase-1 patch). The upstream clone is only kept so the dir
        # structure (e.g. /opt/mars-fm) is present for the flash_attn
        # sed to find files even if we later flip back to upstream-only.
        "cd /opt && git clone https://github.com/valence-labs/mars-fm.git",
        "cd /opt/mars-fm && (pip install -e . --no-deps || true)",
    )
    # Overlay our local mars-fm fork with the Phase-1 multi-chain patch
    # at /tmp/mars-fm-fork (see /home/mhough/.claude/plans/twinkly-
    # whistling-bonbon.md for the design). add_local_dir copies our
    # fork OVER the upstream clone, so the modified model.py and
    # generate.py take effect.
    .add_local_dir("/tmp/mars-fm-fork", "/opt/mars-fm", copy=True)
    .run_commands(
        # MarS-FM's vendored OpenFold uses the pre-v2.0 flash_attn API
        # (flash_attn_unpadded_kvpacked_func, renamed to
        # flash_attn_varlen_kvpacked_func in v2.0+). NGC PyTorch ships
        # newer flash_attn so we alias on import. (Re-apply on the
        # overlaid fork; the local fork doesn't include this sed.)
        "sed -i 's/flash_attn_unpadded_kvpacked_func/flash_attn_varlen_kvpacked_func/g' "
        "/opt/mars-fm/mars/vendored/openfold/primitives.py "
        "/opt/mars-fm/mars/vendored/openfold/ipa.py 2>/dev/null || true",
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


def _pdb_to_atom14_and_seqres_and_chainid(pdb_bytes: bytes, chain_id: str | None = None):
    """Same as _pdb_to_atom14_and_seqres but also returns a per-residue
    chain_id ndarray (0, 1, 2, ... in order of chains taken). Used for
    the Phase-1 multi-chain inference API."""
    atom14, seqres = _pdb_to_atom14_and_seqres(pdb_bytes, chain_id=chain_id)
    # Re-parse to assign chain ids per residue, matching the same chain
    # filtering as _pdb_to_atom14_and_seqres.
    import io
    import numpy as np
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", io.StringIO(pdb_bytes.decode("utf-8")))
    model = next(structure.get_models())
    cids = []
    if chain_id is None:
        # First protein chain only -> all-zeros.
        for chain in model:
            if any(is_aa(r, standard=True) for r in chain):
                for r in chain:
                    if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE:
                        cids.append(0)
                break
    elif chain_id.upper() == "ALL":
        cid = 0
        for chain in model:
            chain_had_any = False
            for r in chain:
                if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE:
                    cids.append(cid)
                    chain_had_any = True
            if chain_had_any:
                cid += 1
    elif "," in chain_id:
        wanted = [c.strip() for c in chain_id.split(",")]
        for cid, wid in enumerate(wanted):
            chain = model[wid]
            for r in chain:
                if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE:
                    cids.append(cid)
    else:
        target_chain = model[chain_id]
        for r in target_chain:
            if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE:
                cids.append(0)
    assert len(cids) == atom14.shape[0], \
        f"chain_id length {len(cids)} != atom14 length {atom14.shape[0]}"
    return atom14, seqres, np.array(cids, dtype=np.int64)


def _pdb_to_atom14_and_seqres(pdb_bytes: bytes, chain_id: str | None = None):
    """Parse a PDB into the atom14 representation MarS-FM expects.

    Returns (atom14 ndarray shape (n_residues, 14, 3), seqres str).
    chain_id options:
        None          -> first protein chain only (single-chain default)
        "<letter>"    -> exactly that chain
        "ALL"         -> all protein chains concatenated into one virtual
                          chain (PDBFixer-renumbered order). For systems
                          where MarS-FM-native multi-chain inference is
                          unavailable but the partners are biologically
                          linked (e.g. Notch1 NEC + NTM that come from
                          one polyprotein cleaved at the S1 site)."""
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
    structure = parser.get_structure("p", io.StringIO(pdb_bytes.decode("utf-8")))
    model = next(structure.get_models())

    if chain_id is None:
        # First protein chain only.
        for chain in model:
            if any(is_aa(r, standard=True) for r in chain):
                residues = [r for r in chain if is_aa(r, standard=True)
                            and r.get_resname() in THREE_TO_ONE]
                break
        else:
            raise ValueError("no protein chain found")
    elif chain_id.upper() == "ALL":
        # Concatenate all protein chains in iteration order.
        residues = []
        for chain in model:
            for r in chain:
                if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE:
                    residues.append(r)
    elif "," in chain_id:
        # Comma-separated subset of chain IDs, e.g. "X,K" for Notch1 NRR
        # (NEC+NTM) without the Fab chains. Order in concat follows the
        # comma-separated list, NOT the PDB order.
        wanted = [c.strip() for c in chain_id.split(",")]
        residues = []
        for wid in wanted:
            chain = model[wid]
            for r in chain:
                if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE:
                    residues.append(r)
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
    atom14, seqres, cids = _pdb_to_atom14_and_seqres_and_chainid(
        pdb_bytes, chain_id=chain_id)
    n_chains = int(cids.max()) + 1
    chain_lens = [int((cids == c).sum()) for c in range(n_chains)]
    print(f"[marsfm] atom14 shape {atom14.shape}, seqres len {len(seqres)}, "
          f"{n_chains} chain(s) of lengths {chain_lens}")
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

        # Save atom14 input. MarS-FM's load_starting_structure expects
        # shape (n_frames, n_residues, 14, 3) and slices [0:1]; we add
        # a leading singleton frame dim.
        np.save(data_dir / f"{name}.npy", atom14[np.newaxis])
        # Phase-1 multi-chain API: emit a per-residue chain_id npy that
        # our forked load_starting_structure will pick up (single-chain
        # input writes all-zeros, which is the trained baseline path).
        np.save(data_dir / f"{name}_chain_id.npy", cids)

        # One-row CSV split.
        split_csv = splits_dir / "single.csv"
        split_csv.write_text(f"name,seqres\n{name},{seqres}\n")

        # MarS-FM's generate.py emits 1 frame per MarS call (the
        # --max_mars_samples flag caps internal MSM exploration per call,
        # not output frame count). So calls_mars == n_samples.
        cmd = [
            sys.executable, "-m", "scripts.generate",
            "--mars_ckpt", mars_ckpt,
            "--data_dir", str(data_dir),
            "--split", str(split_csv),
            "--out_dir", str(out_dir),
            "--pdb_id", name,
            "--calls_mars", str(n_samples),
            "--max_mars_samples", "8",
        ]
        print(f"[marsfm] calls_mars={n_samples} (1 output frame per call)")
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
