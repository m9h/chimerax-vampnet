"""Ensemble loading + featurization for chimerax-vampnet.

Loads conformational ensembles from six sources on equal footing:

  - MD trajectories (anything ChimeraX can open: .dcd, .xtc, .trr, .nc, ...)
  - AlphaFlow / ESMFlow-MD ensembles (numpy .npz with 'coords' (N, A, 3))
  - BioEmu ensembles (numpy .npz or a directory of .pdb files)
  - MarS-FM ensembles (numpy .npz)
  - Boltz-2 ensembles (numpy .npz; shape-compatible with BioEmu)
  - ESMFold2 ensembles (numpy .npz; seed-varied diffusion samples)

After loading, each frame becomes a row in an ensemble-frame index plus a
parallel "source" tag. The featurize() function pulls CA-CA distances,
backbone torsions, or contact maps over the union ensemble into a single
(N_frames, N_features) numpy matrix ready for the VAMPnet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


# In-memory session-scoped ensemble registry. Each entry is a dict:
#   {"source": str, "path": str, "format": str, "coords": np.ndarray,
#    "n_frames": int, "structure": chimerax StructureModel or None}
_ENSEMBLES_KEY = "_vampnet_ensembles"


def _registry(session) -> list:
    if not hasattr(session, _ENSEMBLES_KEY):
        setattr(session, _ENSEMBLES_KEY, [])
    return getattr(session, _ENSEMBLES_KEY)


def _detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith((".dcd", ".xtc", ".trr", ".nc", ".h5", ".pdb")):
        return "md"
    if "alphaflow" in p or "_af" in p or "esmflow" in p:
        return "alphaflow"
    if "boltz" in p:
        return "boltz"
    if "bioemu" in p:
        return "bioemu"
    if "marsfm" in p or "mars_fm" in p or "mars-fm" in p:
        return "marsfm"
    if "esmfold2" in p or "esmfold_2" in p or "esmfold-2" in p:
        return "esmfold2"
    if p.endswith(".npz"):
        return "alphaflow"  # default assumption for npz
    return "md"


def load_ensemble(session, source: str, path: str, format: str = "auto") -> Tuple[int, str]:
    """Load an ensemble from disk into the session registry.

    Returns (n_frames, format_detected).
    """
    import numpy as np

    fmt = format
    if fmt == "auto":
        fmt = _detect_format(path)

    if fmt == "md":
        # Defer to ChimeraX's own trajectory loaders so the structure shows
        # up in the session and we can leverage the coordset machinery.
        from chimerax.core.commands import run
        run(session, f"open {path}")
        # The last opened structure is now the active model; its coordset
        # gives us per-frame coordinates.
        structure = session.models.list()[-1]
        coords_list = []
        if hasattr(structure, "num_coordsets") and structure.num_coordsets > 1:
            for cs_id in structure.coordset_ids:
                structure.active_coordset_id = cs_id
                coords_list.append(structure.atoms.coords.copy())
        else:
            coords_list.append(structure.atoms.coords.copy())
        coords = np.stack(coords_list, axis=0)
    elif fmt == "alphaflow":
        # AlphaFlow output convention: .npz with key 'coords' of shape (N, A, 3).
        data = np.load(path)
        coords = data["coords"] if "coords" in data.files else data[data.files[0]]
        structure = None
    elif fmt in ("bioemu", "boltz"):
        # BioEmu output convention: .npz with key 'samples' of shape
        # (N, A, 3), or a directory of .pdb files. Boltz-2 output is
        # shape-compatible (same (N, A, 3) layout via the adapter in
        # md/boltz_modal.py); we accept the same .npz keys so the
        # bundle loader can ingest either sampler interchangeably.
        if os.path.isdir(path):
            # Glob *.pdb, load each via mdtraj or numpy parser.
            coords = _load_pdb_dir(path)
        else:
            data = np.load(path)
            for key in ("samples", "coords", "coords_ca"):
                if key in data.files:
                    coords = data[key]
                    break
            else:
                coords = data[data.files[0]]
        structure = None
    elif fmt == "marsfm":
        # MarS-FM (Kapusniak et al. 2025, arXiv:2509.24779) generates protein
        # conformational trajectories ~600x faster than explicit-solvent MD by
        # sampling MSM transitions via flow matching. The published model has
        # no released checkpoint or canonical output schema as of v0.2; this
        # loader assumes the output mirrors AlphaFlow/BioEmu (a directory of
        # PDB samples or an .npz of (N, A, 3) coords). When the official
        # checkpoint drops, revisit the key names and add any author-provided
        # metadata (state assignment, lag-time, T condition).
        if os.path.isdir(path):
            coords = _load_pdb_dir(path)
        else:
            data = np.load(path)
            for key in ("samples", "coords", "x", "conformations"):
                if key in data.files:
                    coords = data[key]
                    break
            else:
                coords = data[data.files[0]]
        structure = None
    elif fmt == "esmfold2":
        # ESMFold2 (Rives et al. 2026, "A World Model of Protein Biology")
        # is an AF3-class diffusion structure/complex predictor on top of
        # the ESMC protein language model. It is a SINGLE-STRUCTURE
        # predictor (its authors explicitly note it is not a dynamics
        # model); we treat its seed-varied diffusion draws as one more
        # generative source on equal footing with Boltz-2 / AlphaFlow,
        # primarily to (a) H3-stress-test whether even SOTA single-
        # structure prediction collapses to the modal state, and (b)
        # provide a native multi-chain generative source for Notch1 NRR
        # where the MarS-FM multi-chain path is blocked. The adapter
        # md/esmfold2_modal.py emits an .npz with the same 'coords' /
        # 'coords_ca' layout as the boltz/marsfm adapters, plus
        # 'chain_id', 'plddt', and 'iptm' metadata for QC/filtering.
        if os.path.isdir(path):
            coords = _load_pdb_dir(path)
        else:
            data = np.load(path)
            for key in ("coords", "coords_ca", "samples"):
                if key in data.files:
                    coords = data[key]
                    break
            else:
                coords = data[data.files[0]]
        structure = None
    else:
        raise ValueError(f"unknown ensemble format: {fmt}")

    n_frames = int(coords.shape[0])
    _registry(session).append({
        "source": source,
        "path": path,
        "format": fmt,
        "coords": coords,
        "n_frames": n_frames,
        "structure": structure,
    })
    return n_frames, fmt


def _load_pdb_dir(directory: str):
    """Load a directory of .pdb files into a (N, A, 3) numpy array."""
    import numpy as np
    from glob import glob
    paths = sorted(glob(os.path.join(directory, "*.pdb")))
    if not paths:
        raise FileNotFoundError(f"no .pdb files in {directory}")
    # Minimal PDB parser: keep CA-record coords as the per-frame coords.
    # For richer parsing use mdtraj or biopython.
    out = []
    for p in paths:
        cs = []
        with open(p) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    cs.append((x, y, z))
        out.append(cs)
    arr = np.array(out, dtype=np.float32)
    return arr


def stacked_coords(session):
    """Concatenate all loaded ensembles into a single (N_total, A, 3) array.

    Also returns a per-frame source tag vector for downstream stratification.
    """
    import numpy as np
    regs = _registry(session)
    if not regs:
        raise RuntimeError("no ensembles loaded — run `vampnet load_ensemble` first")
    parts = [r["coords"] for r in regs]
    n_atoms = parts[0].shape[1]
    if any(p.shape[1] != n_atoms for p in parts):
        raise RuntimeError("ensemble atom counts disagree; align selections first")
    coords = np.concatenate(parts, axis=0)
    sources = np.concatenate([[r["source"]] * r["n_frames"] for r in regs])
    return coords, sources


def featurize_ca_distances(coords, ca_pairs=None):
    """coords: (N, A, 3) of just-CA atoms; returns (N, P) distance matrix.

    If ca_pairs is None, uses all upper-triangular pairs.
    """
    import numpy as np
    N, A, _ = coords.shape
    if ca_pairs is None:
        idx_i, idx_j = np.triu_indices(A, k=1)
    else:
        idx_i = np.array([p[0] for p in ca_pairs])
        idx_j = np.array([p[1] for p in ca_pairs])
    diffs = coords[:, idx_i, :] - coords[:, idx_j, :]
    dists = np.sqrt((diffs ** 2).sum(-1))
    # Standardize per pair across the ensemble.
    mu = dists.mean(0, keepdims=True)
    sd = dists.std(0, keepdims=True) + 1e-6
    return (dists - mu) / sd


def featurize_torsions(coords, torsion_atoms):
    """Backbone phi/psi torsions encoded as (sin, cos) pairs for circular continuity.

    Args:
        coords: (N, A_total, 3) full atom coordinates.
        torsion_atoms: list of (i, j, k, l) atom-index quadruples defining
            each dihedral. For a chain of residues, the standard set is
            phi = (C_prev, N, CA, C) and psi = (N, CA, C, N_next) for each
            internal residue.

    Returns:
        (N, 2*P) feature matrix of stacked (sin θ, cos θ) for each torsion.
    """
    import numpy as np

    quads = np.asarray(torsion_atoms, dtype=np.int64)
    p0 = coords[:, quads[:, 0], :]
    p1 = coords[:, quads[:, 1], :]
    p2 = coords[:, quads[:, 2], :]
    p3 = coords[:, quads[:, 3], :]

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9))
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    theta = np.arctan2(y, x)  # shape (N, P)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    feats = np.concatenate([sin_t, cos_t], axis=-1).astype(np.float32)
    return feats


def detect_backbone_torsions(structure):
    """Pick backbone phi/psi atom-index quadruples from a ChimeraX
    Structure. Returns a list of (i, j, k, l) into structure.atoms.

    phi_n = C_{n-1}, N_n,  CA_n, C_n
    psi_n = N_n,    CA_n,  C_n,  N_{n+1}
    """
    atoms = structure.atoms
    # Build per-residue atom-name -> atom-index lookup.
    residues = structure.residues
    res_atoms = []
    for r in residues:
        nm = {a.name: i for i, a in enumerate(atoms) if a.residue == r}
        res_atoms.append(nm)

    quads = []
    for i, r in enumerate(residues):
        prev = res_atoms[i - 1] if i > 0 else None
        cur = res_atoms[i]
        nxt = res_atoms[i + 1] if i + 1 < len(residues) else None
        if prev and "C" in prev and "N" in cur and "CA" in cur and "C" in cur:
            quads.append((prev["C"], cur["N"], cur["CA"], cur["C"]))
        if nxt and "N" in cur and "CA" in cur and "C" in cur and "N" in nxt:
            quads.append((cur["N"], cur["CA"], cur["C"], nxt["N"]))
    return quads
