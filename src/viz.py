"""ChimeraX visualization for the chimerax-vampnet bundle.

Two user-facing operations:
  - color_by_state(): paint the trajectory's coordsets by VAMPnet state,
    so as the user scrubs the timeline (or animates) the structure
    changes color with the metastable assignment.
  - build_state_means(): create one new AtomicStructure model per state,
    holding the per-residue mean structure of all frames assigned to
    that state. Lets the user visually compare metastable conformations
    side-by-side.

Both rely on a fitted VAMPnetModel and the loaded ensemble registry from
featurize.py to know which frames belong to which ChimeraX structure.
"""

from __future__ import annotations

import numpy as np


# Discrete colormap (Tab10-ish) — 8 states max for v0.1. RGBA in 0..255.
_STATE_PALETTE = [
    (31, 119, 180, 255),    # blue
    (255, 127, 14, 255),    # orange
    (44, 160, 44, 255),     # green
    (214, 39, 40, 255),     # red
    (148, 103, 189, 255),   # purple
    (140, 86, 75, 255),     # brown
    (227, 119, 194, 255),   # pink
    (127, 127, 127, 255),   # gray
]


def color_by_state(session, model) -> dict:
    """Color the loaded structure(s) at the current frame by the VAMPnet
    state at that frame. Re-coloring fires on every coordset change so
    timeline scrubbing produces live state coloring.

    Returns: {"n_frames": int, "n_states": int, "state_counts": [int, ...],
              "structure": str | None, "palette": [[r,g,b,a], ...]}
    """
    try:
        from . import featurize
    except ImportError:
        import featurize

    n_frames = len(model.state_assignments)
    counts = [int(sum(1 for s in model.state_assignments if s == k)) for k in range(model.n_states)]
    state_arr = np.asarray(model.state_assignments, dtype=np.int32)

    # Locate the ChimeraX structure associated with the first MD ensemble
    # (AlphaFlow/BioEmu ensembles aren't ChimeraX-visible, so we colour
    # only the MD-backed structure for v0.1).
    regs = featurize._registry(session)
    md_regs = [r for r in regs if r["format"] == "md" and r["structure"] is not None]
    if not md_regs:
        return {
            "n_frames": int(n_frames),
            "n_states": int(model.n_states),
            "state_counts": counts,
            "structure": None,
            "palette": [list(c) for c in _STATE_PALETTE[: model.n_states]],
            "note": "No MD-backed structure in session; coloring is a no-op",
        }

    structure = md_regs[0]["structure"]
    # Index range of frames belonging to this structure in the concatenated
    # state vector. The first MD ensemble starts at 0.
    start = 0
    for r in regs:
        if r is md_regs[0]:
            break
        start += int(r["n_frames"])
    n_str_frames = int(md_regs[0]["n_frames"])
    structure_states = state_arr[start : start + n_str_frames]

    def _apply_for_frame(frame_idx: int):
        s = int(structure_states[frame_idx])
        rgba = _STATE_PALETTE[s % len(_STATE_PALETTE)]
        structure.atoms.colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(structure.atoms), 1))
        try:
            structure.residues.ribbon_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(structure.residues), 1))
        except Exception:
            pass

    # Color the current frame.
    try:
        coord_ids = list(structure.coordset_ids)
        active = structure.active_coordset_id
        frame_idx = coord_ids.index(active)
        _apply_for_frame(frame_idx)
    except Exception:
        frame_idx = 0
        _apply_for_frame(0)

    # Register a callback so future coordset changes recolor automatically.
    if not getattr(structure, "_vampnet_color_handler", None):
        def _handler(trigger_name, changes):
            try:
                ids = list(structure.coordset_ids)
                idx = ids.index(structure.active_coordset_id)
                if 0 <= idx < len(structure_states):
                    _apply_for_frame(idx)
            except Exception:
                pass

        try:
            handler = structure.triggers.add_handler("changes", _handler)
            structure._vampnet_color_handler = handler
        except Exception:
            pass

    return {
        "n_frames": int(n_frames),
        "n_states": int(model.n_states),
        "state_counts": counts,
        "structure": str(structure.id_string) if hasattr(structure, "id_string") else str(structure.id),
        "current_frame": int(frame_idx),
        "current_state": int(structure_states[frame_idx]) if frame_idx < len(structure_states) else None,
        "palette": [list(c) for c in _STATE_PALETTE[: model.n_states]],
    }


def build_state_means(session, model) -> dict:
    """Compute the mean structure over all frames assigned to each state
    and add one new AtomicStructure model per state to the session.

    Returns: {"models": [{"state": int, "model_id": str, "model_name": str,
                          "n_frames": int}, ...]}
    """
    try:
        from . import featurize
    except ImportError:
        import featurize

    regs = featurize._registry(session)
    md_regs = [r for r in regs if r["format"] == "md" and r["structure"] is not None]
    if not md_regs:
        return {"models": [], "note": "No MD-backed structure in session"}

    parent = md_regs[0]["structure"]
    coords_all = md_regs[0]["coords"]   # (n_frames, n_atoms, 3)
    n_str_frames = int(md_regs[0]["n_frames"])

    state_arr = np.asarray(model.state_assignments, dtype=np.int32)
    structure_states = state_arr[:n_str_frames]

    out_models = []
    new_models = []
    for s in range(model.n_states):
        mask = structure_states == s
        if mask.sum() < 1:
            out_models.append({"state": s, "model_id": None, "n_frames": 0})
            continue
        mean_coords = coords_all[mask].mean(axis=0)
        new_structure = _clone_structure_with_coords(session, parent, mean_coords, name=f"vampnet_state_{s}")
        rgba = _STATE_PALETTE[s % len(_STATE_PALETTE)]
        try:
            new_structure.atoms.colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(new_structure.atoms), 1))
            new_structure.residues.ribbon_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(new_structure.residues), 1))
        except Exception:
            pass
        new_models.append(new_structure)
        out_models.append({
            "state": int(s),
            "model_id": str(new_structure.id_string) if hasattr(new_structure, "id_string") else None,
            "model_name": new_structure.name,
            "n_frames": int(mask.sum()),
            "color_rgba": list(rgba),
        })

    if new_models:
        session.models.add(new_models)

    return {"models": out_models}


def _clone_structure_with_coords(session, parent, coords: np.ndarray, name: str):
    """Build an AtomicStructure with the same topology as `parent` but
    with `coords` as the single coordset."""
    from chimerax.atomic import AtomicStructure
    import numpy as np

    new = AtomicStructure(session, name=name)
    # Add atoms in the same order as parent. Each atom gets element + name
    # + residue assignment.
    residue_map = {}
    for parent_atom, xyz in zip(parent.atoms, coords):
        r = parent_atom.residue
        key = (r.chain_id, r.number, r.insertion_code, r.name)
        if key not in residue_map:
            residue_map[key] = new.new_residue(r.name, r.chain_id, r.number, r.insertion_code)
        atom = new.new_atom(parent_atom.name, parent_atom.element)
        atom.coord = np.asarray(xyz, dtype=np.float32)
        residue_map[key].add_atom(atom)

    # Recreate bonds based on parent's bond list.
    parent_atoms = list(parent.atoms)
    new_atoms = list(new.atoms)
    idx_of = {pa: i for i, pa in enumerate(parent_atoms)}
    for bond in parent.bonds:
        a, b = bond.atoms
        new.new_bond(new_atoms[idx_of[a]], new_atoms[idx_of[b]])
    return new
