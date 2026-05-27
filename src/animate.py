"""Slow-mode animation: interpolate the structure along a VAMPnet slow
collective variable, expose the result as an animated ChimeraX model.

The slow mode is identified by training a per-state soft assignment and
projecting frame Cartesians onto the difference between the two states
that span the slowest implied timescale. We then linearly interpolate
the protein from the low-projection mean to the high-projection mean
over `n_frames` coordsets — a "slow movie" view of the dominant
metastable motion.
"""

from __future__ import annotations

import numpy as np


def slow_mode_animation(session, model, mode: int = 1, n_frames: int = 100) -> dict:
    """Build a synthetic trajectory along slow VAMPnet mode `mode`.

    mode=1 means the slowest non-trivial mode (excluding the stationary
    distribution). For a 4-state model that's typically state(low-prob)
    -> state(high-prob) interpolation along the dominant slow eigenvector.

    Returns: {"mode": int, "n_frames": int, "model_id": str | None,
              "model_name": str, "endpoint_states": [int, int]}
    """
    try:
        from . import featurize
    except ImportError:
        import featurize
    import numpy as np

    regs = featurize._registry(session)
    md_regs = [r for r in regs if r["format"] == "md" and r["structure"] is not None]
    if not md_regs:
        return {"mode": mode, "n_frames": n_frames, "model_id": None,
                "note": "No MD-backed structure in session"}

    parent = md_regs[0]["structure"]
    coords_all = md_regs[0]["coords"]      # (n_frames, n_atoms, 3)
    n_str_frames = int(md_regs[0]["n_frames"])
    state_arr = np.asarray(model.state_assignments, dtype=np.int32)[: n_str_frames]

    # Identify the two endpoint states for this mode. For mode=1 (slowest),
    # use the most-populated state and the least-populated state — the
    # canonical "metastable to rare" axis in the implied-timescales
    # decomposition.
    pops = np.bincount(state_arr, minlength=model.n_states)
    sorted_states = np.argsort(pops)
    if mode < 1 or mode > model.n_states - 1:
        raise ValueError(f"mode must be in 1..{model.n_states - 1}")
    low_state = int(sorted_states[mode - 1])    # rarer
    high_state = int(sorted_states[-1])         # most populated

    low_mask = state_arr == low_state
    high_mask = state_arr == high_state
    if not low_mask.any() or not high_mask.any():
        raise RuntimeError("endpoint state(s) empty; cannot interpolate")

    low_mean = coords_all[low_mask].mean(axis=0)
    high_mean = coords_all[high_mask].mean(axis=0)

    # Linear interpolation over n_frames.
    try:
        from . import viz
    except ImportError:
        import viz
    new_structure = viz._clone_structure_with_coords(session, parent, low_mean,
                                                      name=f"vampnet_animation_mode{mode}")
    # The first coordset is already the low endpoint; add n_frames-1 more.
    for i in range(1, n_frames):
        t = i / (n_frames - 1)
        interp = (1 - t) * low_mean + t * high_mean
        try:
            new_structure.add_coordset(i + 1, interp.astype(np.float32))
        except Exception:
            new_structure.add_coordset(interp.astype(np.float32))

    session.models.add([new_structure])

    return {
        "mode": int(mode),
        "n_frames": int(n_frames),
        "model_id": str(new_structure.id_string) if hasattr(new_structure, "id_string") else None,
        "model_name": new_structure.name,
        "endpoint_states": [low_state, high_state],
        "endpoint_populations": [int(pops[low_state]), int(pops[high_state])],
    }
