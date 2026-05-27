"""ChimeraX command registration for the chimerax-vampnet bundle.

Every command in this module returns a JSON-serializable dict (when it
returns anything) so the ChimeraX MCP-server bundle can route results to
an external LLM agent. Side-effects on the session (new models, frame
coloring, animations) happen alongside the structured return.

The actual numerical work lives in featurize.py / vampnet_core.py /
msm.py / viz.py / animate.py. This module is the registration + dispatch
layer.
"""

from chimerax.core.commands import (
    CmdDesc,
    register,
    IntArg,
    FloatArg,
    BoolArg,
    StringArg,
    EnumOf,
    OpenFileNameArg,
    SaveFileNameArg,
    ListOf,
)
from chimerax.core.errors import UserError


# ----------------------------------------------------------------------
# Session-scoped state.
# ----------------------------------------------------------------------
# A single VAMPnet "model object" is held on the session under this key.
# Commands like `vampnet states`, `vampnet animate` consume it after
# `vampnet fit` has produced it.
_SESSION_KEY = "_vampnet_model"


def _state_get(session):
    return getattr(session, _SESSION_KEY, None)


def _state_set(session, model):
    setattr(session, _SESSION_KEY, model)


# ----------------------------------------------------------------------
# Command implementations.
# ----------------------------------------------------------------------
def cmd_load_ensemble(session, source, path, format="auto"):
    """Load a conformational ensemble into the session.

    source: human-readable label ("md_apo", "alphaflow", "bioemu", etc.)
    path:   filesystem path to a trajectory or ensemble file
    format: one of "auto", "alphaflow", "bioemu", "md"

    Returns: {"source": str, "n_frames": int, "format_detected": str}
    """
    from . import featurize
    n_frames, fmt = featurize.load_ensemble(session, source, path, format)
    return {"source": source, "n_frames": n_frames, "format_detected": fmt}


def cmd_fit(session, n_states=4, lag=10, features="ca_distances", epochs=200):
    """Fit a VAMPnet on the union of loaded ensembles.

    Returns: {
        "vamp2_score": float,
        "implied_timescales": [float, ...],
        "state_populations": [float, ...],
        "n_states": int,
        "n_frames": int,
        "model_id": str,
    }
    """
    from . import vampnet_core
    model = vampnet_core.fit(
        session,
        n_states=n_states,
        lag=lag,
        features=features,
        epochs=epochs,
    )
    _state_set(session, model)
    return model.summary()


def cmd_timescales(session, taus=None):
    """Compute the implied timescales test for the active VAMPnet.

    Returns: {"taus": [int, ...], "timescales": [[float, ...], ...]}
    """
    model = _state_get(session)
    if model is None:
        raise UserError("no fitted VAMPnet — run `vampnet fit` first")
    if taus is None:
        taus = [1, 2, 5, 10, 20, 50, 100]
    from . import vampnet_core
    return vampnet_core.implied_timescales(model, taus)


def cmd_states(session):
    """Color each trajectory frame by its VAMPnet-assigned state.

    Returns: {"n_frames": int, "n_states": int, "state_counts": [int, ...]}
    """
    model = _state_get(session)
    if model is None:
        raise UserError("no fitted VAMPnet — run `vampnet fit` first")
    from . import viz
    return viz.color_by_state(session, model)


def cmd_means(session):
    """Build one mean-structure model per VAMPnet state.

    Returns: {"models": [{"state": int, "model_id": str, "n_frames": int}, ...]}
    """
    model = _state_get(session)
    if model is None:
        raise UserError("no fitted VAMPnet — run `vampnet fit` first")
    from . import viz
    return viz.build_state_means(session, model)


def cmd_animate(session, mode=1, n_frames=100):
    """Generate a synthetic trajectory interpolating along a slow VAMPnet mode.

    Returns: {"mode": int, "n_frames": int, "model_id": str}
    """
    model = _state_get(session)
    if model is None:
        raise UserError("no fitted VAMPnet — run `vampnet fit` first")
    from . import animate
    return animate.slow_mode_animation(session, model, mode, n_frames)


def cmd_network(session):
    """Return the MSM transition matrix as a structured dict.

    Returns: {"states": [int, ...], "transition_matrix": [[float, ...]],
              "stationary_distribution": [float, ...]}
    """
    model = _state_get(session)
    if model is None:
        raise UserError("no fitted VAMPnet — run `vampnet fit` first")
    from . import msm
    return msm.transition_graph(model)


def cmd_save(session, path):
    """Save the fitted VAMPnet + MSM to disk.

    Returns: {"path": str, "bytes": int}
    """
    model = _state_get(session)
    if model is None:
        raise UserError("no fitted VAMPnet — nothing to save")
    from . import vampnet_core
    return vampnet_core.save(model, path)


def cmd_load(session, path):
    """Load a previously saved VAMPnet + MSM from disk.

    Returns: {"path": str, "model_id": str, "n_states": int}
    """
    from . import vampnet_core
    model = vampnet_core.load(path)
    _state_set(session, model)
    return model.summary()


def cmd_mcp_serve(session, port=7345):
    """Start the MCP bridge so MCP-capable LLM agents can drive this
    bundle. Returns: {"status": str, "port": int, "tools": [str]}
    """
    from . import mcp_server
    return mcp_server.start(session, port=int(port))


def cmd_mcp_stop(session):
    """Stop the MCP bridge. Returns: {"status": str}"""
    from . import mcp_server
    return mcp_server.stop()


# ----------------------------------------------------------------------
# Command descriptors.
# ----------------------------------------------------------------------
_DESC_LOAD_ENSEMBLE = CmdDesc(
    required=[("source", StringArg), ("path", OpenFileNameArg)],
    keyword=[("format", EnumOf(["auto", "alphaflow", "bioemu", "md"]))],
    synopsis="Load a conformational ensemble",
)

_DESC_FIT = CmdDesc(
    keyword=[
        ("n_states", IntArg),
        ("lag", IntArg),
        ("features", EnumOf(["ca_distances", "torsions", "contacts"])),
        ("epochs", IntArg),
    ],
    synopsis="Fit a VAMPnet on the loaded ensembles",
)

_DESC_TIMESCALES = CmdDesc(
    keyword=[("taus", ListOf(IntArg))],
    synopsis="Implied-timescales test",
)

_DESC_STATES = CmdDesc(synopsis="Color frames by VAMPnet state")
_DESC_MEANS = CmdDesc(synopsis="Build a model per state mean structure")
_DESC_ANIMATE = CmdDesc(
    keyword=[("mode", IntArg), ("n_frames", IntArg)],
    synopsis="Animate along a slow VAMPnet mode",
)
_DESC_NETWORK = CmdDesc(synopsis="Transition matrix as a structured graph")
_DESC_SAVE = CmdDesc(required=[("path", SaveFileNameArg)], synopsis="Save the VAMPnet")
_DESC_LOAD = CmdDesc(required=[("path", OpenFileNameArg)], synopsis="Load a saved VAMPnet")
_DESC_MCP_SERVE = CmdDesc(keyword=[("port", IntArg)], synopsis="Start the MCP bridge for external LLM agents")
_DESC_MCP_STOP = CmdDesc(synopsis="Stop the MCP bridge")


def register_commands(logger):
    register("vampnet load_ensemble", _DESC_LOAD_ENSEMBLE, cmd_load_ensemble, logger=logger)
    register("vampnet fit", _DESC_FIT, cmd_fit, logger=logger)
    register("vampnet timescales", _DESC_TIMESCALES, cmd_timescales, logger=logger)
    register("vampnet states", _DESC_STATES, cmd_states, logger=logger)
    register("vampnet means", _DESC_MEANS, cmd_means, logger=logger)
    register("vampnet animate", _DESC_ANIMATE, cmd_animate, logger=logger)
    register("vampnet network", _DESC_NETWORK, cmd_network, logger=logger)
    register("vampnet save", _DESC_SAVE, cmd_save, logger=logger)
    register("vampnet load", _DESC_LOAD, cmd_load, logger=logger)
    register("vampnet mcp serve", _DESC_MCP_SERVE, cmd_mcp_serve, logger=logger)
    register("vampnet mcp stop", _DESC_MCP_STOP, cmd_mcp_stop, logger=logger)
