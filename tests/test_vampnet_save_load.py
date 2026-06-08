"""Round-trip tests for vampnet_core.save / load — previously untested.

save/load are pure pickle over the VAMPnetModel dataclass, so no torch
is required to exercise persistence + the summary() projection.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _model():
    from vampnet_core import VAMPnetModel
    return VAMPnetModel(
        n_states=3,
        lag=20,
        features="torsions",
        epochs=60,
        vamp2_score=2.71,
        implied_timescales=[100.0, 50.0],
        state_populations=[0.5, 0.3, 0.2],
        state_assignments=[0, 1, 2, 0],
        transition_matrix=[[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.0, 0.8]],
        stationary_distribution=[0.5, 0.3, 0.2],
        sources=["md", "md", "alphaflow", "md"],
        n_frames=4,
        model_id="m1",
    )


def test_save_returns_path_and_size():
    import vampnet_core
    m = _model()
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "m.pkl")
        info = vampnet_core.save(m, p)
        assert info["path"] == p
        assert info["bytes"] > 0
        assert Path(p).exists()


def test_save_load_round_trip_is_identical():
    import vampnet_core
    m = _model()
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "m.pkl")
        vampnet_core.save(m, p)
        loaded = vampnet_core.load(p)
        # dataclass __eq__ compares all fields.
        assert loaded == m
        assert loaded.summary() == m.summary()


def test_summary_projection():
    m = _model()
    s = m.summary()
    assert s == {
        "vamp2_score": 2.71,
        "implied_timescales": [100.0, 50.0],
        "state_populations": [0.5, 0.3, 0.2],
        "n_states": 3,
        "n_frames": 4,
        "model_id": "m1",
    }
