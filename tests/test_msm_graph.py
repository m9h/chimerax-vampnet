"""Tests for msm.transition_graph — previously untested module.

transition_graph turns a fitted VAMPnetModel's transition matrix into a
nodes+edges graph dict for the MCP bridge. It is pure-python (no torch),
so we drive it with a hand-built VAMPnetModel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _toy_model(n_states=2):
    from vampnet_core import VAMPnetModel
    return VAMPnetModel(
        n_states=n_states,
        lag=10,
        features="ca_distances",
        epochs=1,
        vamp2_score=1.5,
        implied_timescales=[42.0],
        state_populations=[0.55, 0.45],
        state_assignments=[0, 1, 0, 1],
        transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
        stationary_distribution=[0.6, 0.4],
        sources=["md", "md", "md", "md"],
        n_frames=4,
        model_id="toy",
    )


def test_transition_graph_shape_and_nodes():
    import msm
    g = msm.transition_graph(_toy_model())
    assert g["states"] == [0, 1]
    assert g["lag"] == 10
    assert len(g["nodes"]) == 2
    assert g["nodes"][0] == {"id": 0, "population": 0.55, "stationary": 0.6}
    assert g["transition_matrix"] == [[0.8, 0.2], [0.3, 0.7]]


def test_transition_graph_excludes_self_loops():
    import msm
    g = msm.transition_graph(_toy_model())
    # 2 off-diagonal positive entries -> 2 edges, no diagonal self-loops.
    assert len(g["edges"]) == 2
    assert all(e["src"] != e["dst"] for e in g["edges"])
    edge = {(e["src"], e["dst"]): e["rate"] for e in g["edges"]}
    assert edge[(0, 1)] == 0.2
    assert edge[(1, 0)] == 0.3
    # Fully connected 2-state graph -> density 1.0.
    assert g["edge_density"] == 1.0


def test_transition_graph_drops_zero_rate_edges():
    import msm
    m = _toy_model()
    m.transition_matrix = [[1.0, 0.0], [0.5, 0.5]]  # 0->1 has zero rate
    g = msm.transition_graph(m)
    assert len(g["edges"]) == 1
    assert g["edges"][0] == {"src": 1, "dst": 0, "rate": 0.5}
    assert g["edge_density"] == 0.5
