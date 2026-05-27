"""Tests for the implied-timescales convergence diagnostic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _markov_chain(n_frames, T, seed=0):
    """Sample a discrete Markov chain from transition matrix T."""
    rng = np.random.RandomState(seed)
    n_states = T.shape[0]
    states = np.zeros(n_frames, dtype=int)
    cur = 0
    for i in range(n_frames):
        states[i] = cur
        cur = int(rng.choice(n_states, p=T[cur]))
    return states


def _model_from_assignments(assignments, n_states, lag=1):
    from vampnet_core import VAMPnetModel
    return VAMPnetModel(
        n_states=n_states, lag=lag, features="ca_distances", epochs=0,
        vamp2_score=0.0, implied_timescales=[], state_populations=[1.0/n_states]*n_states,
        state_assignments=list(int(a) for a in assignments),
        transition_matrix=[[0]*n_states for _ in range(n_states)],
        stationary_distribution=[1.0/n_states]*n_states,
        sources=["test"]*len(assignments), n_frames=len(assignments),
        model_id="t",
    )


def test_convergence_diagnostic_markovian_chain():
    """A truly-Markovian chain should report timescales that are nearly
    constant across lags -> high convergence."""
    import vampnet_core

    # 4-state ring, slow rotation.
    n = 4
    T = np.array([
        [0.95, 0.05, 0.00, 0.00],
        [0.00, 0.95, 0.05, 0.00],
        [0.00, 0.00, 0.95, 0.05],
        [0.05, 0.00, 0.00, 0.95],
    ])
    states = _markov_chain(20000, T, seed=42)
    model = _model_from_assignments(states, n)
    out = vampnet_core.implied_timescales(model, taus=[1, 2, 5, 10, 20, 50])

    assert "convergence_score" in out
    assert "recommended_lag" in out
    assert len(out["timescales"]) == 6
    # The slowest mode CV should be low for a clean Markovian chain.
    assert out["convergence_score"][0] < 0.5, f"slowest mode CV too high: {out['convergence_score']}"


def test_convergence_diagnostic_handles_failed_fits():
    """If some lag-fits fail (e.g., disconnected sub-states), the
    diagnostic should not crash and should report failure cleanly."""
    import vampnet_core

    # Tiny trajectory + many states -> some MSM fits will fail.
    states = np.array([0, 1, 0, 1, 2, 1, 0, 2] * 5, dtype=int)
    model = _model_from_assignments(states, n_states=4)
    out = vampnet_core.implied_timescales(model, taus=[1, 10, 100])

    assert "converged" in out
    # 4 states -> 3 modes; converged is a list of bools
    assert len(out["converged"]) == 3


if __name__ == "__main__":
    test_convergence_diagnostic_markovian_chain()
    test_convergence_diagnostic_handles_failed_fits()
    print("OK")
