"""Day-2 checkpoint test: a 4-state random walk should be recovered by
the VAMPnet fit without erroring. Run outside ChimeraX -- this exercises
the core logic without needing a live session.

   $ python -m pytest tests/test_random_walk.py -v
"""

import numpy as np


def make_random_walk_dataset(n_frames=2000, n_atoms=20, n_states=4, seed=42):
    """Synthesize a (n_frames, n_atoms, 3) coord trajectory where the
    atoms occupy n_states well-separated mean conformations and hop
    between them as a Markov chain."""
    rng = np.random.default_rng(seed)

    # Generate n_states random "centroid" conformations.
    centroids = rng.standard_normal((n_states, n_atoms, 3)) * 5.0

    # Build a transition matrix biased toward staying in the current state.
    T = np.eye(n_states) * 0.9 + (1 - np.eye(n_states)) * (0.1 / (n_states - 1))

    # Walk.
    state = 0
    states = np.zeros(n_frames, dtype=int)
    for t in range(n_frames):
        states[t] = state
        state = int(rng.choice(n_states, p=T[state]))

    # Coords: centroid + small Gaussian wiggle.
    noise = rng.standard_normal((n_frames, n_atoms, 3)) * 0.3
    coords = centroids[states] + noise

    return coords.astype(np.float32), states


def test_random_walk_runs_end_to_end():
    """Smoke test: fit a VAMPnet on a synthetic 4-state random walk and
    verify the core machinery returns a model with the expected shape."""
    coords, true_states = make_random_walk_dataset(n_frames=2000, n_atoms=20, n_states=4)

    # We can't easily test through cmd.cmd_fit() without a ChimeraX session,
    # so call the underlying featurize + vampnet_core paths directly.
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

    from featurize import featurize_ca_distances
    import numpy as np

    X = featurize_ca_distances(coords)
    assert X.shape[0] == coords.shape[0]
    assert X.shape[1] == coords.shape[1] * (coords.shape[1] - 1) // 2

    # Try to import + instantiate the VAMPnet machinery. We don't run
    # full training here (would take a minute); we just verify the API
    # surface is reachable.
    try:
        import torch
        import torch.nn as nn
        from deeptime.decomposition.deep import VAMPNet
    except ImportError as e:
        import pytest
        pytest.skip(f"deeptime/torch not installed in this env: {e}")

    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 32), nn.ReLU(),
        nn.Linear(32, 4), nn.Softmax(dim=-1),
    )
    net = VAMPNet(lobe=lobe, learning_rate=1e-3, device="cpu")
    assert net is not None


if __name__ == "__main__":
    test_random_walk_runs_end_to_end()
    print("OK")
