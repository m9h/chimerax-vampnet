"""Markov state model output formatting.

The fitted VAMPnetModel already holds the transition matrix +
stationary distribution. This module turns those into structured
output suitable for the MCP bridge / a downstream LLM agent.
"""

from __future__ import annotations

from typing import List


def transition_graph(model) -> dict:
    """Return the MSM's transition matrix as a structured graph.

    The dict format mirrors a common graph-API shape (nodes + edges) so
    downstream callers can drop it into networkx, cytoscape, or any
    web visualization library without translation.

    Returns:
        {
          "states": [int, ...],
          "transition_matrix": [[float, ...], ...],   # row-stochastic
          "stationary_distribution": [float, ...],
          "lag": int,
          "nodes": [{"id": int, "population": float, "stationary": float}, ...],
          "edges": [{"src": int, "dst": int, "rate": float}, ...]
              # `rate` is the transition probability per lag step from src
              # to dst, excluding the diagonal (self-loops).
        }
    """
    n_states = int(model.n_states)
    T: List[List[float]] = [list(row) for row in model.transition_matrix]
    pi: List[float] = list(model.stationary_distribution)
    pops: List[float] = list(model.state_populations)

    nodes = [
        {"id": s, "population": float(pops[s]), "stationary": float(pi[s])}
        for s in range(n_states)
    ]
    edges = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            p = float(T[i][j])
            if p > 0.0:
                edges.append({"src": i, "dst": j, "rate": p})

    # Edge density gives the calling agent a quick sense of MSM
    # connectivity (1.0 = fully connected, much less if some states are
    # poorly sampled).
    max_edges = n_states * (n_states - 1)
    density = (len(edges) / max_edges) if max_edges > 0 else 0.0

    return {
        "states": list(range(n_states)),
        "transition_matrix": T,
        "stationary_distribution": pi,
        "lag": int(model.lag),
        "nodes": nodes,
        "edges": edges,
        "edge_density": density,
    }
