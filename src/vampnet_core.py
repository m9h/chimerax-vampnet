"""VAMPnet fitting and (de)serialization, wrapping deeptime."""

from __future__ import annotations

import dataclasses
import os
import pickle
from typing import List, Optional


@dataclasses.dataclass
class VAMPnetModel:
    """Container the rest of the bundle reads from. Lightweight and
    picklable; the heavy deeptime models hang off it but are produced
    lazily where possible."""

    n_states: int
    lag: int
    features: str
    epochs: int
    vamp2_score: float
    implied_timescales: List[float]
    state_populations: List[float]
    state_assignments: list  # length n_frames; int per frame
    transition_matrix: list  # n_states x n_states (list-of-lists)
    stationary_distribution: List[float]
    sources: list  # per-frame source tag (str)
    n_frames: int
    model_id: str

    def summary(self) -> dict:
        return {
            "vamp2_score": float(self.vamp2_score),
            "implied_timescales": [float(t) for t in self.implied_timescales],
            "state_populations": [float(p) for p in self.state_populations],
            "n_states": int(self.n_states),
            "n_frames": int(self.n_frames),
            "model_id": str(self.model_id),
        }


def fit(session, n_states=4, lag=10, features="ca_distances", epochs=200):
    """Fit a VAMPnet on the union of loaded ensembles."""
    from . import featurize

    coords, sources = featurize.stacked_coords(session)

    if features == "ca_distances":
        X = featurize.featurize_ca_distances(coords)
    elif features == "torsions":
        # Pull the backbone torsion quad set from the first loaded structure.
        regs = featurize._registry(session)
        structure = next((r["structure"] for r in regs if r["structure"] is not None), None)
        if structure is None:
            raise RuntimeError("torsions feature needs a ChimeraX structure in the session; load via vampnet load_ensemble source=md first")
        quads = featurize.detect_backbone_torsions(structure)
        if not quads:
            raise RuntimeError("no backbone phi/psi torsions found in the structure")
        X = featurize.featurize_torsions(coords, quads)
    elif features == "contacts":
        raise NotImplementedError(f"feature kind '{features}' not yet implemented in v0.1")
    else:
        raise ValueError(f"unknown feature kind: {features}")

    # deeptime VAMPNet — runs on PyTorch, accepts an (N, P) feature matrix
    # plus a lag time, produces a soft state assignment.
    import numpy as np
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    # Build the lobe network — a small MLP with softmax output over n_states.
    import torch
    import torch.nn as nn

    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_states), nn.Softmax(dim=-1),
    )

    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    vampnet = VAMPNet(lobe=lobe, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")

    # Light training loop. deeptime exposes fit() with default batching.
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    model = vampnet.fit(loader, n_epochs=epochs).fetch_model()

    # Hard state assignment per frame for downstream viz.
    soft = model.transform(X.astype("float32"))
    hard = np.asarray(soft).argmax(axis=-1)

    # Build a Markov state model from the assignments.
    msm_estimator = MaximumLikelihoodMSM(reversible=True, lagtime=lag)
    msm = msm_estimator.fit_fetch(hard)
    pi = msm.stationary_distribution
    T = msm.transition_matrix

    # Implied timescales at the fit's lag.
    its = msm.timescales(k=n_states - 1)

    populations = [float((hard == s).mean()) for s in range(n_states)]

    return VAMPnetModel(
        n_states=int(n_states),
        lag=int(lag),
        features=str(features),
        epochs=int(epochs),
        vamp2_score=float(getattr(model, "vamp_score", 0.0) or 0.0),
        implied_timescales=[float(t) for t in its],
        state_populations=populations,
        state_assignments=[int(h) for h in hard],
        transition_matrix=T.tolist(),
        stationary_distribution=[float(p) for p in pi],
        sources=[str(s) for s in sources],
        n_frames=int(len(hard)),
        model_id=f"vampnet_{n_states}_{lag}",
    )


def implied_timescales(model: VAMPnetModel, taus: List[int]):
    """Implied-timescales test — refit MSMs at each tau, return timescales.

    Returns: {"taus": [int], "timescales": [[float, ...], ...]}
    """
    import numpy as np
    from deeptime.markov.msm import MaximumLikelihoodMSM

    hard = np.asarray(model.state_assignments)
    out_ts = []
    for tau in taus:
        try:
            est = MaximumLikelihoodMSM(reversible=True, lagtime=int(tau))
            msm = est.fit_fetch(hard)
            ts = msm.timescales(k=model.n_states - 1)
            out_ts.append([float(t) for t in ts])
        except Exception:
            out_ts.append([float("nan")] * (model.n_states - 1))
    return {"taus": [int(t) for t in taus], "timescales": out_ts}


def save(model: VAMPnetModel, path: str) -> dict:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return {"path": path, "bytes": os.path.getsize(path)}


def load(path: str) -> VAMPnetModel:
    with open(path, "rb") as f:
        return pickle.load(f)
