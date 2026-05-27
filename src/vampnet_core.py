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


def _restrict_to_ca(session, coords):
    """Subset full-atom coords (N, A_total, 3) to CA atoms only.

    Looks at the first MD-loaded structure in the session registry; if
    that structure exposes atom names, returns coords[:, ca_indices, :].
    For ensembles loaded from AlphaFlow / BioEmu .npz (no ChimeraX
    structure available), the file is assumed to already hold CA-only
    or backbone-equivalent coordinates and is returned unchanged.
    """
    try:
        from . import featurize
    except ImportError:
        import featurize  # type: ignore
    import numpy as np

    regs = featurize._registry(session)
    md_regs = [r for r in regs if r.get("format") == "md" and r.get("structure") is not None]
    if not md_regs:
        return coords

    structure = md_regs[0]["structure"]
    atom_names = [a.name for a in structure.atoms]
    ca_idx = np.array([i for i, n in enumerate(atom_names) if n == "CA"], dtype=np.int64)
    if ca_idx.size == 0:
        return coords
    return coords[:, ca_idx, :]


def fit(session, n_states=4, lag=10, features="ca_distances", epochs=200):
    """Fit a VAMPnet on the union of loaded ensembles."""
    from . import featurize

    coords, sources = featurize.stacked_coords(session)

    if features == "ca_distances":
        coords_ca = _restrict_to_ca(session, coords)
        X = featurize.featurize_ca_distances(coords_ca)
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
    """Implied-timescales test — refit MSMs at each tau, return timescales
    plus a convergence diagnostic.

    For a Markovian MSM, each implied timescale t_k(tau) is INDEPENDENT
    of the chosen lag tau (within noise). In practice short lags
    under-estimate t_k (non-Markovian gap) and longer lags converge to
    the true value. The convergence score per mode is the relative
    spread of t_k(tau) across the supplied taus.

    Returns:
      {"taus": [int],
       "timescales": [[float, ...]],            # rows = taus, cols = modes
       "converged": [bool],                     # per-mode (relative spread < 25%)
       "convergence_score": [float],            # per-mode CV (std/mean) over taus
       "recommended_lag": int}                  # smallest tau where all modes converged
    """
    import numpy as np
    from deeptime.markov.msm import MaximumLikelihoodMSM

    hard = np.asarray(model.state_assignments)
    n_modes = model.n_states - 1

    def _pad(row):
        row = list(row)
        if len(row) < n_modes:
            row = row + [float("nan")] * (n_modes - len(row))
        return row[:n_modes]

    out_ts = []
    for tau in taus:
        try:
            est = MaximumLikelihoodMSM(reversible=True, lagtime=int(tau))
            msm = est.fit_fetch(hard)
            ts = msm.timescales(k=n_modes)
            out_ts.append(_pad([float(t) for t in ts]))
        except Exception:
            out_ts.append([float("nan")] * n_modes)

    arr = np.asarray(out_ts, dtype=float)              # (n_taus, n_modes)
    # Strip rows that are entirely NaN (failed lags).
    finite_mask = ~np.all(np.isnan(arr), axis=1)

    if finite_mask.sum() < 2:
        return {
            "taus": [int(t) for t in taus],
            "timescales": out_ts,
            "converged": [False] * (model.n_states - 1),
            "convergence_score": [float("nan")] * (model.n_states - 1),
            "recommended_lag": int(model.lag),
            "note": "insufficient successful lag-fits to diagnose convergence",
        }

    finite = arr[finite_mask]
    means = np.nanmean(finite, axis=0)
    stds = np.nanstd(finite, axis=0)
    cv = stds / (np.abs(means) + 1e-9)                  # coefficient of variation
    converged = [bool(c < 0.25) for c in cv]            # CV < 25% as convergence

    # Smallest tau at which every mode satisfies (t(tau) - mean) < 25 % of mean.
    rec = int(model.lag)
    valid_taus = [int(t) for t, ok in zip(taus, finite_mask) if ok]
    for i, tau in enumerate(valid_taus):
        row = finite[i]
        if np.all(np.abs(row - means) / (np.abs(means) + 1e-9) < 0.25):
            rec = tau
            break

    return {
        "taus": [int(t) for t in taus],
        "timescales": out_ts,
        "converged": converged,
        "convergence_score": [float(c) for c in cv],
        "mean_timescales": [float(m) for m in means],
        "recommended_lag": rec,
    }


def save(model: VAMPnetModel, path: str) -> dict:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return {"path": path, "bytes": os.path.getsize(path)}


def load(path: str) -> VAMPnetModel:
    with open(path, "rb") as f:
        return pickle.load(f)
