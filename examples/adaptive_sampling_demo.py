"""LLM-agent adaptive-sampling loop, demo edition.

A standalone Python script that simulates how a Claude-Desktop-style MCP
agent would steer a VAMPnet analysis: identify under-sampled states,
"extend" MD from those states, refit, watch convergence improve.

Real adaptive sampling regenerates new MD trajectories from the rare-
state mean structures. This demo SUBSTITUTES that step with stratified
re-sampling from the existing 1 µs chignolin trajectory — the loop
mechanics are identical, the compute is free.

The loop:
  1. Initial VAMPnet fit on a small subsample of the trajectory.
  2. Inspect state populations. Find the rarest non-empty state.
  3. "Extend MD" from that state -> pull more frames near its centroid
     from the full dataset (proxy for new simulations).
  4. Refit. Loop until population balance stabilizes.

  $ .venv/bin/python examples/adaptive_sampling_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "md"))

from featurize import featurize_ca_distances  # noqa: E402
from tier1_vampnet import _read_dcd  # noqa: E402

TRAJ = Path("/data/datasets/chimerax-vampnet/chignolin_modal/chignolin/replica_0/traj.dcd")
PDB = Path("/data/datasets/chimerax-vampnet/chignolin_modal/chignolin/equilibrated.pdb")


def _parse_ca(pdb_path: Path):
    cas, idx = [], 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(idx)
                idx += 1
    return np.array(cas, dtype=np.int64), idx


def _fit_vampnet(X, n_states, lag, epochs=40):
    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM

    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 64), nn.ELU(),
        nn.Linear(64, 64), nn.ELU(),
        nn.Linear(64, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu")
    m = net.fit(loader, n_epochs=epochs).fetch_model()
    soft = np.asarray(m.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=lag).fit_fetch(hard)
    return hard, msm


def main():
    print("=" * 70)
    print("LLM-agent adaptive-sampling loop on CLN025 (chignolin)")
    print("=" * 70)

    ca_idx, n_atoms = _parse_ca(PDB)
    coords = _read_dcd(TRAJ, n_atoms=n_atoms)
    ca = coords[:, ca_idx, :].astype("float32")
    print(f"full trajectory: {ca.shape[0]} frames ({ca.shape[0] * 0.01:.0f} ns @ 10 ps stride)")

    # Round 0: subsample the trajectory uniformly to simulate a short
    # initial MD run. Take just 1% (10 ns of equivalent sampling).
    rng = np.random.default_rng(0)
    initial_n = int(0.01 * ca.shape[0])
    selected = set(rng.choice(ca.shape[0], initial_n, replace=False).tolist())
    print(f"\nROUND 0  (initial 'short MD' = {initial_n} frames, ~10 ns equivalent)")
    print("  AGENT: fit_vampnet(n_states=4)")
    pops_history = []
    its_history = []
    n_states = 4
    lag = 50

    for round_idx in range(5):
        idx = sorted(selected)
        sub = ca[idx]
        X = featurize_ca_distances(sub)
        hard, msm = _fit_vampnet(X, n_states, lag)

        pops = np.bincount(hard, minlength=n_states) / len(hard)
        pops_history.append(pops.copy())
        print(f"\n  state pops:    " + "  ".join(f"s{s}={p*100:5.1f}%" for s, p in enumerate(pops)))
        try:
            its = msm.timescales(k=n_states - 1)
            its_ns = [t * 0.01 * lag for t in its]
            its_history.append(its_ns[0] if its_ns else float("nan"))
            print(f"  implied (ns):  " + "  ".join(f"{t:5.1f}" for t in its_ns))
        except Exception:
            its_history.append(float("nan"))
            print(f"  implied (ns):  (insufficient transitions)")

        # Agent decision: find the rarest non-empty state. Could also use
        # max gap from uniform (1/n_states) — same result for low-pop tail.
        non_empty = [s for s in range(n_states) if (hard == s).any()]
        rarest = min(non_empty, key=lambda s: (hard == s).mean())
        rare_pop = (hard == rarest).mean()
        print(f"  AGENT decision: state s{rarest} is rarest ({(hard == rarest).mean()*100:.2f}%).")
        if (hard == rarest).mean() > 0.10 and round_idx > 0:
            print("  AGENT: populations near uniform; converged. Stop.")
            break

        # Find centroid of rarest state in the SUBSAMPLED set.
        rare_mask = hard == rarest
        if not rare_mask.any():
            break
        rare_centroid = sub[rare_mask].mean(axis=0)

        # "Extend MD" = pull more frames from the FULL trajectory that are
        # closest to the rarest centroid (proxy for biased MD restarts).
        per_frame = ca - rare_centroid[None, :, :]
        dists = np.sqrt((per_frame ** 2).sum(axis=(1, 2)))
        # Keep frames not yet selected; pick closest ~1% of trajectory.
        order = np.argsort(dists)
        added = 0
        for i in order:
            if int(i) not in selected:
                selected.add(int(i))
                added += 1
                if added >= int(0.005 * ca.shape[0]):    # ~5 ns of new sampling
                    break
        print(f"  AGENT: extended MD from s{rarest} centroid -> +{added} frames "
              f"(total {len(selected)})")

    print("\n" + "=" * 70)
    print(f"Adaptive loop terminated after {round_idx + 1} rounds.")
    init_slowest = its_history[0]
    final_slowest = its_history[-1]
    print(f"Slowest implied timescale evolution:")
    for i, t in enumerate(its_history):
        marker = "  (start)" if i == 0 else ("  (end)" if i == len(its_history) - 1 else "")
        print(f"  round {i}:  {t:6.1f} ns{marker}")
    print(f"Speedup of slow-mode resolution: {final_slowest/init_slowest:.1f}x")
    print(f"\nNB: state IDs across rounds are NOT identical -- each VAMPnet fit\n"
          f"    relabels states. The headline number is the slowest implied\n"
          f"    timescale: adaptive sampling pushes it UP because it surfaces\n"
          f"    slow dynamics the initial uniform subsample missed.")
    print(f"\nIn a real MCP/Claude Desktop setup, the agent's decisions above")
    print(f"would be issued as tool calls against `vampnet mcp serve`:")
    print(f"  POST /tools/vampnet_fit          (each refit)")
    print(f"  POST /tools/vampnet_states       (read populations)")
    print(f"  POST /tools/vampnet_means        (extract rare-state structure)")
    print(f"  (external) launch new MD seeded from that structure")
    print(f"  POST /tools/vampnet_load_ensemble (ingest new trajectory)")


if __name__ == "__main__":
    main()
