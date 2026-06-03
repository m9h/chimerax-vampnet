"""Live MCP-driven adaptive sampling applied to Notch1 NRR (v0.6).

The v0.4 PoC (`examples/live_adaptive_sampling.py`) demonstrated the
loop closure: agent decides rare state, launches Modal MD, ingests
the new trajectory. This v0.6 version applies that loop to the
v0.3 Notch1 NRR apo system and asks a concrete scientific
question:

  Does MD, given more replicas with new random seeds, ever surface
  state 1 (AF3-class) or state 2 (flow-matching) within 20 ns of
  the v0.3 equilibrated start?

v0.5 said no with 100 % confidence in 1500 frames (states 1 and 2
were generative-only). v0.6's MCP-loop "feature demonstration" is
to extend that evidence with N more 20 ns bursts via the agent
control path and re-score against the v0.5 5-source VAMPnet.

Phase 1 (this script, v0.6): launch N=5 new MD replicas from the
existing apo equilibrated state with fresh random seeds. Pull
trajectories. Featurize + project onto the v0.5 VAMPnet. Report
state occupancy.

Phase 2 (v0.7): seed bursts from AlphaFlow state-1 mean structure
via a Cα-morph + side-chain re-equilibration pipeline.

Run:
  $ .venv/bin/python examples/live_adaptive_sampling_notch1.py \\
        --replicas 5 --ns 5

Mid-level orchestration (mirrors what an MCP agent would do):
  1. POST /tools/vampnet_fit on the v0.5 5-source ensemble.
  2. AGENT inspects per-state populations, identifies state 1 +
     state 2 as MD-inaccessible.
  3. AGENT POSTs to the modal_md::fanout entrypoint to launch
     `replicas` extra MD replicas of the existing v0.3 apo system.
  4. AGENT polls modal volume for each replica's traj.dcd.
  5. AGENT extracts CAs from new DCDs.
  6. AGENT re-projects via the v0.5 VAMPnet and reports occupancy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "md"))

SYSTEM = "notch1_apo_v3"
ZEN_ROOT = Path("/data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/"
                 "v0.3/notch1_apo_v3")


# ----------------------------------------------------------------------
# Phase 0: load v0.5 5-source ensemble and fit a fresh joint VAMPnet to
# get state assignments we can score new bursts against.
# ----------------------------------------------------------------------

NEC_CA_RANGE = slice(0, 174)


def _load_md_apo_nec_ca():
    cas = []
    for r in range(3):
        d = np.load(ZEN_ROOT / f"replica_{r}/ca_traj.npz")
        coords = d["coords_A"]
        chains = d["chains"].astype("U2")
        cas.append(coords[:, chains == "A", :])
    return np.concatenate(cas, axis=0)


def _load_source(name, path):
    p = ROOT / path
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "coords_ca" not in d.files:
        return None
    ca = d["coords_ca"]
    if ca.shape[1] != 174:
        return None
    return ca.astype(np.float32)


def _featurize(coords, pair_idx=None, max_pairs=500, rng_seed=0,
                 mu=None, sigma=None):
    N, A, _ = coords.shape
    if pair_idx is None:
        iu, ju = np.triu_indices(A, k=1)
        rng = np.random.default_rng(rng_seed)
        sel = rng.choice(len(iu), size=min(max_pairs, len(iu)), replace=False)
        pair_idx = np.stack([iu[sel], ju[sel]], axis=1)
    a = coords[:, pair_idx[:, 0], :]
    b = coords[:, pair_idx[:, 1], :]
    raw = np.sqrt(((a - b) ** 2).sum(-1)).astype(np.float32)
    if mu is None:
        mu = raw.mean(0, keepdims=True)
    if sigma is None:
        sigma = raw.std(0, keepdims=True) + 1e-3
    z = ((raw - mu) / sigma).clip(-5.0, 5.0)
    return z, pair_idx, mu, sigma


def _fit_vampnet(X, n_states=4, lag=20, epochs=60):
    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    torch.manual_seed(0)
    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, n_states), nn.Softmax(dim=-1),
    )
    ds = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True,
                                          drop_last=True)
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu", epsilon=1e-3)
    return net.fit(loader, n_epochs=epochs).fetch_model()


# ----------------------------------------------------------------------
# Modal interactions.
# ----------------------------------------------------------------------

def _launch_replica(replica_idx: int, ns: float) -> bool:
    cmd = [
        "modal", "run", "--detach",
        str(ROOT / "md" / "modal_md.py::fanout"),
        "--system", SYSTEM,
        "--replicas", "1",
        "--start-replica", str(replica_idx),
        "--ns", str(ns),
        "--dcd-interval-ps", "20",
    ]
    print(f"  AGENT: launching replica {replica_idx} ({ns} ns)")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        for line in r.stderr.splitlines()[-5:]:
            print(f"    [stderr] {line}")
    return r.returncode == 0


def _poll_replica(replica_idx: int, expected_min_mib: float = 100,
                    timeout_s: int = 3600) -> bool:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        result = subprocess.run(
            ["modal", "volume", "ls", "--json", "chimerax-vampnet-md",
             f"prepared/{SYSTEM}/replica_{replica_idx}/traj.dcd"],
            capture_output=True, text=True,
        )
        try:
            files = json.loads(result.stdout)
            sz_str = files[0]["Size"] if files else "0 B"
        except Exception:
            sz_str = "?"
        m = re.match(r"([\d.]+)\s*([KMGT])iB", sz_str or "")
        mib = (float(m.group(1)) * {"K": 1/1024, "M": 1.0,
                                       "G": 1024.0, "T": 1048576.0}[m.group(2)]
               if m else 0.0)
        if mib != last:
            print(f"    [{time.strftime('%H:%M:%S')}] replica {replica_idx}: {sz_str}")
            last = mib
        if mib >= expected_min_mib:
            time.sleep(20)
            return True
        time.sleep(60)
    return False


def _pull_replica(replica_idx: int) -> Path:
    dst = Path(f"/tmp/adaptive_notch1_replica_{replica_idx}.dcd")
    subprocess.run(
        ["modal", "volume", "get", "chimerax-vampnet-md",
         f"prepared/{SYSTEM}/replica_{replica_idx}/traj.dcd",
         str(dst), "--force"],
        check=True, capture_output=True, text=True,
    )
    return dst


def _extract_nec_cas(dcd_path: Path) -> np.ndarray:
    """Re-extract chain A NEC CAs from a downloaded DCD."""
    import mdtraj as md
    eq_pdb = "/data/datasets/chimerax-vampnet/notch1_modal/notch1_apo/equilibrated.pdb"
    traj = md.load(str(dcd_path), top=eq_pdb)
    ca_chainA = traj.topology.select("name CA and chainid 0")
    if len(ca_chainA) != 174:
        ca_chainA = traj.topology.select("name CA and chainid 0")[:174]
    return traj.xyz[:, ca_chainA, :] * 10.0  # nm -> A


# ----------------------------------------------------------------------
# Main loop.
# ----------------------------------------------------------------------

def main(replicas: int, ns: float, start_replica: int):
    print("=" * 70)
    print(f"Live MCP-driven adaptive sampling on Notch1 NRR (v0.6 Phase 1)")
    print(f"  System: {SYSTEM},  +{replicas} replicas × {ns} ns each")
    print("=" * 70)

    # Build the v0.5 5-source joint VAMPnet (model lives in memory only;
    # we use it to score the new MD bursts).
    print("\nPHASE 0 — refit v0.5 5-source joint VAMPnet")
    print("-" * 70)
    sources = {}
    sources["MD"] = _load_md_apo_nec_ca()
    for tag, path in [("MarS-FM", "notch1_apo_NEC200_marsfm.npz"),
                       ("BioEmu", "notch1_NEC_bioemu200.npz"),
                       ("Boltz-2", "notch1_NEC_boltz200.npz"),
                       ("AlphaFlow", "notch1_NEC_af200.npz")]:
        s = _load_source(tag, path)
        if s is not None:
            sources[tag] = s
    src_names = list(sources.keys())
    all_coords = np.concatenate([sources[s] for s in src_names], axis=0)
    print(f"  ensemble: {all_coords.shape[0]} frames, {len(src_names)} sources")

    X_train, pair_idx, mu, sigma = _featurize(all_coords)
    print("  fitting joint VAMPnet (4 states, lag 20, ~1 min on CPU)…")
    model = _fit_vampnet(X_train, n_states=4, lag=20)

    # Baseline state populations from v0.5 (sanity check).
    soft = np.asarray(model.transform(X_train.astype("float32")))
    hard_base = soft.argmax(axis=-1)
    print(f"  baseline state populations: " + "  ".join(
        f"s{s}={(hard_base==s).mean()*100:5.1f}%" for s in range(4)
    ))

    # PHASE 1: launch N new MD replicas with fresh random seeds.
    print(f"\nPHASE 1 — launch {replicas} new MD replicas with fresh seeds")
    print("-" * 70)
    print("  AGENT: posting to modal_md::fanout for additional replicas")
    new_indices = list(range(start_replica, start_replica + replicas))
    launched = []
    for r in new_indices:
        if _launch_replica(r, ns):
            launched.append(r)
    if not launched:
        print("  no replicas launched; aborting")
        return
    print(f"  launched {len(launched)} replicas: {launched}")

    # Poll for each replica's traj.dcd.
    expected_mib = 30  # ~5 ns at 20 ps stride writes ~30 MiB for NRR-sized
    print(f"\nPHASE 2 — poll for replica completion (expecting ≥{expected_mib} MiB / replica)")
    print("-" * 70)
    arrived = []
    for r in launched:
        if _poll_replica(r, expected_min_mib=expected_mib):
            arrived.append(r)
        else:
            print(f"  replica {r} timed out")

    # Pull, extract, score.
    print(f"\nPHASE 3 — pull + featurize + project onto v0.5 VAMPnet")
    print("-" * 70)
    per_replica = {}
    for r in arrived:
        try:
            dcd = _pull_replica(r)
            ca = _extract_nec_cas(dcd)
            print(f"  replica {r}: {ca.shape[0]} frames extracted, "
                  f"{ca.shape[1]} CAs")
            X, _, _, _ = _featurize(ca, pair_idx=pair_idx, mu=mu, sigma=sigma)
            soft = np.asarray(model.transform(X.astype("float32")))
            hard = soft.argmax(axis=-1)
            per_replica[r] = {
                "n_frames": int(ca.shape[0]),
                "state_occupancy": {f"s{s}": float((hard == s).mean())
                                       for s in range(4)},
            }
            print(f"    state occupancy: " + "  ".join(
                f"s{s}={(hard==s).mean()*100:5.1f}%" for s in range(4)
            ))
        except Exception as e:
            print(f"  replica {r}: failed -- {e}")

    # Aggregate.
    print("\n" + "=" * 70)
    print("RESULT — does extra MD reach state 1 or state 2?")
    print("=" * 70)
    total_frames = sum(d["n_frames"] for d in per_replica.values())
    if total_frames == 0:
        print("  no frames retrieved")
        return
    aggregated = {f"s{s}": 0 for s in range(4)}
    for d in per_replica.values():
        for k, v in d["state_occupancy"].items():
            aggregated[k] += v * d["n_frames"]
    aggregated = {k: v / total_frames for k, v in aggregated.items()}
    print(f"  total burst frames: {total_frames}")
    print(f"  aggregate occupancy: " + "  ".join(
        f"{k}={v*100:5.1f}%" for k, v in aggregated.items()
    ))
    # Save JSON summary.
    out = ROOT / "md" / "notch1_adaptive_v0p6_results.json"
    out.write_text(json.dumps({
        "phase": "v0.6_phase_1_random_seed",
        "system": SYSTEM,
        "replicas_launched": new_indices,
        "replicas_arrived": arrived,
        "ns_per_replica": ns,
        "per_replica": per_replica,
        "aggregate_occupancy": aggregated,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replicas", type=int, default=5)
    p.add_argument("--ns", type=float, default=5.0)
    p.add_argument("--start-replica", type=int, default=3,
                    help="first new replica index (existing v0.3 used 0/1/2)")
    args = p.parse_args()
    main(args.replicas, args.ns, args.start_replica)
