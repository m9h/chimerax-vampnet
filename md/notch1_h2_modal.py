"""Run the v0.2 headline analysis on Modal against the corrected Notch1
NRR apo + holo trajectories that already live on the `chimerax-vampnet-md`
volume. Avoids pulling ~40 GiB of holo DCDs to the local box.

Inlines the tier2_notch1.py analysis so the Modal function is
self-contained.

  modal run notch1_h2_modal.py::analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-notch1-h2"
VOLUME_NAME = "chimerax-vampnet-md"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "scipy",
        "torch==2.4.1",
        "deeptime",
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME)


# --- DCD parser (lifted from md/tier1_vampnet.py:_read_dcd, openmm-format) ---
def _read_dcd(path, n_atoms):
    import struct
    import numpy as np
    with open(path, "rb") as f:
        data = f.read()
    pos = 4 + 4
    header_ints = struct.unpack("<20i", data[pos:pos + 80])
    has_cell = header_ints[10] != 0
    pos = 0
    pos += 4 + 4 + 80 + 4
    blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
    pos += 4 + blocksize + 4
    pos += 4
    pos += 4 + 4
    frames = []
    while pos + 4 <= len(data):
        if has_cell:
            blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
            pos += 4 + blocksize + 4
        try:
            xyz = []
            for _ in range(3):
                blocksize = struct.unpack("<i", data[pos:pos + 4])[0]
                pos += 4
                vals = np.frombuffer(data[pos:pos + blocksize], dtype="<f4")
                pos += blocksize + 4
                xyz.append(vals)
            frame = np.stack(xyz, axis=-1)
            if frame.shape[0] != n_atoms:
                break
            frames.append(frame)
        except Exception:
            break
    return np.stack(frames, axis=0).astype(np.float32)


def _ca_indices_and_chains(pdb_path):
    import numpy as np
    ca_pos_by_chain = {}
    cas = []
    all_idx = 0
    ca_pos = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas.append(all_idx)
                    chain = line[21:22]
                    ca_pos_by_chain.setdefault(chain, []).append(ca_pos)
                    ca_pos += 1
                all_idx += 1
    return (np.array(cas, dtype=np.int64), all_idx,
            {c: np.array(v, dtype=np.int64) for c, v in ca_pos_by_chain.items()})


def _featurize_ca_dist(coords, pair_idx):
    """Return per-pair CA-CA distances, standardized + clipped to mitigate
    outlier-driven ill-conditioning during VAMPnet training."""
    import numpy as np
    a = coords[:, pair_idx[:, 0], :]
    b = coords[:, pair_idx[:, 1], :]
    raw = np.sqrt(((a - b) ** 2).sum(-1)).astype(np.float32)
    mu = raw.mean(0, keepdims=True)
    sigma = raw.std(0, keepdims=True) + 1e-3
    z = (raw - mu) / sigma
    # Clip extreme outliers (membrane-detached frames blow up some pairs).
    return np.clip(z, -5.0, 5.0).astype(np.float32)


def _fit_vampnet(X, n_states, lag, epochs, epsilon=1e-3):
    import numpy as np
    import torch
    import torch.nn as nn
    from deeptime.decomposition.deep import VAMPNet
    from deeptime.util.data import TrajectoryDataset
    from deeptime.markov.msm import MaximumLikelihoodMSM
    torch.manual_seed(0)
    P = X.shape[1]
    lobe = nn.Sequential(
        nn.Linear(P, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, n_states), nn.Softmax(dim=-1),
    )
    dataset = TrajectoryDataset(lagtime=lag, trajectory=X.astype("float32"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    # epsilon regularises the Koopman-matrix eigendecomposition; the default
    # 1e-6 trips on near-singular covariances from highly-correlated CA
    # distances on large proteins. 1e-3 is the deeptime tutorials' value for
    # similar size systems.
    net = VAMPNet(lobe=lobe, learning_rate=5e-4, device="cpu",
                  epsilon=epsilon)
    model = net.fit(loader, n_epochs=epochs).fetch_model()
    soft = np.asarray(model.transform(X.astype("float32")))
    hard = soft.argmax(axis=-1)
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=lag).fit_fetch(hard)
    return hard, msm


def _analyze_one(system_dir, n_replicas, frame_stride, n_states, lag, epochs,
                  ps_per_dcd_frame):
    import numpy as np
    print(f"\n=== {Path(system_dir).name} ===")
    eq_pdb = Path(system_dir) / "equilibrated.pdb"
    ca_idx, n_atoms_total, chain_ca_pos = _ca_indices_and_chains(eq_pdb)
    print(f"  topology: {n_atoms_total} atoms, {len(ca_idx)} CAs, "
          f"chains: " + ", ".join(f"{c}({len(v)})" for c, v in chain_ca_pos.items()))

    all_ca = []
    for r in range(n_replicas):
        dcd = Path(system_dir) / f"replica_{r}" / "traj.dcd"
        if not dcd.exists():
            print(f"  [warn] missing {dcd}")
            continue
        print(f"  reading {dcd} ...")
        coords = _read_dcd(str(dcd), n_atoms=n_atoms_total)
        coords = coords[::frame_stride]
        ca = coords[:, ca_idx, :]
        print(f"    replica {r}: {coords.shape[0]} frames (stride {frame_stride})")
        all_ca.append(ca)
    nrr_all = np.concatenate(all_ca, axis=0)
    print(f"  combined: {nrr_all.shape}")

    # Identify NRR fragments.
    # Apo: chains A (NEC, ~174 CAs) + B (NTM, ~60 CAs).
    # Holo: NEC+NTM are the two smallest-but-not-tiny chains (Fab heavy ~220, light ~110).
    chains_sorted = sorted(chain_ca_pos.items(), key=lambda kv: len(kv[1]))
    if "A" in chain_ca_pos and "B" in chain_ca_pos and len(chain_ca_pos["A"]) > len(chain_ca_pos.get("B", [])):
        nec_pos = chain_ca_pos["A"]
        ntm_pos = chain_ca_pos["B"]
    else:
        small = [c for c, idx in chains_sorted if 40 <= len(idx) <= 200]
        if len(small) >= 2:
            nec_pos = chain_ca_pos[small[-1]]
            ntm_pos = chain_ca_pos[small[0]]
        else:
            nec_pos = chains_sorted[0][1]
            ntm_pos = chains_sorted[1][1] if len(chains_sorted) > 1 else np.array([], dtype=np.int64)
    print(f"  NRR fragments: NEC ({len(nec_pos)} CAs) + NTM ({len(ntm_pos)} CAs)")

    nec = nrr_all[:, nec_pos, :]
    ntm = nrr_all[:, ntm_pos, :]
    com_sep = np.linalg.norm(nec.mean(1) - ntm.mean(1), axis=-1)
    print(f"  NEC-NTM COM sep: mean {com_sep.mean():.1f} +/- {com_sep.std():.1f} A, "
          f"range {com_sep.min():.1f}-{com_sep.max():.1f}")

    # Feature matrix: CA-CA distances on the NRR fragment (subsample to 2000 pairs).
    nrr_only = nrr_all[:, np.concatenate([nec_pos, ntm_pos]), :]
    A = nrr_only.shape[1]
    total_pairs = A * (A - 1) // 2
    # 500 pairs ~ keeps feature/sample ratio well-conditioned at ~3k frames.
    max_pairs = 500
    iu, ju = np.triu_indices(A, k=1)
    if total_pairs > max_pairs:
        rng = np.random.default_rng(0)
        sel = rng.choice(total_pairs, size=max_pairs, replace=False)
        pair_idx = np.stack([iu[sel], ju[sel]], axis=1)
    else:
        pair_idx = np.stack([iu, ju], axis=1)
    X = _featurize_ca_dist(nrr_only, pair_idx)
    print(f"  feature matrix: {X.shape}")

    hard, msm = _fit_vampnet(X, n_states, lag, epochs)
    pops = [(hard == s).mean() for s in range(n_states)]
    try:
        its_frames = msm.timescales(k=n_states - 1)
        its_ns = [float(t) * ps_per_dcd_frame * frame_stride / 1000.0
                  for t in its_frames]
    except Exception as e:
        its_ns = []
        print(f"  [warn] implied timescales unavailable: {e}")
    stationary = [float(p) for p in msm.stationary_distribution]

    # Per-state COM separation -> label states by NEC-NTM proximity.
    state_sep_mean = []
    state_sep_std = []
    for s in range(n_states):
        m = hard == s
        if m.any():
            state_sep_mean.append(float(com_sep[m].mean()))
            state_sep_std.append(float(com_sep[m].std()))
        else:
            state_sep_mean.append(float("nan"))
            state_sep_std.append(float("nan"))

    # Auto-inhibited = smallest mean NEC-NTM separation; activated = largest.
    finite_states = [s for s in range(n_states)
                     if state_sep_mean[s] == state_sep_mean[s]]  # not NaN
    autoinh_state = min(finite_states, key=lambda s: state_sep_mean[s])
    active_state = max(finite_states, key=lambda s: state_sep_mean[s])

    p_autoinh = float(pops[autoinh_state])
    p_active = float(pops[active_state])

    print(f"  state populations: " + ", ".join(f"{p*100:5.1f}%" for p in pops))
    print(f"  implied timescales: " + (", ".join(f"{t:.1f} ns" for t in its_ns) if its_ns else "(insufficient)"))
    print(f"  per-state NEC-NTM COM separation:")
    for s in range(n_states):
        tag = ""
        if s == autoinh_state:
            tag = "  <- auto-inhibited"
        elif s == active_state:
            tag = "  <- activated"
        print(f"    state {s}: pop {pops[s]*100:5.1f}%  sep {state_sep_mean[s]:5.1f} +/- {state_sep_std[s]:4.1f} A{tag}")

    return {
        "system": Path(system_dir).name,
        "n_frames_subsampled": int(nrr_all.shape[0]),
        "frame_stride": int(frame_stride),
        "n_cas_nec": int(len(nec_pos)),
        "n_cas_ntm": int(len(ntm_pos)),
        "com_sep_overall_mean_A": float(com_sep.mean()),
        "com_sep_overall_std_A":  float(com_sep.std()),
        "com_sep_p_gt_20A":  float((com_sep > 20).mean()),
        "com_sep_p_gt_50A":  float((com_sep > 50).mean()),
        "state_populations": [float(p) for p in pops],
        "state_sep_mean_A":  state_sep_mean,
        "state_sep_std_A":   state_sep_std,
        "stationary_distribution": stationary,
        "implied_timescales_ns": its_ns,
        "auto_inhibited_state": int(autoinh_state),
        "activated_state":      int(active_state),
        "P_auto_inhibited":     p_autoinh,
        "P_activated":          p_active,
    }


@app.function(cpu=8, memory=49152, timeout=3600, volumes={"/vol": vol})
def analyze_remote(apo_system: str = "notch1_apo",
                    holo_system: str = "notch1_holo_diag",
                    n_states: int = 4, lag: int = 50, epochs: int = 60,
                    frame_stride: int = 5, n_replicas: int = 3,
                    ps_per_dcd_frame: float = 20.0):
    """Run the H2 apo-vs-holo analysis on the Modal volume."""
    base = Path("/vol/prepared")
    apo  = _analyze_one(base / apo_system,  n_replicas, frame_stride,
                        n_states, lag, epochs, ps_per_dcd_frame)
    holo = _analyze_one(base / holo_system, n_replicas, frame_stride,
                        n_states, lag, epochs, ps_per_dcd_frame)

    # H2 verdict.
    delta = apo["P_auto_inhibited"] - holo["P_auto_inhibited"]
    h2_predicted = (apo["P_auto_inhibited"] >= 0.5 and
                    holo["P_auto_inhibited"] <= 0.3)
    h2_directional = delta > 0  # apo more auto-inhibited than holo

    print("\n" + "=" * 70)
    print("H2 RESULT (pre-registered: apo P_auto-inhibited >= 0.5, holo <= 0.3)")
    print("=" * 70)
    print(f"  apo  P(auto-inhibited) = {apo['P_auto_inhibited']*100:5.1f}%   "
          f"(state {apo['auto_inhibited_state']}, sep {apo['state_sep_mean_A'][apo['auto_inhibited_state']]:.1f} A)")
    print(f"  holo P(auto-inhibited) = {holo['P_auto_inhibited']*100:5.1f}%   "
          f"(state {holo['auto_inhibited_state']}, sep {holo['state_sep_mean_A'][holo['auto_inhibited_state']]:.1f} A)")
    print(f"  delta apo - holo       = {delta*100:+.1f} pp")
    print(f"  pre-registered H2 met: {h2_predicted}")
    print(f"  directional (apo > holo auto-inhibited): {h2_directional}")

    return {
        "apo":  apo,
        "holo": holo,
        "H2": {
            "delta_p_autoinh": float(delta),
            "preregistered_met": bool(h2_predicted),
            "directional_apo_more_autoinh": bool(h2_directional),
            "preregistration": {
                "apo_P_auto_inhibited_ge": 0.5,
                "holo_P_auto_inhibited_le": 0.3,
            },
        },
        "hyperparams": {
            "n_states": n_states, "lag_frames": lag, "epochs": epochs,
            "frame_stride": frame_stride, "n_replicas": n_replicas,
            "ps_per_dcd_frame": ps_per_dcd_frame,
        },
    }


@app.local_entrypoint()
def analyze(apo_system: str = "notch1_apo",
            holo_system: str = "notch1_holo_diag",
            n_states: int = 4, lag: int = 50, epochs: int = 60,
            out: str = "notch1_h2_results.json"):
    result = analyze_remote.remote(
        apo_system=apo_system, holo_system=holo_system,
        n_states=n_states, lag=lag, epochs=epochs)
    out_path = Path(__file__).parent / out
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[local] wrote {out_path}")
