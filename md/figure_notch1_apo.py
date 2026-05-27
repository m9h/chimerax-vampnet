"""Figure for the chimerax-vampnet paper draft: Notch1 NRR apo
activation-direction order parameter (NEC-NTM COM separation) across
3 replicas of 100 ns MD.

This is the one valid Tier-2 deliverable from the May 26 run: it shows
the bundle's analysis pipeline running on a real 350-residue protein
trajectory and quantifying the rate-limiting motion of Notch activation
(NEC-NTM fragment separation -> S2 cleavage site exposure).

Output: figure_notch1_apo.png (300 DPI, 3-panel)

  $ .venv/bin/python md/figure_notch1_apo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier1_vampnet import _read_dcd  # noqa: E402

DATA_ROOT = Path("/data/datasets/chimerax-vampnet/notch1_modal/notch1_apo")
OUT = Path(__file__).resolve().parent / "figure_notch1_apo.png"
DCD_STRIDE_PS = 20.0   # produce_remote default for Notch1


def _parse_ca(pdb_path: Path):
    cas_per_chain: dict[str, list[int]] = {}
    ca_pos = 0
    n_atoms = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[12:16].strip() == "CA":
                    cas_per_chain.setdefault(line[21:22], []).append(ca_pos)
                    ca_pos += 1
                n_atoms += 1
    return cas_per_chain, n_atoms


def _com_separation(coords, idx_a, idx_b):
    return np.linalg.norm(coords[:, idx_a, :].mean(1) - coords[:, idx_b, :].mean(1), axis=-1)


def main():
    cas_per_chain, n_atoms = _parse_ca(DATA_ROOT / "equilibrated.pdb")
    nec_idx = np.array(cas_per_chain["A"], dtype=np.int64)   # 174 CAs (NEC)
    ntm_idx = np.array(cas_per_chain["B"], dtype=np.int64)   # 60 CAs (NTM)

    per_replica_sep = []
    for r in range(3):
        coords = _read_dcd(DATA_ROOT / f"replica_{r}" / "traj.dcd", n_atoms=n_atoms)
        ca_coords = coords[:, np.concatenate([nec_idx, ntm_idx]), :]
        sep = _com_separation(coords, nec_idx, ntm_idx)
        time_ns = np.arange(len(sep)) * DCD_STRIDE_PS / 1000.0
        per_replica_sep.append((time_ns, sep))
        print(f"replica {r}: {len(sep)} frames, mean sep {sep.mean():.1f} +/- {sep.std():.1f} A, "
              f"P(sep>20A)={ (sep>20).mean()*100:.1f}%, P(sep>50A)={(sep>50).mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for r, (t, sep) in enumerate(per_replica_sep):
        axes[0].plot(t, sep, color=colors[r], lw=0.6, alpha=0.85, label=f"replica {r}")
    axes[0].axhline(20.0, color="k", ls="--", lw=0.5, label="20 A: S2 exposing")
    axes[0].axhline(50.0, color="r", ls="--", lw=0.5, label="50 A: dissociated")
    axes[0].set_xlabel("time (ns)")
    axes[0].set_ylabel("NEC-NTM COM separation (Å)")
    axes[0].set_title("Activation-direction order parameter")
    axes[0].legend(fontsize=8, loc="upper left")

    all_sep = np.concatenate([sep for _, sep in per_replica_sep])
    axes[1].hist(all_sep, bins=60, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.5)
    axes[1].axvline(all_sep.mean(), color="k", ls="--", lw=1.0, label=f"mean {all_sep.mean():.1f} Å")
    axes[1].set_xlabel("NEC-NTM COM separation (Å)")
    axes[1].set_ylabel("frame count")
    axes[1].set_title(f"Distribution across 3 replicas\n(N = {len(all_sep)} frames, 303 ns total)")
    axes[1].legend(fontsize=9)

    pcts = []
    thresholds = np.linspace(10, 80, 30)
    for thr in thresholds:
        pcts.append((all_sep > thr).mean() * 100)
    axes[2].plot(thresholds, pcts, color="#d62728", lw=1.6)
    axes[2].axvline(20, color="k", ls=":", lw=0.6, alpha=0.7)
    axes[2].axvline(50, color="r", ls=":", lw=0.6, alpha=0.7)
    axes[2].set_xlabel("threshold (Å)")
    axes[2].set_ylabel("% frames with separation > threshold")
    axes[2].set_title("Cumulative distribution")
    axes[2].set_ylim(-2, 100)

    fig.suptitle("Notch1 NRR apo: NEC–NTM dissociation across 3×100 ns unbiased MD\n"
                  "(no membrane / disulfide constraints; in vivo would suppress separation > 20 Å)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\nfigure -> {OUT}")
    print(f"  total frames: {len(all_sep)}")
    print(f"  total sim time: {len(all_sep) * DCD_STRIDE_PS / 1000:.1f} ns")
    print(f"  mean separation: {all_sep.mean():.1f} A")
    print(f"  std separation: {all_sep.std():.1f} A")
    print(f"  P(>20 A): {(all_sep > 20).mean()*100:.1f}%")
    print(f"  P(>50 A): {(all_sep > 50).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
