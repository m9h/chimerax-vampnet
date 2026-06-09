"""Multi-chain fanout utilities for chimerax-vampnet adapter ensembles.

Bridges the v0.10 W1 diagnostics (md/mcmc_diagnostics.py) to multi-chain
analysis by adding a `chain_idx` field to ensemble npz files. With ≥2
chains present, arviz can compute R-hat (Gelman-Rubin) — the canonical
between-chain convergence diagnostic.

Two operating modes, one CLI:

  1. MERGE — combine K independent single-chain npz files (produced by
     parallel runs of an adapter's `sample` entrypoint with different
     seeds or different conditioning frames) into one chain_idx-labeled
     npz:

       python md/multichain.py merge \\
           ad2_timewarp_seed{0,1,2,3}.npz --out ad2_timewarp_4chains.npz

  2. CHUNK — split a single existing npz into K pseudo-chains by
     contiguous-block segmentation. This is *not* a real multi-chain
     analysis (the same generator produced all K blocks, so R-hat from
     this will be optimistic) — but it does back-apply chain semantics
     to already-collected data, useful when the original adapter run
     wasn't fanned out:

       python md/multichain.py chunk \\
           notch1_NEC_esmfold2200.npz --n-chunks 4 \\
           --out notch1_NEC_esmfold2_4chunks.npz

  3. PIPE TO DIAGNOSTICS — the output of either mode is consumed by
     md/mcmc_diagnostics.py without any further argument:

       python md/multichain.py chunk X.npz --n-chunks 4 --out X_4c.npz
       python md/mcmc_diagnostics.py X_4c.npz   # now shows R-hat

What "chain" means per adapter:

  Timewarp (MCMC)      — independent MH chains seeded from different
                         conditioning frames; merging real fanned-out
                         runs is the canonical use.
  AlphaFlow/BioEmu/
  Boltz-2/MarS-FM/
  ESMFold2 (iid)       — separate batches of independent draws. Across
                         batches, R-hat measures whether the generator
                         is reproducibly reaching the same distribution
                         — a non-trivial check for diffusion samplers
                         with different RNG seeds.
  UMA (MD trajectory)  — independent replicate trajectories from
                         different initial velocities, in the spirit of
                         the existing md/modal_md.py:fanout pattern.

A future v0.10 follow-up wires Modal-fanout directly into each adapter's
sample() entrypoint with an `n_chains` keyword that calls .spawn() K
times then merges; this file is the merge engine and the diagnostic-side
chain pseudo-split, both of which are immediately useful with already-
collected v0.7/v0.8/v0.9 ensembles.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np


# Keys whose first axis is the sample axis; the per-chain copies are
# concatenated along axis=0. Everything else (seqres, chain_id of
# residues, accept_rate, elapsed_seconds, dt_fs, temperature_K, ...) is
# metadata: take the chain-0 value (or warn on disagreement).
_SAMPLE_AXIS_KEYS = {"coords", "coords_ca", "plddt", "iptm", "energy"}


def _load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def merge(chain_paths: list[str | Path],
          out_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Merge K single-chain npz files into one chain_idx-labeled dict.

    Per-sample arrays (coords, coords_ca, plddt, iptm, energy) are
    concatenated along axis=0. Metadata fields (seqres, accept_rate,
    elapsed_seconds, ...) are taken from chain 0 with a stderr warning
    if any chain disagrees. A new `chain_idx` array (n_total,) labels
    each sample with its source-chain index.
    """
    if len(chain_paths) < 1:
        raise ValueError("merge needs at least 1 npz path")
    dicts = [_load_npz_dict(p) for p in chain_paths]
    if "coords_ca" not in dicts[0]:
        raise KeyError(f"{chain_paths[0]} has no coords_ca — not an adapter output?")

    out: dict[str, np.ndarray] = {}

    # Per-sample concatenation.
    for k in dicts[0]:
        if k in _SAMPLE_AXIS_KEYS:
            arrs = [cd[k] for cd in dicts if k in cd]
            if len(arrs) != len(dicts):
                print(f"[merge] warn: {k} missing in {len(dicts)-len(arrs)} chains; "
                      "skipping field", file=sys.stderr)
                continue
            out[k] = np.concatenate(arrs, axis=0)

    # chain_idx — built from coords_ca sample counts (the canonical
    # per-sample axis present in every adapter).
    chain_lens = [cd["coords_ca"].shape[0] for cd in dicts]
    chain_idx = np.concatenate(
        [np.full(n, i, dtype=np.int64) for i, n in enumerate(chain_lens)]
    )
    out["chain_idx"] = chain_idx

    # Metadata — pick chain 0, warn on disagreement.
    for k in dicts[0]:
        if k in _SAMPLE_AXIS_KEYS or k in out:
            continue
        out[k] = dicts[0][k]
        for j, cd in enumerate(dicts[1:], start=1):
            if k in cd and not _array_equal_lenient(cd[k], dicts[0][k]):
                print(f"[merge] warn: metadata key {k!r} differs in chain {j} "
                      f"(taking chain 0's value)", file=sys.stderr)

    print(f"[merge] merged {len(dicts)} chains -> total {int(chain_lens[0])} per "
          f"chain × {len(dicts)} = {chain_idx.size} samples")

    if out_path is not None:
        np.savez_compressed(str(out_path), **out)
        print(f"[merge] wrote {out_path}")
    return out


def chunk(in_path: str | Path, n_chunks: int,
          out_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Split a single-chain npz into K contiguous pseudo-chains.

    Adds a chain_idx field but does NOT change any per-sample data. The
    K pseudo-chains all come from the same generator so R-hat from a
    chunked output is optimistically biased; use it as a sanity proxy,
    not as a substitute for real multi-chain runs. The diagnostic report
    on a chunked file is most useful for catching adapter outputs that
    are sequentially correlated (e.g. an MD trajectory misrepresented as
    iid samples), where R-hat across contiguous chunks would spike.
    """
    d = _load_npz_dict(in_path)
    if "coords_ca" not in d:
        raise KeyError(f"{in_path} has no coords_ca — not an adapter output?")
    n = d["coords_ca"].shape[0]
    if n_chunks < 1 or n_chunks > n:
        raise ValueError(f"n_chunks={n_chunks} must be in 1..{n}")

    # Contiguous-block chunking — keep adjacent samples in the same
    # "chain" so any temporal correlation is preserved within-chain and
    # exposed cross-chain. Round-robin assignment would mask exactly the
    # pathology we want to surface.
    edges = np.linspace(0, n, n_chunks + 1, dtype=int)
    chain_idx = np.zeros(n, dtype=np.int64)
    for i in range(n_chunks):
        chain_idx[edges[i]:edges[i + 1]] = i

    out = dict(d)
    out["chain_idx"] = chain_idx

    print(f"[chunk] split {n} samples into {n_chunks} contiguous pseudo-chains "
          f"({[int(edges[i+1]-edges[i]) for i in range(n_chunks)]} each)")
    if out_path is not None:
        np.savez_compressed(str(out_path), **out)
        print(f"[chunk] wrote {out_path}")
    return out


def _array_equal_lenient(a: np.ndarray, b: np.ndarray) -> bool:
    """Element-equality that tolerates scalar / 0-d arrays and NaN."""
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("merge", help="merge K single-chain npz files")
    pm.add_argument("paths", nargs="+",
                     help="One or more single-chain adapter npz files")
    pm.add_argument("--out", required=True, help="Output merged npz")

    pc = sub.add_parser("chunk", help="split one npz into K pseudo-chains")
    pc.add_argument("path", help="Input single-chain adapter npz")
    pc.add_argument("--n-chunks", type=int, required=True,
                     help="Number of pseudo-chains to split into")
    pc.add_argument("--out", required=True, help="Output chunked npz")

    args = p.parse_args(argv)

    if args.cmd == "merge":
        merge(args.paths, out_path=args.out)
    elif args.cmd == "chunk":
        chunk(args.path, n_chunks=args.n_chunks, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
