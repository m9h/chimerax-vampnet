"""Fetcher for ATLAS protein MD trajectories.

ATLAS (Vander Meersche, Cretin et al. 2024) is a public database of
1-3 us MD trajectories for ~1700 protein chains, with 3 independent
replicas each at 100 ns post-equilibration. Hosted at:

  https://www.dsimb.inserm.fr/ATLAS/

This script wraps the protein-only download endpoint:

  GET https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/{pdb_chain}

returns a ~600 MB zip with the reference structure (.pdb) + 3
GROMACS XTC trajectories (10,000 frames each, protein atoms only).

Usage:

  python md/atlas_fetch.py 1k5n_A /data/datasets/chimerax-vampnet/atlas/
  python md/atlas_fetch.py 1k5n_A /data/.../atlas/ --replicas 1,2,3

The output directory layout mirrors the bundle's own MD pipeline:

  <out_root>/atlas_<pdb_chain>/
    reference.pdb         (renamed for clarity)
    replica_0/traj.xtc    (renamed from *_prod_R1_fit.xtc)
    replica_1/traj.xtc
    replica_2/traj.xtc
    metadata.json         (full ATLAS metadata dict)
    README.txt            (ATLAS-provided notes)

Once landed, load into the bundle with:

  open <out_root>/atlas_<pdb_chain>/reference.pdb
  vampnet load_ensemble md <out_root>/atlas_<pdb_chain>/replica_0/traj.xtc

The reference PDB starts at the post-equilibration structure;
ChimeraX's coordset machinery handles XTC natively via mdtraj.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ATLAS_BASE = "https://www.dsimb.inserm.fr/ATLAS/api"


def fetch_metadata(pdb_chain: str) -> dict:
    url = f"{ATLAS_BASE}/ATLAS/metadata/{pdb_chain}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get(pdb_chain, data)


def fetch_protein_archive(pdb_chain: str, dst_zip: Path) -> int:
    """Download the protein-only trajectory zip for one ATLAS entry.

    Returns total bytes downloaded.
    """
    url = f"{ATLAS_BASE}/ATLAS/protein/{pdb_chain}"
    print(f"[atlas] GET {url}")
    sys.stdout.flush()
    total = 0
    chunk_count = 0
    with urllib.request.urlopen(url, timeout=120) as r:
        with dst_zip.open("wb") as f:
            while True:
                buf = r.read(1 << 20)  # 1 MB chunks
                if not buf:
                    break
                f.write(buf)
                total += len(buf)
                chunk_count += 1
                if chunk_count % 50 == 0:
                    print(f"[atlas]   {total/(1<<20):,.0f} MB...")
                    sys.stdout.flush()
    print(f"[atlas] wrote {total/(1<<20):,.1f} MB -> {dst_zip}")
    return total


def extract_and_rename(zip_path: Path, out_dir: Path, pdb_chain: str,
                       replicas: list[int]):
    """Unpack the ATLAS zip and rename files into the bundle's layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    # PDB reference structure -> reference.pdb at top.
    pdb_files = list(raw_dir.glob(f"{pdb_chain}.pdb"))
    if pdb_files:
        ref = out_dir / "reference.pdb"
        pdb_files[0].rename(ref)
        print(f"[atlas]   reference -> {ref}")

    # README.txt at top.
    for nm in ("README.txt",):
        src = raw_dir / nm
        if src.exists():
            src.rename(out_dir / nm)

    # XTC trajectories -> replica_<i>/traj.xtc.
    for r in replicas:
        r_atlas = r + 1     # ATLAS uses R1/R2/R3, we use 0/1/2
        xtc_src = raw_dir / f"{pdb_chain}_prod_R{r_atlas}_fit.xtc"
        if not xtc_src.exists():
            print(f"[atlas]   [warn] no replica R{r_atlas} in archive")
            continue
        rep_dir = out_dir / f"replica_{r}"
        rep_dir.mkdir(exist_ok=True)
        dst = rep_dir / "traj.xtc"
        xtc_src.rename(dst)
        print(f"[atlas]   replica {r} -> {dst}")

    # Drop the raw subdir (now empty of items we wanted; remaining TPR/etc
    # stay there for reference).
    print(f"[atlas]   ancillary files kept in {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_chain", help="ATLAS entry id, e.g. 1k5n_A")
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--replicas", default="0,1,2",
                    help="comma-separated 0-based replica indices to keep")
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse <out_root>/<pdb>.zip if it already exists")
    args = ap.parse_args()

    replicas = [int(x) for x in args.replicas.split(",")]
    out_dir = args.out_root / f"atlas_{args.pdb_chain}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[atlas] fetching metadata for {args.pdb_chain}")
    meta = fetch_metadata(args.pdb_chain)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[atlas]   {meta.get('protein_name', '?')} ({meta.get('length', '?')} residues, "
          f"alpha {meta.get('alpha%', '?')}%, beta {meta.get('beta%', '?')}%)")

    zip_path = args.out_root / f"{args.pdb_chain}.zip"
    if args.skip_download and zip_path.exists():
        print(f"[atlas] reusing existing {zip_path}")
    else:
        fetch_protein_archive(args.pdb_chain, zip_path)

    extract_and_rename(zip_path, out_dir, args.pdb_chain, replicas)
    print(f"\n[atlas] DONE. {out_dir}")
    print(f"\nLoad into the bundle via:")
    print(f"  open {out_dir}/reference.pdb")
    print(f"  vampnet load_ensemble md {out_dir}/replica_0/traj.xtc")


if __name__ == "__main__":
    main()
