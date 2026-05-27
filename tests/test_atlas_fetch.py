"""Tests for the ATLAS fetcher.

The live HTTP path isn't tested here (single ATLAS download is ~600 MB
and slow); we exercise extract_and_rename() on a synthetic ZIP that
mimics the ATLAS archive shape: <pdb>.pdb, <pdb>_prod_R{1,2,3}_fit.xtc,
<pdb>_prod_R{1,2,3}.tpr, README.txt.
"""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "md"))


def _make_synthetic_atlas_zip(zip_path: Path, pdb_chain: str):
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{pdb_chain}.pdb",
                    "HEADER fake\nATOM      1  N   ALA A   1       0.0   0.0   0.0\nEND\n")
        zf.writestr("README.txt", "This is the ATLAS protein-only archive readme.")
        for r in (1, 2, 3):
            zf.writestr(f"{pdb_chain}_prod_R{r}_fit.xtc", b"\x01\x02\x03 fake xtc bytes")
            zf.writestr(f"{pdb_chain}_prod_R{r}.tpr", b"fake tpr bytes")


def test_extract_and_rename_default_replicas():
    from atlas_fetch import extract_and_rename

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "1abc_A.zip"
        _make_synthetic_atlas_zip(zip_path, "1abc_A")

        out_dir = tmp / "out"
        out_dir.mkdir()
        extract_and_rename(zip_path, out_dir, "1abc_A", [0, 1, 2])

        assert (out_dir / "reference.pdb").exists()
        assert (out_dir / "README.txt").exists()
        for r in (0, 1, 2):
            traj = out_dir / f"replica_{r}" / "traj.xtc"
            assert traj.exists(), f"missing {traj}"
        assert (out_dir / "_raw").exists()


def test_extract_subset_of_replicas():
    """Asking for only replicas 0 and 2 should produce just those dirs."""
    from atlas_fetch import extract_and_rename

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "1abc_A.zip"
        _make_synthetic_atlas_zip(zip_path, "1abc_A")

        out_dir = tmp / "out"
        extract_and_rename(zip_path, out_dir, "1abc_A", [0, 2])

        assert (out_dir / "replica_0" / "traj.xtc").exists()
        assert (out_dir / "replica_2" / "traj.xtc").exists()
        assert not (out_dir / "replica_1" / "traj.xtc").exists()


def test_metadata_fetch_returns_a_dict():
    """Hit the live metadata endpoint; very small (~3 KB) so OK in CI.

    Marked skippable so an offline test environment doesn't fail.
    """
    import urllib.error
    from atlas_fetch import fetch_metadata
    try:
        meta = fetch_metadata("1k5n_A")
    except (urllib.error.URLError, urllib.error.HTTPError):
        import pytest
        pytest.skip("ATLAS metadata endpoint unreachable")
    assert isinstance(meta, dict)
    assert "protein_name" in meta
    assert meta["length"] > 0


if __name__ == "__main__":
    test_extract_and_rename_default_replicas()
    test_extract_subset_of_replicas()
    test_metadata_fetch_returns_a_dict()
    print("OK")
