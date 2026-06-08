"""Tests for the newer ensemble loader format branches.

test_multi_source covers alphaflow + bioemu. This covers the boltz,
marsfm, and esmfold2 (v0.7.5, just added) branches + the esmfold2
extension to _detect_format — the paths a typo would otherwise break
silently until a live run.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class _Sess:
    pass


def _write_npz(path: Path, key: str, n=7, a=10, seed=0):
    coords = np.random.RandomState(seed).randn(n, a, 3).astype("float32")
    np.savez(path, **{key: coords})
    return coords


def test_detect_format_recognizes_new_sources():
    import featurize
    assert featurize._detect_format("run_esmfold2.npz") == "esmfold2"
    assert featurize._detect_format("notch1_NEC_boltz200.npz") == "boltz"
    assert featurize._detect_format("x_marsfm.npz") == "marsfm"
    # esmfold2 spelling variants
    assert featurize._detect_format("x_esmfold-2.npz") == "esmfold2"


def test_esmfold2_branch_loads_coords():
    import featurize
    sess = _Sess()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "hsp90_esmfold2.npz"
        coords = _write_npz(p, "coords")
        n_frames, fmt = featurize.load_ensemble(sess, "esmfold2", str(p),
                                                format="esmfold2")
        assert fmt == "esmfold2"
        assert n_frames == coords.shape[0]
        reg = featurize._registry(sess)
        assert reg[-1]["source"] == "esmfold2"
        assert reg[-1]["coords"].shape == coords.shape


def test_esmfold2_branch_falls_back_to_coords_ca():
    import featurize
    sess = _Sess()
    with tempfile.TemporaryDirectory() as tmp:
        # CA-only deposit: only coords_ca present, no all-atom 'coords'.
        p = Path(tmp) / "ca_only_esmfold2.npz"
        ca = _write_npz(p, "coords_ca")
        n_frames, fmt = featurize.load_ensemble(sess, "esmfold2", str(p),
                                                format="esmfold2")
        assert n_frames == ca.shape[0]
        assert featurize._registry(sess)[-1]["coords"].shape == ca.shape


def test_boltz_and_marsfm_branches_load():
    import featurize
    for fmt, key in (("boltz", "samples"), ("marsfm", "coords")):
        sess = _Sess()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"x_{fmt}.npz"
            coords = _write_npz(p, key)
            n_frames, detected = featurize.load_ensemble(sess, fmt, str(p),
                                                         format=fmt)
            assert detected == fmt
            assert n_frames == coords.shape[0]


def test_auto_detect_then_load_esmfold2():
    import featurize
    sess = _Sess()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "auto_esmfold2.npz"
        coords = _write_npz(p, "coords")
        n_frames, fmt = featurize.load_ensemble(sess, "gen", str(p),
                                                format="auto")
        assert fmt == "esmfold2"
        assert n_frames == coords.shape[0]
