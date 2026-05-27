"""Tests for the multi-source ensemble loader.

We synthesize AlphaFlow- and BioEmu-shaped .npz files and verify that
`featurize.load_ensemble` accepts them, populates the session
registry, and that `stacked_coords` produces a single concatenated
array tagged by source. This is the path the bundle takes when an MD
trajectory and an AF-class prediction are fused into one VAMPNet
embedding.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class _Sess:
    pass


def _write_npz(path: Path, key: str, n_frames: int, n_atoms: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    coords = rng.randn(n_frames, n_atoms, 3).astype("float32")
    np.savez(path, **{key: coords})
    return coords


def test_alphaflow_npz_load():
    import featurize

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "alphaflow_clntest.npz"
        coords = _write_npz(npz_path, key="coords", n_frames=200, n_atoms=20)

        sess = _Sess()
        n_frames, fmt = featurize.load_ensemble(sess, source="alphaflow", path=str(npz_path), format="alphaflow")

        assert n_frames == 200
        assert fmt == "alphaflow"
        registry = featurize._registry(sess)
        assert len(registry) == 1
        assert registry[0]["source"] == "alphaflow"
        assert registry[0]["coords"].shape == (200, 20, 3)


def test_bioemu_npz_load():
    import featurize

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "bioemu_clntest.npz"
        _write_npz(npz_path, key="samples", n_frames=150, n_atoms=20)

        sess = _Sess()
        n_frames, fmt = featurize.load_ensemble(sess, source="bioemu", path=str(npz_path), format="bioemu")

        assert n_frames == 150
        assert fmt == "bioemu"


def test_stacked_coords_multi_source():
    """Loading AlphaFlow + BioEmu into the same session should produce a
    single concatenated (N_total, A, 3) array plus a per-frame source tag."""
    import featurize

    with tempfile.TemporaryDirectory() as tmp:
        af_path = Path(tmp) / "alphaflow_test.npz"
        be_path = Path(tmp) / "bioemu_test.npz"
        _write_npz(af_path, key="coords", n_frames=100, n_atoms=20)
        _write_npz(be_path, key="samples", n_frames=50, n_atoms=20, seed=1)

        sess = _Sess()
        featurize.load_ensemble(sess, "alphaflow", str(af_path), format="alphaflow")
        featurize.load_ensemble(sess, "bioemu", str(be_path), format="bioemu")

        coords, sources = featurize.stacked_coords(sess)
        assert coords.shape == (150, 20, 3)
        assert len(sources) == 150
        assert (sources[:100] == "alphaflow").all()
        assert (sources[100:] == "bioemu").all()


def test_featurize_ca_distances_on_joint_ensemble():
    """The output of featurize_ca_distances on a fused AlphaFlow+BioEmu
    ensemble should be a (N, A*(A-1)/2) standardized feature matrix."""
    import featurize

    with tempfile.TemporaryDirectory() as tmp:
        af = Path(tmp) / "af.npz"
        be = Path(tmp) / "be.npz"
        _write_npz(af, "coords", 80, 15)
        _write_npz(be, "samples", 40, 15, seed=2)
        sess = _Sess()
        featurize.load_ensemble(sess, "alphaflow", str(af), "alphaflow")
        featurize.load_ensemble(sess, "bioemu", str(be), "bioemu")
        coords, _ = featurize.stacked_coords(sess)
        X = featurize.featurize_ca_distances(coords)
        # 15 CAs -> 15*14/2 = 105 pair distances
        assert X.shape == (120, 105)
        # Standardized — per-pair mean ~0, std ~1.
        assert abs(X.mean()) < 0.05
        assert abs(X.std() - 1.0) < 0.05


if __name__ == "__main__":
    test_alphaflow_npz_load()
    test_bioemu_npz_load()
    test_stacked_coords_multi_source()
    test_featurize_ca_distances_on_joint_ensemble()
    print("OK")
