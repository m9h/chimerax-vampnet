"""Tests for featurize.featurize_torsions — previously untested feature mode.

Only the ca_distances path had coverage; the backbone-torsion path (the
other half of the bundle's featurization) had none.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_torsions_output_shape():
    import featurize
    # 5 frames, 6 atoms; two dihedral quads -> P=2 -> output 2*P = 4 cols.
    coords = np.random.RandomState(0).randn(5, 6, 3).astype("float32")
    quads = [(0, 1, 2, 3), (2, 3, 4, 5)]
    feats = featurize.featurize_torsions(coords, quads)
    assert feats.shape == (5, 4)  # [sin(t0), sin(t1), cos(t0), cos(t1)]


def test_torsions_sin_cos_unit_circle():
    import featurize
    coords = np.random.RandomState(1).randn(8, 5, 3).astype("float32")
    quads = [(0, 1, 2, 3), (1, 2, 3, 4)]
    feats = featurize.featurize_torsions(coords, quads)
    P = len(quads)
    sin_t, cos_t = feats[:, :P], feats[:, P:]
    # sin^2 + cos^2 == 1 for every torsion of every frame.
    assert np.allclose(sin_t**2 + cos_t**2, 1.0, atol=1e-5)


def test_planar_dihedral_is_zero():
    import featurize
    # A planar (cis, 0-degree) dihedral: all four atoms in the z=0 plane,
    # arranged so the torsion angle is exactly 0 -> sin=0, cos=1.
    pts = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype="float32")
    coords = pts[None]  # (1 frame, 4 atoms, 3)
    feats = featurize.featurize_torsions(coords, [(0, 1, 2, 3)])
    sin_t, cos_t = feats[0, 0], feats[0, 1]
    assert abs(sin_t) < 1e-5
    assert abs(cos_t - 1.0) < 1e-5
