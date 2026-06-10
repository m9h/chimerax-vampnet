"""Smoke + pure-helper tests for the md/*_modal.py Modal adapters.

The adapters do their real work inside GPU `@app.function` bodies that
can't run in CI. But the whole module — constants, `modal.App`
construction, decorators, and any module-level pure helpers — executes
at *import* time, so an import-smoke test catches the most common
scaffold rot (syntax errors, NameErrors, bad signatures) across all
~1,400 lines of adapter code that the coverage gate otherwise can't see
(they live in md/, outside --cov=src --cov=md/rsi).

Importing an adapter only needs `modal` installed (top-level imports are
modal + stdlib; torch/biopython/etc. are imported lazily inside the
remote functions). `importorskip` keeps this green where modal is
absent; CI installs modal so it actually runs there.
"""

import glob
import importlib
import sys
from pathlib import Path

import pytest

MD = Path(__file__).resolve().parent.parent / "md"
sys.path.insert(0, str(MD))

pytest.importorskip("modal", reason="modal not installed; adapter smoke skipped")

ADAPTERS = sorted(Path(p).stem for p in glob.glob(str(MD / "*_modal.py")))


def test_there_are_adapters_to_smoke():
    # Guard against the glob silently finding nothing (which would make
    # the parametrized test vacuously pass).
    assert len(ADAPTERS) >= 13


@pytest.mark.parametrize("module_name", ADAPTERS)
def test_adapter_imports(module_name):
    """Each adapter module imports without error and exposes a modal App."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "app"), f"{module_name} has no modal App named 'app'"
    assert mod.APP_NAME.startswith("chimerax-vampnet")


# ----------------------------------------------------- uma_modal pure helpers

def _atom_line(serial, atomname, resname, chain="A", resseq=1):
    """Build a column-correct PDB ATOM record. `^4` centres a 2-char
    atom name as ' CA ', matching real PDB columns 12-16."""
    return (
        f"ATOM  {serial:>5} {atomname:^4} {resname:>3} {chain}{resseq:>4}"
        f"      0.000   0.000   0.000  1.00  0.00           C"
    )


def test_uma_seqres_from_pdb():
    import uma_modal
    pdb = "\n".join([
        _atom_line(1, "N", "ALA", resseq=1),
        _atom_line(2, "CA", "ALA", resseq=1),
        _atom_line(3, "C", "ALA", resseq=1),
        _atom_line(4, "CA", "GLY", resseq=2),
        _atom_line(5, "CA", "TRP", resseq=3),
    ])
    # One letter per CA record, in order.
    assert uma_modal._seqres_from_pdb(pdb) == "AGW"


def test_uma_seqres_unknown_residue_is_X():
    import uma_modal
    pdb = _atom_line(1, "CA", "XYZ", resseq=1)
    assert uma_modal._seqres_from_pdb(pdb) == "X"


def test_uma_ca_indices_are_atom_order_positions():
    import uma_modal
    pdb = "\n".join([
        _atom_line(1, "N", "ALA"),   # atom 0
        _atom_line(2, "CA", "ALA"),  # atom 1  <- CA
        _atom_line(3, "C", "ALA"),   # atom 2
        _atom_line(4, "O", "ALA"),   # atom 3
        _atom_line(5, "N", "GLY", resseq=2),   # atom 4
        _atom_line(6, "CA", "GLY", resseq=2),  # atom 5  <- CA
    ])
    assert uma_modal._ca_indices_from_pdb(pdb) == [1, 5]


def test_uma_ca_indices_ignore_hetatm():
    import uma_modal
    pdb = "\n".join([
        _atom_line(1, "CA", "ALA"),
        "HETATM    2  CA  CA  A 101       0.000   0.000   0.000  1.00  0.00",
    ])
    # The HETATM calcium ion must not be counted as a CA backbone atom.
    assert uma_modal._ca_indices_from_pdb(pdb) == [0]


# ----------------------------------------- mcmc_diagnostics autocorr helper
# (this helper was centralized into md/mcmc_diagnostics.py in v0.10; it used
#  to live in md/timewarp_modal.py — test it where it now lives.)

def test_mcmc_autocorr_short_series():
    import mcmc_diagnostics
    assert mcmc_diagnostics._integrated_autocorr([1.0, 2.0]) == 1.0


def test_mcmc_autocorr_constant_series_is_one():
    import mcmc_diagnostics
    assert mcmc_diagnostics._integrated_autocorr([5.0] * 50) == pytest.approx(1.0)


def test_mcmc_autocorr_lower_bound_invariant():
    import numpy as np
    import mcmc_diagnostics
    # acf is truncated at the first negative lag, so the summed terms are
    # non-negative -> tau >= 1.0 for any series.
    rng = np.random.RandomState(0)
    tau = mcmc_diagnostics._integrated_autocorr(rng.randn(500))
    assert tau >= 1.0
    assert tau < 5.0  # white noise -> small integrated autocorr time


def test_mcmc_autocorr_correlated_exceeds_white_noise():
    import numpy as np
    import mcmc_diagnostics
    rng = np.random.RandomState(1)
    white = rng.randn(2000)
    # AR(1) with phi=0.9 -> strongly autocorrelated -> larger tau.
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.9 * ar[i - 1] + white[i]
    tau_white = mcmc_diagnostics._integrated_autocorr(white)
    tau_ar = mcmc_diagnostics._integrated_autocorr(ar)
    assert tau_ar > tau_white
    assert tau_ar > 3.0
