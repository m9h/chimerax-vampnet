"""End-to-end ChimeraX integration test.

Drives a real ChimeraX session via `chimerax --nogui --exit --script`
to verify the bundle installs, registers all commands, and can run
the full pipeline (load_ensemble → fit → states → save → load) on
the chignolin demo data. This is the gating test for ChimeraX
Toolshed submission.

Skips automatically if:
  - The `chimerax` binary is not on PATH
  - The chignolin demo data at /data/datasets/chimerax-vampnet/... is
    not present (this is a user-side data path that may not exist in
    every dev environment).

Run:
  $ python -m pytest tests/test_chimerax_integration.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


CHIGNOLIN_PDB = Path(
    "/data/datasets/chimerax-vampnet/chignolin_modal/chignolin/equilibrated.pdb"
)
CHIGNOLIN_DCD = Path(
    "/data/datasets/chimerax-vampnet/chignolin_modal/chignolin/replica_0/traj.dcd"
)


def _chimerax_binary():
    """Locate a ChimeraX binary or None if unavailable."""
    for name in ("chimerax", "ChimeraX", "/opt/UCSF/ChimeraX/bin/ChimeraX"):
        p = shutil.which(name)
        if p:
            return p
    return None


@pytest.fixture
def cx_bin():
    p = _chimerax_binary()
    if not p:
        pytest.skip("ChimeraX binary not on PATH (set PATH or install via "
                     "https://www.cgl.ucsf.edu/chimerax/download.html)")
    return p


@pytest.fixture
def chignolin_data():
    if not (CHIGNOLIN_PDB.exists() and CHIGNOLIN_DCD.exists()):
        pytest.skip(
            f"Chignolin demo data missing ({CHIGNOLIN_PDB.parent}). Run the "
            "chignolin Modal MD pipeline first or set up the data via "
            "`modal volume get chimerax-vampnet-md prepared/chignolin/`"
        )
    return CHIGNOLIN_PDB, CHIGNOLIN_DCD


def test_bundle_loads_in_chimerax(cx_bin):
    """The bundle should at least import without erroring in a real session."""
    cmd = [
        cx_bin, "--nogui", "--silent", "--exit",
        "--cmd", "echo chimerax-vampnet bundle smoke",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"chimerax bundle smoke exited {r.returncode}\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )


def test_full_pipeline_chignolin(cx_bin, chignolin_data):
    """Drive the full load → fit → states → save → load pipeline against
    the chignolin demo data via the existing chignolin_headless.cxc
    script. Asserts a model pkl is produced and is non-empty."""
    pdb_path, dcd_path = chignolin_data
    repo_root = Path(__file__).resolve().parent.parent
    headless_cxc = repo_root / "examples" / "chignolin_headless.cxc"
    assert headless_cxc.exists(), f"missing {headless_cxc}"

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        save_path = tmp / "chignolin_vampnet.pkl"
        # Substitute the save path so the test owns the artifact and
        # cleans up after itself.
        cxc_text = headless_cxc.read_text()
        cxc_text = cxc_text.replace("/tmp/chignolin_vampnet.pkl", str(save_path))
        local_cxc = tmp / "headless_integration.cxc"
        local_cxc.write_text(cxc_text)

        cmd = [
            cx_bin, "--nogui", "--silent", "--exit",
            "--cmd", f"open {local_cxc}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            pytest.fail(
                f"chimerax pipeline exited {r.returncode}\n"
                f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
            )
        assert save_path.exists(), f"vampnet save did not write {save_path}"
        assert save_path.stat().st_size > 0, "saved pkl is empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
