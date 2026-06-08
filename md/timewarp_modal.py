"""Timewarp (time-coarsened normalising-flow MCMC proposal) on Modal.

Paper: Klein, Foong, Fjelde, Mlodozeniec, Brockschmidt, Nowozin, Noé, Tomioka
2023, "Timewarp: Transferable Acceleration of Molecular Dynamics by Learning
Time-Coarsened Dynamics" (NeurIPS 2023; arXiv:2302.01170). Code at
https://github.com/microsoft/timewarp (MIT). Presented at Starkly Speaking
2023-10-25.

Why it earns a slot in the v0.9+ multisource pipeline:

  Timewarp learns a normalising flow that proposes moves of 10⁵–10⁶ fs (1-10
  ns of MD wall time) per step, used as a proposal in an MCMC chain that
  targets the exact Boltzmann distribution. Unlike BioEmu/Prose/AlphaFlow,
  Timewarp is *trajectory-aware* — it learns time-dependence, not just an
  equilibrium emulator. This makes it the only one of our generative
  candidates that can in principle estimate dynamical quantities (transition
  rates, residence times) and not just equilibrium populations.

  Applicability: the released Timewarp checkpoint was trained on 2-4-residue
  peptides (per the paper) and demonstrates transferability *within* that
  range. The v0.9 use case is therefore the same as Prose's — dipeptide /
  tetrapeptide benchmarking — not direct application to Notch1/Hsp90/β2AR.

  Newer alternative: the same group released "UniSim: A Unified Simulator for
  Time-Coarsened Dynamics of Biomolecules" (github.com/transferable-samplers/
  UniSim) as a Timewarp successor with broader coverage. If the v0.9 dipeptide
  benchmark passes, the next step is to swap to UniSim for larger systems —
  trivial in this adapter (change REPO_URL + import path).

  Base image:   modal.Image.debian_slim(python_version=3.11)
  Pip extras:   torch (cu124), microsoft/timewarp from git, biopython,
                mdtraj, numpy<2.
  Checkpoint:   Pretrained 2-4-residue model — released alongside the paper.
                The Modal volume `chimerax-vampnet-md` will cache it on
                first use to avoid repeated HF/Azure pulls.
  GPU pin:      A100-40GB (sub-100M params).
  Tested:       UNTESTED as of 2026-06-07 — scaffold only. First-run unknowns:
                  1. Repo's pip-installable name (`timewarp`? `microsoft-timewarp`?).
                  2. The exact MCMC entry point — the repo exposes
                     `timewarp.sample(system, n_steps)` or analogous.
                  3. Whether the released checkpoint expects a topology file
                     or just a sequence.

  modal run md/timewarp_modal.py::sample \\
      --pdb md/ala_dipeptide.pdb --name ala_dipeptide \\
      --n-mcmc-steps 5000 --burnin 500 \\
      --out ala_dipeptide_timewarp.npz
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-timewarp"
REPO_URL = "git+https://github.com/microsoft/timewarp.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"timewarp @ {REPO_URL}",  # TODO: verify package name
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "numpy<2",
        "einops",
        "gemmi",
    )
)

VOLUME_NAME = "chimerax-vampnet-md"  # reuse the main MD volume for weight cache
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_MOUNT = "/vol"


@app.function(gpu="A100-40GB", timeout=4 * 3600, volumes={VOL_MOUNT: vol})
def sample_remote(pdb_bytes: bytes, name: str,
                  n_mcmc_steps: int = 5000, burnin: int = 500,
                  save_stride: int = 10, temperature_K: float = 300.0) -> bytes:
    """Run Timewarp MCMC: each step proposes a large-time-step move via the
    learned flow, accepts/rejects against the energy model. Returns the
    post-burnin chain at the requested save stride.

    Output schema matches the other generative adapters (coords / coords_ca /
    seqres / chain_id / plddt-placeholder), with extra fields:
      accept_rate     : float
      autocorr_time   : float   (estimated from the saved chain)
    """
    import io

    import numpy as np
    import torch

    # First-invocation TODO: verify package name + entry-point names.
    from timewarp import MCMCSampler, load_pretrained  # noqa: F401 — verify

    pdb_path = Path("/tmp/system.pdb")
    pdb_path.write_bytes(pdb_bytes)

    print(f"[timewarp] {name}: {n_mcmc_steps} MCMC steps, "
          f"burnin={burnin}, save_stride={save_stride}, T={temperature_K} K")

    cache_dir = Path(VOL_MOUNT) / "timewarp_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = load_pretrained(cache_dir=str(cache_dir))  # T2-4 residue release
    model = model.cuda().eval()

    sampler = MCMCSampler(model, temperature_K=temperature_K)
    chain = sampler.run(pdb_path=str(pdb_path),
                         n_steps=n_mcmc_steps,
                         burnin=burnin, save_stride=save_stride)
    # Expected chain fields:
    #   chain.coords      : (n_kept, n_atoms, 3)
    #   chain.ca_mask     : (n_atoms,) bool
    #   chain.accept_rate : float
    coords_all = np.asarray(chain.coords, dtype=np.float32)
    ca_mask = np.asarray(chain.ca_mask, dtype=bool)
    coords_ca = coords_all[:, ca_mask, :]
    n_kept = coords_all.shape[0]
    accept_rate = float(chain.accept_rate)
    print(f"[timewarp] kept {n_kept} samples, accept rate {accept_rate:.3f}, "
          f"coords {coords_all.shape}, coords_ca {coords_ca.shape}")

    # Crude autocorr estimate on CA-RMSD-to-frame-0 for diagnostics.
    diff = coords_ca - coords_ca[:1]
    rmsd = np.sqrt(((diff) ** 2).sum(-1).mean(-1))
    autocorr = float(_integrated_autocorr(rmsd))

    chain_id_per_res = np.zeros(int(ca_mask.sum()), dtype=np.int64)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=coords_all,
        coords_ca=coords_ca,
        seqres=np.array(getattr(chain, "seqres", "")),
        chain_id=chain_id_per_res,
        plddt=np.full(n_kept, np.nan, dtype=np.float32),
        iptm=np.full(n_kept, np.nan, dtype=np.float32),
        accept_rate=np.array(accept_rate, dtype=np.float32),
        autocorr_time=np.array(autocorr, dtype=np.float32),
        temperature_K=np.array(temperature_K, dtype=np.float32),
    )
    vol.commit()
    return buf.getvalue()


def _integrated_autocorr(x):
    """Simple integrated autocorrelation time estimate for a 1D series."""
    import numpy as np
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    if n < 4:
        return 1.0
    # FFT-based autocorr
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0] + 1e-12
    # Truncate at first negative sample
    neg = np.where(acf < 0)[0]
    cutoff = int(neg[0]) if len(neg) else min(n, 100)
    return 1.0 + 2.0 * float(acf[1:cutoff].sum())


@app.local_entrypoint()
def sample(pdb: str, name: str = "system", n_mcmc_steps: int = 5000,
           burnin: int = 500, save_stride: int = 10,
           temperature: float = 300.0, out: str = ""):
    """Run Timewarp MCMC on a small peptide system (released checkpoint
    trained for 2-4-residue peptides — extrapolation is exploratory)."""
    pdb_bytes = Path(pdb).read_bytes()
    print(f"[local] Timewarp on {name}, {len(pdb_bytes)} bytes PDB, "
          f"{n_mcmc_steps} steps")
    data = sample_remote.remote(pdb_bytes, name,
                                 n_mcmc_steps=n_mcmc_steps,
                                 burnin=burnin, save_stride=save_stride,
                                 temperature_K=temperature)
    out_path = Path(out) if out else Path(f"{name}_timewarp.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
