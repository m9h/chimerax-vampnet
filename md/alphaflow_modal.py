"""AlphaFlow ensemble generation on Modal.

Wraps Bowen Jing et al.'s AlphaFlow (github.com/bjing2016/alphaflow,
ICML 2024) to produce N conformational samples from a single protein
sequence. Output is saved as an AlphaFlow-shape .npz that the bundle's
`vampnet load_ensemble path source alphaflow` consumes natively.

Inference settings follow the "alphaflow-md-base" recipe -- the
checkpoint trained on MD-like ensemble distributions, which is the
right comparison for chignolin/notch1 MD analyses.

  modal run alphaflow_modal.py::sample --sequence "YYDPETGTWY" --n 200 --out chignolin_af.npz

Notes:
  - AlphaFlow MD checkpoint is ~3 GB; the Modal image caches it across
    runs in /root/.cache/huggingface so repeat invocations don't re-download.
  - 200 samples on an H100 takes ~10 minutes for a small protein and
    burns ~$1.5 of Modal credit.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-alphaflow"

image = (
    # NGC PyTorch base: tuned CUDA + cuDNN + torch + NCCL + apex +
    # flash_attn already installed (we'll patch openfold for any
    # flash_attn API name changes). Same base as the v0.4 marsfm
    # adapter — see md/marsfm_modal.py for the bisection that landed
    # on this combination.
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.04-py3",
                                add_python=None)
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        # OpenFold uses Bio.Data.SCOPData which was removed from
        # biopython after 1.79; biopython 1.79 doesn't build for
        # python 3.12 (no wheel). We patch openfold's import to use a
        # local fallback in the sed step below instead of pinning the
        # ancient biopython.
        "biopython",
        "huggingface_hub",
        "einops",
        "pyyaml",
        "ml-collections",
        "absl-py",
        "dm-tree",
        "pytorch-lightning>=2.4",
        "deeptime",  # alphaflow's analysis helpers reach for it
        "mdtraj",
        "modelcif",  # openfold writes mmcif via this
        "fair-esm",  # required for ESMFold mode
        # numpy/scipy/pandas/tqdm already in NGC pytorch image
    )
    .run_commands(
        "cd /opt && git clone https://github.com/bjing2016/alphaflow.git",
        "cd /opt/alphaflow && (pip install -e . --no-deps || true)",
        # AlphaFlow imports openfold modules but the OpenFold wheel
        # build requires a specific CUDA toolkit version that NGC PyTorch
        # 26.04 doesn't match. We need only the python-level modules
        # (e.g. openfold.data.mmcif_parsing), so clone the source and
        # add it to PYTHONPATH instead.
        "cd /opt && git clone --branch pl_upgrades "
        "https://github.com/aqlaboratory/openfold.git openfold-src "
        "|| git clone https://github.com/aqlaboratory/openfold.git openfold-src",
        "cd /opt/openfold-src && git checkout 103d037 2>/dev/null || true",
        # Patch flash_attn name change if needed.
        "find /opt/alphaflow /opt/openfold-src -name 'primitives.py' "
        "-o -name 'ipa.py' "
        "| xargs sed -i 's/flash_attn_unpadded_kvpacked_func/"
        "flash_attn_varlen_kvpacked_func/g' 2>/dev/null || true",
        # Replace the dead SCOPData import with a shim that pulls the
        # equivalent 3-to-1 mapping from Bio.SeqUtils.IUPACData.
        "sed -i 's|from Bio.Data import SCOPData|"
        "from Bio.SeqUtils import IUPACData as _Iupac\\n"
        "class SCOPData:\\n"
        "    protein_letters_3to1 = {k.upper(): v for k, v in "
        "_Iupac.protein_letters_3to1_extended.items()}|' "
        "/opt/openfold-src/openfold/data/mmcif_parsing.py 2>/dev/null || true",
        # numpy 2.0 removed np.object, np.int, np.float, np.bool, np.long.
        # OpenFold/AlphaFlow still uses the deprecated aliases in several
        # places. Replace with the builtin equivalents across the python tree.
        "find /opt/openfold-src /opt/alphaflow -name '*.py' "
        "-exec sed -i "
        "-e 's/np\\.object\\b/object/g' "
        "-e 's/np\\.int\\b/int/g' "
        "-e 's/np\\.float\\b/float/g' "
        "-e 's/np\\.bool\\b/bool/g' "
        "-e 's/np\\.long\\b/int/g' "
        "{} \\; 2>/dev/null || true",
    )
    .env({"PYTHONPATH": "/opt/openfold-src:/opt/alphaflow"})
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="H100", timeout=3600)
def sample_remote(sequence: str, name: str = "query", n_samples: int = 200,
                   checkpoint: str = "esmflow_md_base_202402") -> bytes:
    """Run AlphaFlow (in ESMFlow mode) inference.

    We use ESMFlow rather than the MSA-requiring AlphaFlow mode to
    avoid the MSA preprocessing overhead. ESMFlow-MD is the
    flow-matched ESMFold trained on the same MD trajectories as the
    AlphaFlow-MD model. Single-sequence input; no MSA required.

    Returns the packed ensemble as .npz bytes.
    """
    import subprocess
    import tempfile

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    print(f"[af] downloading checkpoint params/{checkpoint}.pt from bjing-mit/alphaflow")
    ckpt_path = hf_hub_download(repo_id="bjing-mit/alphaflow",
                                 filename=f"params/{checkpoint}.pt",
                                 cache_dir="/root/.cache/huggingface")
    print(f"[af] checkpoint at {ckpt_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # AlphaFlow's predict.py expects an --input_csv with name,seqres
        # columns. For ESMFold mode no MSA directory is needed.
        csv_path = tmp / "input.csv"
        csv_path.write_text(f"name,seqres\n{name},{sequence}\n")
        out_dir = tmp / "out"
        out_dir.mkdir()
        cmd = [
            "python", "/opt/alphaflow/predict.py",
            "--mode", "esmfold",
            "--input_csv", str(csv_path),
            "--samples", str(n_samples),
            "--weights", ckpt_path,
            "--outpdb", str(out_dir),
        ]
        print(f"[af] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd="/opt/alphaflow")

        # AlphaFlow writes a single multi-MODEL PDB per protein name.
        # Use mdtraj which loads multi-model PDB as a trajectory.
        import mdtraj as md
        sample_paths = sorted(out_dir.glob("*.pdb"))
        if not sample_paths:
            raise RuntimeError("AlphaFlow produced no PDB outputs")
        pdb_path = sample_paths[0]
        traj = md.load(str(pdb_path))
        coords_all = (traj.xyz * 10.0).astype(np.float32)  # nm -> A
        ca_indices = [a.index for a in traj.topology.atoms if a.name == "CA"]
        coords_ca = coords_all[:, ca_indices, :]
        print(f"[af] ensemble shape: all-atom {coords_all.shape}, "
              f"CA-only {coords_ca.shape}")

        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, coords=coords_all, coords_ca=coords_ca,
                            seqres=np.array(sequence))
        return buf.getvalue()


@app.local_entrypoint()
def sample(sequence: str, name: str = "query", n: int = 200,
           out: str = "alphaflow_ensemble.npz",
           checkpoint: str = "esmflow_md_base_202402"):
    """Generate an ESMFlow-MD ensemble and save to a local .npz."""
    print(f"[local] generating {n} samples for seq len {len(sequence)} "
          f"(checkpoint {checkpoint})")
    data = sample_remote.remote(sequence, name, n, checkpoint)
    Path(out).write_bytes(data)
    print(f"[local] wrote {out} ({len(data)/(1<<20):.1f} MB)")
