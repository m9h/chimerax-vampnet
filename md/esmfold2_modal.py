"""ESMFold2 ensemble generation on Modal.

Self-contained environment recipe — each Modal adapter in md/ builds
its own image rather than sharing a base, so dep collisions stay
isolated to the tool that needs them.

  Base image:   modal.Image.debian_slim(python_version=3.12)
                (biohub/esm pyproject pins requires-python >=3.12,<3.13)
  Pip extras:   torch 2.4.1 (cu124); `esm` and `transformers` BOTH from
                the Biohub git forks (PyPI `esm` 3.2.3 is the unrelated
                facebookresearch/esm package and lacks models.esmfold2);
                rdkit/biotite/msgpack-numpy/einops/ipython per the
                upstream pyproject's required deps
  Weights:      biohub/ESMFold2 on HuggingFace (MIT) — auto-pulled by
                ESMFold2Model.from_pretrained on first call
  GPU pin:      A100-80GB
  Tested:       v0.8 Phase 1 first attempt failed on 2026-06-07 with
                ModuleNotFoundError (PyPI `esm` was facebookresearch/esm,
                Python 3.11 violated upstream pyproject); image recipe
                now matches https://github.com/Biohub/esm pyproject.toml
                main, retest pending.

ESMFold2 (Rives et al. 2026, "A World Model of Protein Biology",
biohub.org) is an AF3-class diffusion structure/complex predictor on
top of the ESMC protein language model. It is explicitly a SINGLE-
STRUCTURE predictor — its authors state it is *not* designed for
conformational dynamics or flexibility. We use it here exactly as we
use the other AF3-class samplers (Boltz-2, AlphaFlow): draw N samples
of one input and feed them into the joint VAMPnet as one more source.

Two reasons it earns a slot in the v0.7 6-source pipeline:

  1. H3 stress test. It is the newest, strongest AF3-class predictor.
     Our H3 finding is that AF3-class diffusion samplers concentrate
     near the training manifold (Boltz-2 over-compact, AlphaFlow
     low-variance) while MD / MarS-FM explore. The prediction is that
     even SOTA ESMFold2 *collapses* to the modal/crystal state on
     Hsp90 and *fails to expand* on Notch1 — confirming that single-
     structure prediction, however good, is not a dynamics substitute.

  2. Native multi-chain. ESMFold2 is SOTA on antibody-antigen and PPI
     and takes real multi-chain input (one ProteinInput per chain with
     its own `id`). This is exactly the Notch1 NRR case (NEC + NTM,
     and the holo anti-NRR Fab) where MarS-FM's multi-chain path is
     blocked (the released MD-CATH 450 checkpoint has abs_pos_emb=False
     — see md/marsfm_multichain_phase1_results.md). ESMFold2 gives a
     working multi-chain generative source for the H2/H3 complex
     analysis *today*, MIT-licensed, no retrain.

It does NOT address the H2 magnitude / sampling-horizon question — it
is a single-structure-manifold sampler, not dynamics. Treat its
ensemble like Boltz-2's: a manifold probe, not an MD replacement.

Diversity strategy: a diffusion model's per-sample diversity comes
from the noise seed. Rather than trusting how `result.complex` packs
`num_diffusion_samples>1` (which the release README only exercises at
=1), we draw an ensemble the robust way — one fold() call per seed,
num_diffusion_samples=1, seed=i for i in range(n_samples) — mirroring
how boltz_modal.py collects N independent model PDBs. Same all-atom
ordering across seeds (same complex), so the stack is well-defined.

  # single chain (give it a sequence directly)
  modal run md/esmfold2_modal.py::sample \\
      --sequence "ACELPECQ..." --name notch1_NEC \\
      --n-samples 200 --out notch1_NEC_esmfold2.npz

  # multi-chain from a prepped PDB (the ESMFold2 edge — e.g. Notch1
  # NRR apo NEC+NTM, or holo NRR+Fab). Parses one ProteinInput per
  # chain present (or per the --chain-id subset, comma-separated).
  modal run md/esmfold2_modal.py::sample \\
      --pdb md/3i08_apo.pdb --chain-id "X,K" --name notch1_apo \\
      --n-samples 200 --out notch1_apo_esmfold2.npz

Per-sample cost on A100-80GB: ESMFold2 is fast (the release headline
is days-not-months design throughput); ~200 samples of a ~234-residue
complex expected in the low single-dollar range. Update this line with
the measured wall-clock after the first run.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chimerax-vampnet-esmfold2"
HF_REPO = "biohub/ESMFold2"

# Centralised so a single edit fixes the whole adapter if the released
# import path differs from the pre-release README.
_ESM_IMPORT = (
    "from esm.models.esmfold2 import "
    "ProteinInput, StructurePredictionInput, ESMFold2InputBuilder; "
    "from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model"
)

image = (
    # biohub/esm pyproject pins requires-python = ">=3.12,<3.13" and a custom
    # transformers fork — PyPI `esm` (3.2.3) is facebookresearch/esm and has
    # no models.esmfold2 module, so we MUST install both from git.
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "esm @ git+https://github.com/Biohub/esm.git@main",
        "transformers @ git+https://github.com/Biohub/transformers.git@main",
        "accelerate",
        "biopython",
        "mdtraj",
        "huggingface_hub",
        "numpy<2",
        "rdkit",
        "biotite>=1.0.0",
        "msgpack-numpy",
        "einops",
        "ipython",
        # gemmi: canonical mmCIF parser. ESMFold2's complex.to_mmcif()
        # emits a minimal mmCIF without _atom_site.occupancy, which
        # Biopython's MMCIFParser rejects; gemmi is tolerant.
        "gemmi",
    )
)

app = modal.App(APP_NAME, image=image)


THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _pdb_to_chain_seqs(pdb_bytes: bytes, chain_id: str | None = None):
    """Parse a PDB into [(chain_label, seqres), ...].

    chain_id options (mirrors marsfm_modal._pdb_to_atom14_and_seqres):
        None        -> every protein chain, in PDB iteration order
        "X,K"       -> exactly those chains, in the order listed
        "<letter>"  -> that single chain
    Chain labels handed to ESMFold2's ProteinInput.id are re-issued as
    A, B, C, ... in selection order (decoupled from the PDB letters) so
    the input is always a clean contiguous set.
    """
    import io
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", io.StringIO(pdb_bytes.decode("utf-8")))
    model = next(structure.get_models())

    def chain_seq(chain):
        return "".join(
            THREE_TO_ONE[r.get_resname()]
            for r in chain
            if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE
        )

    out = []
    if chain_id is None:
        for chain in model:
            s = chain_seq(chain)
            if s:
                out.append((chain.id, s))
    elif "," in chain_id:
        for wid in (c.strip() for c in chain_id.split(",")):
            s = chain_seq(model[wid])
            if s:
                out.append((wid, s))
    else:
        s = chain_seq(model[chain_id])
        if s:
            out.append((chain_id, s))
    if not out:
        raise ValueError("no protein chain found for the requested selection")
    # Re-issue clean ids A, B, C, ... in selection order.
    return [(chr(ord("A") + i), s) for i, (_, s) in enumerate(out)]


def _parse_chains_arg(chains: str):
    """Parse a '--chains' spec into [(id, seq), ...].

    Accepts either bare comma-separated sequences ("SEQ1,SEQ2" -> A, B)
    or explicit "id:seq" pairs ("A:SEQ1,B:SEQ2").
    """
    items = [c.strip() for c in chains.split(",") if c.strip()]
    out = []
    for i, item in enumerate(items):
        if ":" in item:
            cid, seq = item.split(":", 1)
            out.append((cid.strip(), seq.strip()))
        else:
            out.append((chr(ord("A") + i), item))
    return out


@app.function(gpu="A100-80GB", timeout=4 * 3600)
def sample_remote(chain_seqs: list, name: str, n_samples: int = 200,
                  num_loops: int = 3, num_sampling_steps: int = 50) -> bytes:
    """Run ESMFold2 inference on one complex (list of (id, seq) chains).

    Draws `n_samples` structures by varying the diffusion seed, parses
    each to coords, and returns a packed (all-atom + CA-only) .npz as
    bytes — shape-compatible with the boltz/marsfm loaders."""
    import io

    import gemmi
    import numpy as np
    import torch

    exec(_ESM_IMPORT, globals())

    chain_lens = [len(s) for _, s in chain_seqs]
    print(f"[esmfold2] name={name}, {len(chain_seqs)} chain(s) "
          f"ids={[c for c, _ in chain_seqs]} lengths={chain_lens}, "
          f"n_samples={n_samples}, num_loops={num_loops}, "
          f"num_sampling_steps={num_sampling_steps}")

    print(f"[esmfold2] loading weights {HF_REPO}")
    model = ESMFold2Model.from_pretrained(HF_REPO).cuda().eval()  # noqa: F821
    builder = ESMFold2InputBuilder()  # noqa: F821

    spi = StructurePredictionInput(  # noqa: F821
        sequences=[ProteinInput(id=cid, sequence=seq)  # noqa: F821
                   for cid, seq in chain_seqs]
    )

    coords_all = []
    atom_names_ref = None
    plddts = []
    iptms = []
    for s in range(n_samples):
        with torch.inference_mode():
            result = builder.fold(
                model, spi,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=1,
                seed=s,
            )
        # result.complex.to_mmcif() -> mmCIF text for the predicted complex.
        # Parse with gemmi (tolerant of missing _atom_site.occupancy that
        # ESMFold2's minimal writer omits, which Biopython rejects).
        cif_doc = gemmi.cif.read_string(result.complex.to_mmcif())
        structure = gemmi.make_structure_from_block(cif_doc.sole_block())
        xyz = []
        names = []
        for sm_model in structure:
            for chain in sm_model:
                for res in chain:
                    for atom in res:
                        xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
                        names.append(atom.name)
            break  # only the first NMR-style model (ESMFold2 emits one)
        xyz = np.asarray(xyz, dtype=np.float32)
        coords_all.append(xyz)
        # Confidence metadata for downstream filtering / QC.
        try:
            plddts.append(float(np.asarray(result.plddt).mean()))
            iptms.append(float(result.iptm) if result.iptm is not None else float("nan"))
        except Exception:
            pass
        if s == 0:
            atom_names_ref = names
            # Atom ordering is identical across seeds (same complex spec),
            # so the CA mask from sample 0 applies to all samples.
            ca_mask = np.array([n == "CA" for n in names], dtype=bool)
            n_ca = int(ca_mask.sum())
            print(f"[esmfold2] sample 0: {xyz.shape[0]} atoms, {n_ca} CAs")
        if (s + 1) % 25 == 0:
            print(f"[esmfold2] {s + 1}/{n_samples} samples done")

    coords_all = np.stack(coords_all, axis=0)  # (N, A, 3) Angstroms
    if not all(c.shape[0] == coords_all.shape[1] for c in coords_all):
        raise RuntimeError("ESMFold2 samples have inconsistent atom counts")
    coords_ca = coords_all[:, ca_mask, :]
    print(f"[esmfold2] all-atom {coords_all.shape}, CA-only {coords_ca.shape}")

    seqres = "".join(seq for _, seq in chain_seqs)
    chain_id_per_res = np.concatenate(
        [np.full(len(seq), i, dtype=np.int64) for i, (_, seq) in enumerate(chain_seqs)]
    )

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        coords=coords_all,
        coords_ca=coords_ca,
        seqres=np.array(seqres),
        chain_id=chain_id_per_res,
        plddt=np.array(plddts, dtype=np.float32),
        iptm=np.array(iptms, dtype=np.float32),
    )
    return buf.getvalue()


@app.local_entrypoint()
def sample(sequence: str = "", chains: str = "", pdb: str = "",
           chain_id: str = "", name: str = "protein", n_samples: int = 200,
           num_loops: int = 3, num_sampling_steps: int = 50, out: str = ""):
    """Generate an ESMFold2 ensemble and save locally.

    Provide exactly one of:
      --sequence "SEQ"        single protein chain (id A)
      --chains "A:S1,B:S2"    multiple chains (or bare "S1,S2" -> A,B)
      --pdb file.pdb          parse chains from a PDB; --chain-id "X,K"
                              restricts/orders the selection
    """
    if pdb:
        chain_seqs = _pdb_to_chain_seqs(Path(pdb).read_bytes(),
                                        chain_id=chain_id or None)
    elif chains:
        chain_seqs = _parse_chains_arg(chains)
    elif sequence:
        chain_seqs = [("A", sequence)]
    else:
        raise SystemExit("provide one of --sequence, --chains, or --pdb")

    print(f"[local] requesting {n_samples} ESMFold2 samples; "
          f"chains={[(c, len(s)) for c, s in chain_seqs]}")
    data = sample_remote.remote(chain_seqs, name, n_samples=n_samples,
                                num_loops=num_loops,
                                num_sampling_steps=num_sampling_steps)
    out_path = Path(out) if out else Path(f"{name}_esmfold2.npz")
    out_path.write_bytes(data)
    print(f"[local] wrote {out_path} ({len(data)/(1<<20):.1f} MB)")
    print(f"[local] load with: vampnet load_ensemble {name}_esmfold2 "
          f"{out_path} format esmfold2")
