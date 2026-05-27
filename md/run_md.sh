#!/usr/bin/env bash
# Run a chimerax-vampnet MD command inside the openmm:gb10 container,
# with /data NAS storage and --gpus all for CUDA.
#
# Examples:
#   ./run_md.sh python alanine_dipeptide.py /data/ala
#   ./run_md.sh python prep.py 3i08.pdb /data/notch1_apo
#   ./run_md.sh python produce.py /data/notch1_apo --replica 0 --steps 50000000
set -euo pipefail

MD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT=/data/datasets/chimerax-vampnet
mkdir -p "$DATA_ROOT"

TTY=(); [ -t 0 ] && TTY=(-it)

exec docker run --rm "${TTY[@]}" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$MD_DIR":/workspace \
  -v "$DATA_ROOT":/data \
  -e HOME=/workspace/.containerhome \
  -e OPENMM_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -w /workspace \
  openmm:gb10 "$@"
