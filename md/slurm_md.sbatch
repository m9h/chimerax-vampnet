#!/bin/bash
# DGX Spark Slurm sbatch script for long-timeframe Notch1 MD (v0.6 W3a).
#
# The v0.5 H2 magnitudes are blocked by the 100-ns Modal MD horizon
# but spending $250-500 in Modal to reach 1 us per replica was
# explicitly de-scoped by the user. The DGX Spark is already owned;
# its GB10 GPU runs ~100 ns/day on a Notch1-NRR-sized system. Queue
# 3 replicas of 1 us each here and harvest in v0.7.
#
# Usage:
#   sbatch md/slurm_md.sbatch <system> <replica>  [<ns>]
# Examples:
#   sbatch md/slurm_md.sbatch notch1_apo_v3  3 1000   # replica 3, 1 us
#   sbatch md/slurm_md.sbatch notch1_apo_v3  4 1000   # replica 4, 1 us
#   sbatch md/slurm_md.sbatch notch1_holo_v3 3 1000
#
# Replica indices >= 3 are used so they don't collide with the existing
# v0.3 0/1/2 replicas already on the Modal volume / Zenodo deposit.
#
# Prerequisites (one-time, before sbatch):
#   1. Build an MD-capable env at $PROJECT_ROOT/.venv-md/. The
#      micromamba recipe in md/modal_md.py is the reference:
#
#        micromamba create -p .venv-md -c conda-forge -y \
#          python=3.13 openmm=8.5.1 openmmtools pdbfixer numpy scipy \
#          cuda-version=12.9
#
#      Or wrap that as `bash md/install_dgx_md_env.sh`.
#   2. Stage the prepared system to a local path. The v0.3 deposit at
#      /data/datasets/chimerax-vampnet/zenodo-v0.3-analysis-ready/ has
#      the system+state XMLs needed.
#
#SBATCH --job-name=cxv_md
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0
#SBATCH --output=slurm_md_%j.out
#SBATCH --error=slurm_md_%j.err

set -euo pipefail

SYSTEM=${1:?usage: sbatch slurm_md.sbatch <system> <replica> [<ns>]}
REPLICA=${2:?usage: sbatch slurm_md.sbatch <system> <replica> [<ns>]}
NS=${3:-1000}

PROJECT_ROOT="/home/mhough/Workspace/chimerax-vampnet"
PREPARED_ROOT="${PREPARED_ROOT:-/data/datasets/chimerax-vampnet/notch1_modal}"
RESULTS_ROOT="${RESULTS_ROOT:-/data/datasets/chimerax-vampnet/notch1_v0.6_long}"
MD_PYTHON="${MD_PYTHON:-$PROJECT_ROOT/.venv-md/bin/python}"

if [[ ! -x "$MD_PYTHON" ]]; then
    echo "MD env not found at $MD_PYTHON. Set MD_PYTHON or build the env"
    echo "via the micromamba recipe in the file header."
    exit 1
fi

# DGX Spark dt budget: 4 fs HMR at GB10 → ~100 ns/day for Notch1 NRR.
# 1000 ns = ~10 days of wall clock. Slurm time=0 means no walltime
# limit; the job will run to completion.

PREPARED_DIR="$PREPARED_ROOT/$SYSTEM"
OUT_REPLICA="$RESULTS_ROOT/$SYSTEM/replica_${REPLICA}"
mkdir -p "$OUT_REPLICA"

# Stage prepared XMLs into the output location so produce.py finds them
# alongside its replica subdir.
RESULTS_PREPARED="$RESULTS_ROOT/$SYSTEM"
mkdir -p "$RESULTS_PREPARED"
for f in system.xml integrator.xml state.xml equilibrated.pdb anchor_specs.json; do
    if [[ -f "$PREPARED_DIR/$f" && ! -f "$RESULTS_PREPARED/$f" ]]; then
        cp -v "$PREPARED_DIR/$f" "$RESULTS_PREPARED/$f"
    fi
done

DT_FS=4
STEPS_PER_PS=$(awk "BEGIN { print 1000.0/$DT_FS }")  # 250 at 4 fs
STEPS=$(awk "BEGIN { printf \"%d\", $NS * 1000 * $STEPS_PER_PS }")
DCD_INTERVAL=$(awk "BEGIN { printf \"%d\", 20 * $STEPS_PER_PS }")   # 20 ps stride
REPORT_INTERVAL=$(awk "BEGIN { printf \"%d\", $DCD_INTERVAL / 5 }")

echo "=== chimerax-vampnet v0.6 long-MD on DGX Spark ==="
echo "  system:         $SYSTEM"
echo "  replica:        $REPLICA"
echo "  ns:             $NS"
echo "  prepared:       $RESULTS_PREPARED"
echo "  output:         $OUT_REPLICA"
echo "  steps:          $STEPS"
echo "  dcd_interval:   $DCD_INTERVAL  (every $((DCD_INTERVAL / STEPS_PER_PS * 1)) ps approx)"
echo "  python:         $MD_PYTHON"
echo "  GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "==================================================="

cd "$PROJECT_ROOT"
exec "$MD_PYTHON" "$PROJECT_ROOT/md/produce.py" "$RESULTS_PREPARED" \
    --replica "$REPLICA" \
    --steps "$STEPS" \
    --report-interval "$REPORT_INTERVAL" \
    --dcd-interval "$DCD_INTERVAL"
