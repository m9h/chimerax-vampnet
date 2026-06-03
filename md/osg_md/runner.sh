#!/bin/bash
# Per-job runner for chimerax-vampnet long-MD on OSG.
#
# Called by HTCondor for each replica. Args:
#   $1 SYSTEM         e.g. notch1_apo_v3
#   $2 REPLICA        0-based replica index from $(Process)
#   $3 NS             nanoseconds to simulate
#   $4 STASH_PREFIX   osdf://... root containing prepared/<system>/
#                     and where outputs will be uploaded
#
# Pulls the prepared system+state+integrator from stash, runs
# produce.py for $NS ns, and re-uploads traj.dcd + log.csv + a
# replica.json metadata stub.
#
# Container: docker://nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04
# (declared in submit.sub). Inside the container we install OpenMM
# via pip; the container ships CUDA runtime + cuDNN.

set -euo pipefail

SYSTEM="${1:?missing system}"
REPLICA="${2:?missing replica index}"
NS="${3:?missing ns}"
STASH_PREFIX="${4:?missing stash prefix}"

echo "[osg.runner] system=$SYSTEM replica=$REPLICA ns=$NS"

# OSG worker provides ${OSG_WN_TMP} or a default tmp area.
WORK="${OSG_WN_TMP:-/tmp}/cxv_${SYSTEM}_${REPLICA}"
mkdir -p "$WORK"
cd "$WORK"

# Pull prepared system from stash via stashcp (preferred) or
# osdf-client. Both are available on OSG worker nodes.
for FILE in system.xml integrator.xml state.xml equilibrated.pdb anchor_specs.json; do
    SRC="${STASH_PREFIX}/prepared/${SYSTEM}/${FILE}"
    if command -v stashcp >/dev/null 2>&1; then
        stashcp "$SRC" "./${FILE}" || true
    else
        osdf-client cp "$SRC" "./${FILE}" || true
    fi
done

# Pip-install OpenMM into the container's Python. Plain `pip install
# openmm` pulls a CUDA wheel from the official conda-forge mirror's
# pip channel. OSG workers have network access.
pip install --quiet --user openmm mdtraj numpy

# Move produce.py into the job's scratch (it's not transferred by
# default; OSG users typically ship a small "code.tar.gz" via
# transfer_input_files in submit.sub — for prototyping, fetch from
# the repo's stash mirror).
PRODUCE_PY_URL="${STASH_PREFIX}/code/produce.py"
if command -v stashcp >/dev/null 2>&1; then
    stashcp "$PRODUCE_PY_URL" produce.py
else
    osdf-client cp "$PRODUCE_PY_URL" produce.py
fi

DT_FS=4
STEPS_PER_PS=250
STEPS=$(awk "BEGIN { printf \"%d\", $NS * 1000 * $STEPS_PER_PS }")
DCD_INTERVAL=$(awk "BEGIN { printf \"%d\", 20 * $STEPS_PER_PS }")
REPORT_INTERVAL=$(awk "BEGIN { printf \"%d\", $DCD_INTERVAL / 5 }")

# produce.py expects a prepared_dir with replica_<i>/ as the output
# subdir. Lay out:
#   $WORK/
#     system.xml, integrator.xml, state.xml, equilibrated.pdb
#     replica_<REPLICA>/   (created by produce.py)
python ~/.local/bin/produce.py "$WORK" \
    --replica "$REPLICA" \
    --steps "$STEPS" \
    --report-interval "$REPORT_INTERVAL" \
    --dcd-interval "$DCD_INTERVAL"

# Copy outputs back to job sandbox so HTCondor's transfer_output_files
# can pick them up. submit.sub declares: traj.dcd, log.csv, replica.json
cp "$WORK/replica_${REPLICA}/traj.dcd" ./traj.dcd
cp "$WORK/replica_${REPLICA}/log.csv"  ./log.csv

cat > replica.json <<EOF
{
  "system": "$SYSTEM",
  "replica": $REPLICA,
  "ns": $NS,
  "steps": $STEPS,
  "host": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# Push trajectory back to stash so the main repo can ingest it.
TRAJ_DST="${STASH_PREFIX}/results/${SYSTEM}/replica_${REPLICA}.dcd"
if command -v stashcp >/dev/null 2>&1; then
    stashcp ./traj.dcd "$TRAJ_DST" || echo "[osg.runner] stash upload failed (non-fatal)"
fi

echo "[osg.runner] done"
