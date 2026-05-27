# MD workflow for chimerax-vampnet

Generates the conformational ensembles consumed by the
`vampnet load_ensemble` ChimeraX command. All scripts run inside the
`openmm:gb10` container (conda-forge OpenMM with CUDA 12).

## Build the container (once)

```
cd /home/mhough/Workspace/chimerax-vampnet/md
docker build -t openmm:gb10 .
```

The image bundles micromamba + conda-forge `openmm` + `pdbfixer` and
links against the on-host CUDA driver via `nvidia-container-toolkit`
at run time.

## Tier-1 validation: alanine dipeptide

```
./run_md.sh python alanine_dipeptide.py /data/ala
```

Runs 50 ns of Ace-Ala-Nme in implicit GBn2 solvent (no PME) at 300 K.
~1 hr on a GB10. Output: `/data/datasets/chimerax-vampnet/ala/traj.dcd`
and `ala_dipeptide.pdb`. The bundle should recover the 5 canonical
Ramachandran basins (α-R, β, P_II, α-L, γ).

## Tier-2 validation: Notch1 NRR (PDB 3I08, apo)

```
# Prep (PDBFixer + solvate + minimize + NVT/NPT equilibration ~1 hr GB10)
./run_md.sh python prep.py 3i08.pdb /data/notch1_apo

# 3 replicas x 100 ns each (~16 hr each on a GB10)
for r in 0 1 2; do
  ./run_md.sh python produce.py /data/notch1_apo --replica $r --steps 50000000
done
```

## Tier-2 validation: Notch1 NRR + Fab (PDB 3L95, holo)

```
./run_md.sh python prep.py 3l95.pdb /data/notch1_holo
for r in 0 1 2; do
  ./run_md.sh python produce.py /data/notch1_holo --replica $r --steps 50000000
done
```

## Outputs

Each MD run writes:

  /data/datasets/chimerax-vampnet/<system>/replica_<r>/traj.dcd  -- 50k frames at 10 ps stride
  /data/datasets/chimerax-vampnet/<system>/replica_<r>/log.csv   -- temperature, density, speed, ETA
  /data/datasets/chimerax-vampnet/<system>/replica_<r>/checkpoint.chk  -- resume point

These are then loaded into ChimeraX via the bundle:

  open /data/datasets/chimerax-vampnet/notch1_apo/equilibrated.pdb
  vampnet load_ensemble #1 ~/data/notch1_apo/replica_0/traj.dcd source=md
  vampnet load_ensemble #1 ~/data/notch1_apo/replica_1/traj.dcd source=md
  vampnet load_ensemble #1 ~/data/notch1_apo/replica_2/traj.dcd source=md
  vampnet fit nStates=4 lag=20 features=ca_distances
