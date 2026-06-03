# OSG long-MD scaffold (v0.6 W3b)

OpenScienceGrid via [OSG Connect](https://osgconnect.net) provides
free, embarrassingly-parallel compute (CPU and opportunistic GPU)
to U.S. academic researchers. This directory contains a HTCondor
submission scaffold for running chimerax-vampnet's Notch1 MD
replicas on OSG.

## Why OSG (vs Modal)

- **Cost**: free. The Modal path costs ~$25 per 100 ns of a Notch1-
  sized system; OSG is $0 once on.
- **Scale**: OSG can hold 10-30 simultaneous independent MD
  replicas easily, perfect for the multi-replica MSM convergence
  story (the v0.5 H2 magnitudes need more replicas, not longer
  individual runs).
- **Latency**: OSG matchmaking can take minutes-to-hours per
  job; not ideal for iterative debug but excellent for fire-and-
  forget production.

## Onboarding

This requires a one-time user action; estimate ~1 week:

1. Apply for an OSG account at <https://osgconnect.net/signup>
   (free; needs a U.S. academic affiliation).
2. Get a project ID assigned (your PI's or self-create one).
3. Verify SSH access to login03.osgconnect.net (or login05).
4. Upload the prepared Notch1 system XMLs + equilibrated PDB to
   OSDF stash:

   ```sh
   # From the OSG login node:
   stashcp /path/to/prepared/notch1_apo_v3/system.xml \
       osdf:///ospool/uc-shared/public/$USER/chimerax-vampnet/prepared/notch1_apo_v3/system.xml
   # Repeat for integrator.xml, state.xml, equilibrated.pdb, anchor_specs.json
   # Also push the local produce.py:
   stashcp /path/to/chimerax-vampnet/md/produce.py \
       osdf:///ospool/uc-shared/public/$USER/chimerax-vampnet/code/produce.py
   ```

## Submitting

From the OSG login node, with this directory rsync'd over:

```sh
cd /home/$USER/chimerax-vampnet/md/osg_md/
mkdir -p logs

# Edit the variables in submit.sub or pass on cmd line:
condor_submit \
    -append "SYSTEM=notch1_apo_v3" \
    -append "STASH_PREFIX=osdf:///ospool/uc-shared/public/$USER/chimerax-vampnet" \
    -append "PROJECT=$YOUR_OSG_PROJECT" \
    -append "REPLICAS=10" \
    -append "NS_PER_REPLICA=100" \
    submit.sub
```

10 replicas × 100 ns each = ~24 hours of opportunistic matchmaking
wall clock for the first replicas to finish; tail replicas
typically land within 4 days.

## Monitoring

```sh
condor_q             # see queued + running jobs
condor_q -hold       # see jobs OSG sent to hold (usually transient evictions)
tail -f logs/replica_0.out
```

## Ingestion

When trajectories land in stash, pull them locally for analysis:

```sh
mkdir -p ~/data/osg_notch1_v0.6/
for r in 0 1 2 3 4 5 6 7 8 9; do
  stashcp osdf:///ospool/uc-shared/public/$USER/chimerax-vampnet/results/notch1_apo_v3/replica_${r}.dcd \
      ~/data/osg_notch1_v0.6/replica_${r}.dcd
done
```

Then re-run the H3 multi-source analysis (`md/notch1_h3_multisource.py`)
with the new replicas added to the MD source path; auto-detection
should pick them up.

## Files

- `submit.sub` — HTCondor submit description
- `runner.sh` — per-job script (one MD replica per invocation)
- `README.md` — this file

## Status

The scaffold ships in v0.6 but execution requires the OSG account
onboarding step above. v0.7 will harvest the long-MD trajectories
into the H3 analysis once they land.
