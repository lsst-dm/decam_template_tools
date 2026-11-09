#!/usr/bin/bash -l
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -o /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/chunkPsfQa_%j.out
#SBATCH -e /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/chunkPsfQa_%j.err

# Usage: sbatch run-chunk-psf-qa.sh <chunk> [measure_template_psf.py args]
#   e.g. sbatch run-chunk-psf-qa.sh 0
#        sbatch run-chunk-psf-qa.sh 0 --bands g,r
#
# Template PSF QA for one endurance chunk; endurance/submit/run_chunk.sh submits it
# as its last stage. Reads the template_coadds and preliminary_visit_summaries in
# u/<operator>/DECam/endurance/template/chunk<N> and writes
# template_psf_per_patch.csv and template_psf_per_tract.csv to
# $LOGDIR/templatePsf/chunk<N>/. Worker count follows --cpus-per-task above; each
# worker holds one Postgres connection. The pilot took 12.5 min for 7810 patches
# on 8 workers; a full chunk is roughly 10-20x that.

CHUNK=${1:?usage: sbatch run-chunk-psf-qa.sh <chunk> [measure_template_psf.py args]}
shift

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_43/loadLSST.sh && setup lsst_distrib
export REPO=/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo
export LOGDIR=${LOGDIR:-$REPO/logs}
# Same operator name bps uses for u/<operator>/... (see run_chunk.sh).
OPERATOR=$(python -c 'import getpass; print(getpass.getuser())')

OUTDIR=$LOGDIR/templatePsf/chunk$CHUNK
mkdir -p $OUTDIR
LOGFILE=$LOGDIR/templatePsf-chunk$CHUNK.log

date | tee $LOGFILE
python /global/u1/d/descdm/seanmacb/decam_template_tools/verify/measure_template_psf.py \
--repo $REPO \
--collection u/$OPERATOR/DECam/endurance/template/chunk$CHUNK \
--output-dir $OUTDIR --workers ${SLURM_CPUS_PER_TASK:-8} \
"$@" \
2>&1 | tee -a $LOGFILE
status=${PIPESTATUS[0]}
date | tee -a $LOGFILE
exit $status
