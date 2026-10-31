#!/usr/bin/bash -l
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -o /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/calibQa_%j.out
#SBATCH -e /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/calibQa_%j.err

# Usage: sbatch run-calib-qa.sh <bias|flat|fringe> <tag> [run_<type>_qa.py args]
#   e.g. sbatch run-calib-qa.sh bias certified
#        sbatch run-calib-qa.sh bias 2016 --collection u/seanmacb/DECam/calib/ELAIS_pilot/bias/2016
#
# Writes the table, summary and figures to $LOGDIR/calibQa/<type>/<tag>/. Worker count
# follows --cpus-per-task above; each worker holds one Postgres connection and about
# 0.5 GB per detector being processed.

TYPE=${1:?usage: sbatch run-calib-qa.sh <bias|flat|fringe> <tag> [run_<type>_qa.py args]}
TAG=${2:?usage: sbatch run-calib-qa.sh <bias|flat|fringe> <tag> [run_<type>_qa.py args]}
shift 2

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_43/loadLSST.sh && setup lsst_distrib
export REPO=/global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo
export LOGDIR=$REPO/logs

OUTDIR=$LOGDIR/calibQa/$TYPE/$TAG
mkdir -p $OUTDIR
LOGFILE=$LOGDIR/calibQa-$TYPE-$TAG.log

date | tee $LOGFILE
python /global/u1/d/descdm/seanmacb/decam_template_tools/verify/calibs/run_${TYPE}_qa.py \
--repo $REPO \
--out $OUTDIR --workers ${SLURM_CPUS_PER_TASK:-8} \
"$@" \
2>&1 | tee -a $LOGFILE
date | tee -a $LOGFILE
