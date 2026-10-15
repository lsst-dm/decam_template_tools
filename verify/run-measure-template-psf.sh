#!/usr/bin/bash -l
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH -A m5042
#SBATCH -C cpu
#SBATCH -o /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/templatePsf_%j.out
#SBATCH -e /global/cfs/cdirs/lsst/production/gen3/rubin/decam/repo/logs/templatePsf_%j.err

# Usage: sbatch run-measure-template-psf.sh <event> [extra measure_template_psf.py args]
#   e.g. sbatch run-measure-template-psf.sh S250830bp
#        sbatch run-measure-template-psf.sh S250830bp --tracts 195,196 --bands i,z
#
# Reads DECam/templates/<event> and writes template_psf_per_patch.csv and
# template_psf_per_tract.csv to $LOGDIR/templatePsf/<event>/. Worker count follows
# --cpus-per-task above; each worker holds one Postgres connection.

WEEKLY=${1:?usage: sbatch run-measure-template-psf.sh <event> [measure_template_psf.py args]}
shift

source /global/homes/s/seanmacb/startup_desc_commands.sh

OUTDIR=$LOGDIR/templatePsf/$WEEKLY
mkdir -p $OUTDIR
LOGFILE=$LOGDIR/templatePsf-$WEEKLY.log

date | tee $LOGFILE
python /global/homes/s/seanmacb/decam_template_tools/verify/measure_template_psf.py \
--repo $REPO \
--collection u/seanmacb/DECam/calib/ELAIS_pilot_w40/template/bestSeeing_1_8 \
--output-dir $OUTDIR --workers 8\
"$@" \
2>&1 | tee -a $LOGFILE
date | tee -a $LOGFILE
