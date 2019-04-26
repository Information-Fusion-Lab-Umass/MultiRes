#!/usr/bin/env bash
#
#SBATCH --mem=2000
#SBATCH --job-name=student-life-data-processor
#SBATCH --partition=titanx-short
#SBATCH --output=exploded_binned_data-%A.out
#SBATCH --error=exploded_binned_data-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/student_life
PYTHONPATH=../ python -m src.data_processing.explode_duration_based_features
