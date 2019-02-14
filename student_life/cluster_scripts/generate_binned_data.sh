#!/usr/bin/env bash
#
#SBATCH --mem=2000
#SBATCH --job-name=student-life-data-processor
#SBATCH --partition=titanx-short
#SBATCH --output=var_binned_data-%A.out
#SBATCH --error=var_binned_data-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

PYTHONPATH=../ python src.data_processing.student_life_binned_aggregator
