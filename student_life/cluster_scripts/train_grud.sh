#!/usr/bin/env bash
#
#SBATCH --mem=10000
#SBATCH --job-name=student-life-data-processor
#SBATCH --partition=titanx-short
#SBATCH --output=gru_d-%A.out
#SBATCH --error=gru_d-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/student_life
PYTHONPATH=../ python -m src.experiments.grud.train
