#!/usr/bin/env bash
#
#SBATCH --mem=15000
#SBATCH --job-name=student-life-grid-search
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --output=grid_search-%A.out
#SBATCH --error=grid_search-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=1
#SBATCH --time=14-00:00:00
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/student_life
PYTHONPATH=../ python -m src.grid_search.grid_search
