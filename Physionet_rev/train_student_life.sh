#!/usr/bin/env bash
#
#SBATCH --mem=5000
#SBATCH --job-name=student-life-data-dbm
#SBATCH --partition=titanx-short
#SBATCH --output=student_life_trainer-%A.out
#SBATCH --error=student_life_trainer-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/Physionet_rev
PYTHONPATH=../ python train_student_life.py
