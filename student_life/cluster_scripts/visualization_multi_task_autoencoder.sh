#!/usr/bin/env bash
#
#SBATCH --mem=10000
#SBATCH --job-name=viz-mul
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --output=visualization_multi_task_autoencoder-%A.out
#SBATCH --error=visualization_multi_task_autoencoder-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/student_life
PYTHONPATH=../ python -m src.experiments.visualization.multi_task_over_different_no_students
