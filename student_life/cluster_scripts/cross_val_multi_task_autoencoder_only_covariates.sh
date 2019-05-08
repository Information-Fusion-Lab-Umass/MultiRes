#!/usr/bin/env bash
#
#SBATCH --mem=5000
#SBATCH --job-name=only_co-variates
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --output=cross_val_multi_task_autoencoder_only_covariates-%A.out
#SBATCH --error=cross_val_multi_task_autoencoder_only_covariates-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/student_life
PYTHONPATH=../ python -m src.experiments.multitask_learning.multi_task_covariate
