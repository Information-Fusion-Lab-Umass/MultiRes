#!/usr/bin/env bash
#
#SBATCH --mem=2000
#SBATCH --job-name=new_data
#SBATCH --partition=titanx-long
#SBATCH --output=new_data-%A.out
#SBATCH --error=new_data-%A.err
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=4
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/gypsum-jobs.txt

cd ~/projects/MultiRes/student_life
PYTHONPATH=../ python -m src.data_manager.generate_data
