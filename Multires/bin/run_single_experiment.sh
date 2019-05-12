#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

source activate multires
cd /home/ruppaal/Work/FusionLab/MultiRes/Multires/src

data=$1
use_second_attention=$2
hidden_dim_slow=$3
hidden_dim_moderate=$4
hidden_dim_fast=$5
model_name=$6
epochs=$7
layers=$8
opt=$9
lr=${10}

python \[Physionet\]SignalSplit.py --data $data \
                     --use_second_attention $use_second_attention \
                     --hidden_dim_slow $hidden_dim_slow \
                     --hidden_dim_moderate $hidden_dim_moderate \
                     --hidden_dim_fast $hidden_dim_fast \
                     --model_name $model_name \
                     --epochs $epochs \
                     --layers $layers \
                     --opt $opt \
                     --lr $lr



#python \[Physionet\]SignalSplit.py --data "preprocessed_easy_residual_set_1.pkl" --use_second_attention False --hidden_dim_slow 150 --hidden_dim_moderate 200 --hidden_dim_fast 70 --model_name test
