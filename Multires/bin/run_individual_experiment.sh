#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruppaal@cs.umass.edu
#SBATCH --partition=titanx-long
#SBATCH --output=r8-%A.out
#SBATCH --error=r8-%A.err

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

source activate multires
cd /home/ruppaal/Work/FusionLab/MultiRes/Multires/src

data='preprocessed_hard_0.3_residual_set_1_unit_std-dev.pkl'
use_second_attention='True'
hidden_dim_slow='150'
hidden_dim_moderate='200'
hidden_dim_fast='70'
layers='1'

model_name=$data-$use_second_attention-$layers-$hidden_dim_slow-$hidden_dim_moderate-$hidden_dim_fast
echo "sup"
python \[Physionet\]SignalSplit.py --data $data \
                     --use_second_attention $use_second_attention \
                     --hidden_dim_slow $hidden_dim_slow \
                     --hidden_dim_moderate $hidden_dim_moderate \
                     --hidden_dim_fast $hidden_dim_fast \
                     --model_name $model_name \
                     --layers $layers \
                     > $model_name.outerr 2>&1


#python \[Physionet\]SignalSplit.py --data "preprocessed_easy_residual_set_1.pkl" --use_second_attention False --hidden_dim_slow 150 --hidden_dim_moderate 200 --hidden_dim_fast 70 --model_name test
