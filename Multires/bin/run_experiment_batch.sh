# Change this for new experiments
hidden_dim_slow='200'
hidden_dim_moderate='200'
hidden_dim_fast='200'
epochs='200'
partition='1080ti-long'

#data_vals=('preprocessed_easy_0.3_residual_set_1.pkl' 'preprocessed_easy_no_residual_set_1.pkl' 'preprocessed_easy_residual_set_1.pkl')
#attn_vals=('True' 'False')
#layer_vals=('2' '1')

data_vals=('preprocessed_TBM_residual_set_1.pkl' 'preprocessed_TBM_residual_set_2.pkl' 'preprocessed_TBM_residual_set_3.pkl')
attn_vals=('False')
layer_vals=('2')
lr_vals=('1e-4')
opt=('adam')

for data in "${data_vals[@]}"
do
    for lr in "${lr_vals[@]}"
    do
        for use_second_attention in "${attn_vals[@]}"
        do
            for layers in "${layer_vals[@]}"
            do
                model_name=$use_second_attention-$layers-$lr-$hidden_dim_slow-$hidden_dim_moderate-$hidden_dim_fast-$data
                echo $model_name
                jobname=$model_name

                sbatch -p $partition -e "../stdout/"$jobname.err -o "../stdout/"$jobname.out -J $jobname run_single_experiment.sh \
                    $data \
                    $use_second_attention \
                    $hidden_dim_slow \
                    $hidden_dim_moderate \
                    $hidden_dim_fast \
                    $model_name \
                    $epochs \
                    $layers \
                    $opt \
                    $lr
            done
        done
    done
done












