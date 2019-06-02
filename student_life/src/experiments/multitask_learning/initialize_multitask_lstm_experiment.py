from src.experiments.multitask_learning.experiment_imports import *
import inspect

# Derive Model name from the experiment script calling this module.
frame_info = inspect.stack()[-1]
file_name = frame_info[1]
model_name = os.path.basename(file_name).replace(".py", "")
print("Model Name:", model_name)

##### Read Data and Statistics #####
data_file_path = os.path.join(definitions.SHUFFLED_DATA_ROOT,
                              'training_data_normalized_no_prev_stress_students_greater_than_40_labels.pkl')
data = read_pickle(data_file_path)
splits = cross_val.get_k_fod_cross_val_splits_stratified_by_students(data=data, n_splits=5)
print("Number of Splits: ", len(splits))
print(statistics.get_train_test_val_label_counts_from_raw_data(data))

##### Model Configs #####
(use_histogram,
 autoencoder_bottle_neck_feature_size,
 autoencoder_num_layers,
 shared_hidden_layer_size,
 user_dense_layer_hidden_size,
 num_classes,
 decay,
 shared_layer_dropout_prob,
 user_head_dropout_prob,
 alpha,
 beta,
 learning_rate,
 n_epochs) = model_config_loader.load_static_configs_for_lstm_n_multitask_models(model_name)

(num_features,
 num_covariates,
 device,
 class_weights,
 cuda_enabled,
 student_list) = model_config_loader.load_derived_configs_for_lstm_n_multitask_models(use_histogram, data)

tensorified_data = tensorify.tensorify_data_gru_d(deepcopy(data), cuda_enabled)

print_utils.print_experiment_output(num_features=num_features,
                                    device=device,
                                    num_covariates=num_covariates,
                                    learning_rate=learning_rate,
                                    class_weights=class_weights)

# K fold Cross val score.
split_val_scores = []
best_score_epoch_log = []
best_models = []
