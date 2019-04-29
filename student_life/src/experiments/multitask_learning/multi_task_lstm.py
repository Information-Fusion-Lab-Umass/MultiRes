import os
import torch
import tqdm

import src.bin.tensorify as tensorify
import src.utils.data_conversion_utils as conversions
import src.data_manager.student_life_var_binned_data_manager as data_manager
import src.bin.trainer as trainer
from statistics import mean as list_mean

from sklearn import metrics

from torch import nn
from copy import deepcopy
from src import definitions
from src.bin import statistics
from src.data_manager import cross_val
from src.models.multitask_learning import multitask_autoencoder
from src.utils.read_utils import read_pickle
from src.utils import write_utils

feature_list = data_manager.FEATURE_LIST

# ##### Pickle #####
data_file_path = os.path.join(definitions.DATA_DIR,
                              'training_data/shuffled_splits',
                              'training_date_normalized_shuffled_splits_select_features_no_prev_stress_2.pkl')
data = read_pickle(data_file_path)
splits = cross_val.get_k_fod_cross_val_splits_stratified_by_students(data=data, n_splits=5)
print("Splits: ", len(splits))

############ Stats #############
print(statistics.get_train_test_val_label_counts_from_raw_data(data))

################################## Init ##################################
use_historgram = True
autoencoder_bottle_neck_feature_size = 128
autoencoder_num_layers = 1
alpha, beta = 0, 1
decay = 0.0001
first_key = next(iter(data['data'].keys()))
if use_historgram:
    num_features = len(data['data'][first_key][4][0])
else:
    num_features = len(data['data'][first_key][0][0])
num_covariates = len(data['data'][first_key][definitions.COVARIATE_DATA_IDX])
shared_hidden_layer_size = 256
user_dense_layer_hidden_size = 64
num_classes = 3
learning_rate = 0.000001
n_epochs = 350
shared_layer_dropout_prob=0.00
user_head_dropout_prob=0.00

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

print("Num Features:", num_features)
print("Device: ", device)
print("Num_covariates:", num_covariates)

cuda_enabled = torch.cuda.is_available()
tensorified_data = tensorify.tensorify_data_gru_d(deepcopy(data), cuda_enabled)
student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())

# K fold Cross val score.

split_val_scores = []
best_score_epoch_log = []

for split_no, split in enumerate(splits):
    print("Split No: ", split_no)

    best_split_score = -1
    epoch_at_best_score = 0

    tensorified_data['train_ids'] = split['train_ids']
    data['train_ids'] = split['train_ids']

    tensorified_data['val_ids'] = split['val_ids']
    data['val_ids'] = split['val_ids']

    tensorified_data['test_ids'] = []

    validation_user_statistics_over_epochs = []

    class_weights = torch.tensor(statistics.get_class_weights_in_inverse_proportion(data))
    class_weights = torch.tensor([0.93, 0.82, 1])
    print(class_weights)

    model = multitask_autoencoder.MultiTaskAutoEncoderLearner(
        conversions.prepend_ids_with_string(student_list, "student_"),
        num_features,
        autoencoder_bottle_neck_feature_size,
        autoencoder_num_layers,
        shared_hidden_layer_size,
        user_dense_layer_hidden_size,
        num_classes,
        num_covariates,
        shared_layer_dropout_prob,
        user_head_dropout_prob)
    if cuda_enabled:
        model.cuda()
        class_weights = class_weights.cuda()

    reconstruction_criterion = torch.nn.L1Loss(reduction="sum")
    classification_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    for epoch in tqdm.tqdm(range(n_epochs)):

        (train_total_loss, train_total_reconstruction_loss, train_total_classification_loss,
         train_labels, train_preds, train_users) = trainer.evaluate_multitask_learner(tensorified_data,
                                                                                      'train_ids',
                                                                                      num_classes,
                                                                                      model,
                                                                                      reconstruction_criterion,
                                                                                      classification_criterion,
                                                                                      device,
                                                                                      optimizer=optimizer,
                                                                                      alpha=alpha,
                                                                                      beta=beta,
                                                                                      use_histogram=use_historgram)

        (val_total_loss, val_total_reconstruction_loss, val_total_classification_loss,
         val_labels, val_preds, val_users) = trainer.evaluate_multitask_learner(tensorified_data,
                                                                                'val_ids',
                                                                                num_classes,
                                                                                model,
                                                                                reconstruction_criterion,
                                                                                classification_criterion,
                                                                                device,
                                                                                alpha=alpha,
                                                                                beta=beta,
                                                                                use_histogram=use_historgram)

        ######## Appending Metrics ########
        train_label_list = conversions.tensor_list_to_int_list(train_labels)
        train_pred_list = conversions.tensor_list_to_int_list(train_preds)
        val_label_list = conversions.tensor_list_to_int_list(val_labels)
        val_pred_list = conversions.tensor_list_to_int_list(val_preds)

        train_scores = metrics.precision_recall_fscore_support(train_label_list, train_pred_list, average='weighted')
        val_scores = metrics.precision_recall_fscore_support(val_label_list, val_pred_list, average='weighted')

        validation_user_statistics_over_epochs.append(statistics.generate_training_statistics_for_user(val_labels,
                                                                                                       val_preds,
                                                                                                       val_users))

        if val_scores[2] > best_split_score:
            best_split_score = val_scores[2]
            epoch_at_best_score = epoch

        print("Score This Epoch: {} Best Score: {}".format(val_scores[2], best_split_score))

    split_val_scores.append(best_split_score)
    best_score_epoch_log.append(epoch_at_best_score)

print("Avg Cross Val Score: {}".format(list_mean(split_val_scores)))

scores_and_epochs = (split_val_scores, epoch_at_best_score)
scores_and_epochs_file_name = os.path.join(definitions.DATA_DIR, "cross_val_scores/multitask_lstm.pkl")
write_utils.data_structure_to_pickle(scores_and_epochs, scores_and_epochs_file_name)
