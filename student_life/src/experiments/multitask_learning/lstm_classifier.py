from src.experiments.multitask_learning.initialize_multitask_lstm_experiment import *

feature_list = data_manager.FEATURE_LIST

for split_no, split in enumerate(splits):
    print("Split No: ", split_no)

    best_split_score = -1
    epoch_at_best_score = 0
    best_model = None

    tensorified_data['train_ids'] = split['train_ids']
    data['train_ids'] = split['train_ids']

    tensorified_data['val_ids'] = split['val_ids']
    data['val_ids'] = split['val_ids']

    tensorified_data['test_ids'] = []

    validation_user_statistics_over_epochs = []

    model = autoencoder_classifier.AutoEncoderClassifier(
        conversions.prepend_ids_with_string(student_list, "student_"),
        num_features,
        autoencoder_bottle_neck_feature_size,
        autoencoder_num_layers,
        shared_hidden_layer_size,
        user_dense_layer_hidden_size,
        num_classes,
        num_covariates,
        shared_layer_dropout_prob,
        user_head_dropout_prob,
        bidirectional=bidirectional)

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
                                                                                      use_histogram=use_histogram)

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
                                                                                use_histogram=use_histogram)

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
            best_model = deepcopy(model)

        print("Split: {} Score This Epoch: {} Best Score: {}".format(split_no, val_scores[2], best_split_score))

    split_val_scores.append(best_split_score)
    best_score_epoch_log.append(epoch_at_best_score)
    best_models.append(deepcopy(best_model))

print("Avg Cross Val Score: {}".format(list_mean(split_val_scores)))
print("alpha: {} Beta: {}".format(alpha, beta))
print("Data File Path:", data_file_path)
max_idx = split_val_scores.index(max(split_val_scores))

scores_and_epochs = (split_val_scores, epoch_at_best_score)
scores_and_epochs_file_name = os.path.join(definitions.DATA_DIR, "cross_val_scores/lstm_classifier.pkl")
write_utils.data_structure_to_pickle(scores_and_epochs, scores_and_epochs_file_name)

# Saving Best model
model_file_name = "saved_models/lstm.model"
model_file_name = os.path.join(definitions.DATA_DIR, model_file_name)
torch.save(best_models[max_idx], model_file_name)
