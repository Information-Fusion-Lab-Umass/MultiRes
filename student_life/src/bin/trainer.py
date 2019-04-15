from src.bin import validations
from src.utils import data_conversion_utils as conversions

HISTOGRAM_IDX_AFTER_TENSORIFY = 2


def validate_key_set_str(key_set: str):
    assert key_set in ['test_ids', 'val_ids', 'train_ids'], "Invalid Key Set. Must be either test or val!"


def evaluate_set(data, key_set: str, model, criterion, optimizer=None, train_covariates=False):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)
    total_loss = 0
    labels = []
    predictions = []

    if not optimizer:
        model.eval()
    else:
        model.train()

    for key in data[key_set]:
        actual_data, covariate_data, train_label = data['data'][key]
        y_pred = model(actual_data, covariate_data) if train_covariates else model(actual_data)
        y_pred_unqueezed = y_pred.unsqueeze(0)
        loss = criterion(y_pred_unqueezed, train_label)
        total_loss += loss.item()

        # Check if training
        if criterion and optimizer:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        labels.append(train_label)
        _, max_idx = y_pred.max(0)
        predictions.append(max_idx)

    return total_loss, labels, predictions


def evaluate_autoencoder_set(data, key_set: str, autoencoder, criterion, optimizer, use_histogram=False):
    validate_key_set_str(key_set)

    total_loss = 0
    decoded_outputs = {}

    for key in data[key_set]:
        if use_histogram:
            input_seq = data['data'][key][HISTOGRAM_IDX_AFTER_TENSORIFY]
        else:
            input_seq = data['data'][key][0][0].unsqueeze(0)

        decoded_output = autoencoder(input_seq)
        decoded_outputs[key] = decoded_output

        loss = criterion(input_seq, decoded_output)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss, decoded_outputs


def evaluate_multitask_learner(data,
                               key_set: str,
                               multitask_lerner_model,
                               reconstruction_criterion,
                               classification_criterion,
                               optimizer=None,
                               alpha=1,
                               beta=1,
                               use_histogram=False):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_reconstruction_loss = 0
    total_classification_loss = 0
    total_joint_loss = 0

    labels = []
    predictions = []
    users = []

    if not optimizer:
        multitask_lerner_model.eval()
    else:
        multitask_lerner_model.train()

    for key in data[key_set]:
        student_id = conversions.extract_student_id_from_key(key)
        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]
        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            actual_data = histogram_data.unsqueeze(0)
        decoded_output, y_pred = multitask_lerner_model(student_key, actual_data, covariate_data)

        reconstruction_loss = reconstruction_criterion(actual_data, decoded_output)
        total_reconstruction_loss += reconstruction_loss.item()

        classification_loss = classification_criterion(y_pred, train_label)
        total_classification_loss += classification_loss.item()

        joint_loss = alpha * reconstruction_loss + beta * classification_loss
        total_joint_loss += joint_loss.item()

        # Check if training
        if optimizer:
            multitask_lerner_model.zero_grad()
            joint_loss.backward()
            optimizer.step()

        labels.append(train_label)
        y_pred_squeezed = y_pred.squeeze(0)
        _, max_idx = y_pred_squeezed.max(0)
        predictions.append(max_idx)
        users.append(student_id)

    return total_joint_loss, total_reconstruction_loss, total_classification_loss, labels, predictions, users


def evaluate_multitask_lstm_learner(data,
                               key_set: str,
                               multitask_lerner_model,
                               classification_criterion,
                               optimizer=None,
                               use_histogram=False):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_classification_loss = 0

    labels = []
    predictions = []
    users = []

    if not optimizer:
        multitask_lerner_model.eval()
    else:
        multitask_lerner_model.train()

    for key in data[key_set]:
        student_id = conversions.extract_student_id_from_key(key)
        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]
        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            actual_data = histogram_data.unsqueeze(0)
        y_pred = multitask_lerner_model(student_key, actual_data, covariate_data)

        classification_loss = classification_criterion(y_pred, train_label)
        total_classification_loss += classification_loss.item()

        # Check if training
        if optimizer:
            multitask_lerner_model.zero_grad()
            classification_loss.backward()
            optimizer.step()

        labels.append(train_label)
        y_pred_squeezed = y_pred.squeeze(0)
        _, max_idx = y_pred_squeezed.max(0)
        predictions.append(max_idx)
        users.append(student_id)

    return total_classification_loss, labels, predictions, users


def is_reconstruction_loss_available(y_pred):
    if isinstance(y_pred, tuple) and len(y_pred) == 2:
        return True
    return False
