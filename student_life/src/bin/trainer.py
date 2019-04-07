from src.bin import validations
from src.utils import data_conversion_utils as conversions


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


def evaluate_autoencoder_set(data, key_set: str, autoencoder, criterion, optimizer):
    validate_key_set_str(key_set)

    total_loss = 0
    decoded_outputs = {}

    for key in data[key_set]:
        input_seq = data['data'][key][0][0].unsqueeze(0)
        decoded_output = autoencoder(input_seq)
        decoded_outputs[key] = decoded_output

        loss = criterion(input_seq, decoded_output)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    # print("Total Loss:", total_loss)

    return total_loss, decoded_outputs


def evaluate_multitask_learner(data,
                               key_set: str,
                               multitask_lerner_model,
                               reconstruction_criterion,
                               classification_criterion,
                               optimizer=None,
                               alpha=1,
                               beta=1):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_reconstruction_loss = 0
    total_classification_loss = 0
    total_loss = 0

    labels = []
    predictions = []

    if not optimizer:
        multitask_lerner_model.eval()
    else:
        multitask_lerner_model.train()

    for key in data[key_set]:

        student_key = 'student_' + str(conversions.extract_student_id_from_key(key))
        actual_data, covariate_data, train_label = data['data'][key]
        actual_data = actual_data[0].unsqueeze(0)
        decoded_output, y_pred = multitask_lerner_model(student_key, actual_data, covariate_data)

        reconstruction_loss = reconstruction_criterion(actual_data, decoded_output)
        total_reconstruction_loss += reconstruction_loss.item()

        classification_loss = classification_criterion(y_pred, train_label)
        total_classification_loss += classification_loss.item()

        total_loss = alpha * reconstruction_loss + beta * classification_loss

        # Check if training
        if optimizer:
            multitask_lerner_model.zero_grad()
            total_loss.backward()
            optimizer.step()

        labels.append(train_label)
        y_pred_squeezed = y_pred.squeeze(0)
        _, max_idx = y_pred_squeezed.max(0)
        predictions.append(max_idx)

    return total_loss, total_reconstruction_loss, total_classification_loss, labels, predictions
