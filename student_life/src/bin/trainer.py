from src.bin import validations


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
