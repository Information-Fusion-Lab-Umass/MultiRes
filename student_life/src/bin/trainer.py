from src.bin import validations


def evaluate_set(data, key_set: str, model, criterion, optimizer=None):
    validations.validate_data_dict_keys(data)
    assert key_set in ['test_ids', 'val_ids', 'train_ids'], "Invalid Key Set. Must be either test or val!"

    total_loss = 0
    labels = []
    predictions = []
    for key in data[key_set]:
        actual_data, covariate_data, train_label = data['data'][key]
        y_pred = model(actual_data)
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
