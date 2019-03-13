import torch

from src.bin import validations


def get_data_and_label_tensor(data: dict, key, cuda_enabled):
    """

    @param data: Data dict containing the data in our rich data structure.
    @param key: Key in the data, usually time series key.
    @param  cuda_enabled: If true, returns cuda tensors.
    @return: Returns tensors that can be used for training on the models.
    """
    validations.validate_data_dict_keys(data)
    assert len(data['data'][key]) == 4, \
        "Missing one of the following 'Actual Data', 'Missing Flags', 'Time Deltas', 'label' "

    tensor_data = torch.tensor(list(data['data'][key][:3]), dtype=torch.float)
    train_label = torch.from_numpy(data['data'][key][3]).item()
    train_label = torch.tensor([train_label], dtype=torch.long)

    if cuda_enabled:
        tensor_data.cuda()
        train_label.cuda()

    return tensor_data, train_label


def tensorify_data_gru_d(data: dict, cuda_enabled=False):
    """

    @param data: Data dictionary that needs to be converted to tensors in GRUD style of data.
    @param cuda_enabled: If true, will convert data into cuda tensors.
    @return: Return Data dictionary with tensors which can be used to train.
    """
    validations.validate_data_dict_keys(data)
    for key in data['data'].keys():
        data['data'][key] = get_data_and_label_tensor(data, key, cuda_enabled)

    return data
