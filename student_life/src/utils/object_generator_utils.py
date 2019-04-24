import torch


def get_empty_data_dict():
    data = dict()
    data["data"] = {}
    data["train_ids"] = []
    data["val_ids"] = []
    data["test_ids"] = []

    return data


def get_tensor_on_correct_device(data:list):
    """

    @param data: The list of lists that contain the data for the tensor in PyTorch format.
    @return: New created tensor, it is on the GPU when CUDA is availble else CPU.
    """
    new_tensor = torch.Tensor(data)
    if torch.cuda.is_available():
        return new_tensor.cuda()

    return new_tensor
