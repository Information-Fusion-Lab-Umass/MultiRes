import torch


def validate_no_nans_in_tensor(tensor):
    assert not (tensor != tensor).any(), "null exists in input!"