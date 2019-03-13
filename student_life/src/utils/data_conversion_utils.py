import numpy as np
import pandas as pd

import src.bin.validations as validations


def convert_logical_not_missing_flags(data):
    validations.validate_data_dict_keys(data)

    new_dict = {}
    data_dict = {}

    new_dict['train_ids'] = data['train_ids']
    new_dict['val_ids'] = data['val_ids']
    new_dict['test_ids'] = data['test_ids']

    for key in data['data'].keys():
        mutable_data = list(data['data'][key])
        mutable_data[1] = np.logical_not(np.array(data['data'][key][1])).astype(int).tolist()
        data_dict[key] = tuple(mutable_data)

    new_dict['data'] = data_dict

    return new_dict


def transpose_data(data: list):
    np_data_array = np.array(data, dtype=np.float32)
    return np.transpose(np_data_array)


def get_transposed_data(data: dict):
    validations.validate_data_dict_keys(data)
    for key in data['data']:
        transposed_data = [transpose_data(datum) for datum in data['data'][key]]
        data['data'][key] = tuple(transposed_data)

    return data


def get_mean_for_series(series, mask):
    assert len(series) == len(mask), "Length mismatch of series: {} and mask: {}".format(
        len(series),
        len(mask))
    return np.mean(series[mask.astype(bool)])


def get_mean_for_series(series, mask):
    return np.mean(series[mask.astype(bool)])


def add_mean_vector_to_data(data: dict):
    validations.validate_data_dict_keys(data)
    validations.validate_data_dict_data_len(data)

    for key in data['data']:
        data_list = list(data['data'][key])
        feature_data = data_list[0]
        missing_flags = data_list[1]
        time_delta = data_list[2]
        label = data_list[3]
        mean_vector = [0] * len(feature_data)

        for i in range(len(feature_data)):
            mean_vector[i] = get_mean_for_series(feature_data[i],
                                                 missing_flags[i])

        data_tuple = (feature_data, missing_flags, time_delta, mean_vector, label)

        data['data'][key] = data_tuple

    return data


def normalize(data_frame:pd.DataFrame, norm_type="mean")-> pd.DataFrame:
    if norm_type == "min_max":
        result = (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())
    else:
        result = (data_frame-data_frame.mean())/data_frame.std()

    return result.fillna(0)


def adjust_classes_wrt_median(label):

    if label < 2:
        return 0
    elif label > 2:
        return 2
    else:
        return 1
