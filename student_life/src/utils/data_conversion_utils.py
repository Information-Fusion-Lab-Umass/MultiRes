import numpy as np
import pandas as pd

from src import definitions
import src.bin.validations as validations


def normalize(data_frame: pd.DataFrame, norm_type="mean",
              df_mean: pd.Series = None, df_std: pd.Series = None,
              df_min: pd.Series = None, df_max: pd.Series = None) -> pd.DataFrame:
    if norm_type == "min_max":
        if df_min is None:
            df_min = data_frame.min()
        if df_max is None:
            df_max = data_frame.max()
        result = (data_frame - df_min) / (df_max - df_min)
    else:
        if df_mean is None:
            df_mean = data_frame.mean()
        if df_std is None:
            df_std = data_frame.std()
        result = (data_frame - df_mean) / df_std

    return result.fillna(0)

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


def adjust_classes_wrt_median(label):
    if label < 2:
        return 0
    elif label > 2:
        return 2
    else:
        return 1


def flatten_matrix(matrix):
    """

    @param matrix: Accepts numpy matrix of list to be flattened.
    @return: Flattened list or Matrix.
    """
    assert isinstance(matrix, np.ndarray) or isinstance(matrix,
                                                        list), "Invalid data type, please give either np.ndarray or a lists."

    if isinstance(matrix, np.ndarray):
        return matrix.flatten()
    else:
        return np.array(matrix).flatten().tolist()


def extract_keys_and_labels_from_dict(data: dict):
    keys = []
    labels = []

    for key in data['data']:
        keys.append(key)
        labels.append(data['data'][key][definitions.LABELS_IDX])

    return keys, labels


def extract_student_ids_from_keys(keys):
    student_ids = []
    for key in keys:
        student_ids.append(extract_student_id_from_key(key))

    return student_ids


def extract_distinct_student_idsfrom_keys(keys):
    return set(extract_student_ids_from_keys(keys))


def extract_student_id_from_key(key):
    return key.split("_")[0]


def extract_actual_missing_and_time_delta_from_raw_data_for_student(raw_data, student_id):
    assert len(raw_data) == 3, \
        "Invalid raw data, it missing one of the following: Actual data, Missing flags or Time Deltas"

    (student_data, missing_data, time_delta) = raw_data
    student_data = student_data[student_data['student_id'] == student_id]
    missing_data = missing_data[missing_data['student_id'] == student_id]
    time_delta = time_delta[time_delta['student_id'] == student_id]

    return student_data, missing_data, time_delta


def extract_keys_of_student_from_data(data: dict, student_id):
    keys = []

    for key in data['data']:
        if str(student_id) == extract_student_id_from_key(key):
            keys.append(key)

    return keys


def extract_labels_for_student_id_form_data(data: dict, student_id):
    student_keys = extract_keys_of_student_from_data(data, student_id)
    labels = []

    for key in student_keys:
        labels.append(data['data'][key][definitions.LABELS_IDX])

    return labels


def get_filtered_keys_for_these_students(*student_id, keys):
    filtered_keys = []
    student_ids = list(student_id)

    for key in keys:
        curr_student = key.split("_")[0]
        if curr_student in student_ids:
            filtered_keys.append(key)

    return filtered_keys


def flatten_data(data: list):
    """

    @param data: Data to be flattened, i.e. the rows will be appended as columns.
    @return: Convert sequences to columns by flattening all rows into a single row.
    """
    assert len(data) == 4, "Missing either of the one in data - Actual data, missing flags, time deltas or label"
    flattened_data_list = []
    # Cannot flatten the labels.
    for i in range(len(data) - 1):
        flattened_data_list.append(flatten_matrix(data[i]))
    # Append the label as well.
    flattened_data_list.append(data[-1])

    return flattened_data_list


def convert_df_to_tuple_of_list_values(*data_frames):
    data_frames_as_list = []
    for df in data_frames:
        data_frames_as_list.append(df.values.tolist())

    return tuple(data_frames_as_list)


def get_indices_list_in_another_list(a, b):
    """

    @param a: List of elements who's indices need to be found.
    @param b: Base list containing superset of a.
    @return: indices of elements of list a in list b.
    """
    indices = []
    for element in a:
        indices.append(b.index(element))

    return indices


def drop_duplicate_indices_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="first")]


def convert_to_string_if_int(value):
    return str(value) if isinstance(value, int) else value


def convert_to_int_if_str(value):
    if value.isdigit():
        return int(value)


def convert_list_of_strings_to_list_of_ints(string_list):
    return [convert_to_int_if_str(x) for x in string_list]


def prepend_ids_with_string(ids, string):
    return [string + str(x) for x in ids]


def tensor_list_to_int_list(tensor_list):
    int_list = []
    for t in tensor_list:
        int_list.append(t.item())

    return int_list
