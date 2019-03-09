"""
Stores all validations required by the Lib.
"""
import pandas as pd

DATA_DICT_KEYS = ['data', 'train_ids', 'test_ids', 'val_ids']
DATA_TUPLE_LEN = 4


def validate_student_id_in_data(*data: pd.DataFrame):
    for df in data:
        assert "student_id" in df.columns, "Invalid data. missing column 'student_id'."


def validate_config_key(*keys: str, config):
    for key in keys:
        assert key in config, "Invalid config!. Key: {} not present in config.".format(key)


def validate_student_id_in_data_as_first_col(*data: pd.DataFrame):
    validate_student_id_in_data(*data)

    for df in data:
        assert "student_id" == df.columns[0], "First Column in DataFrame is not 'student_id'."


def validate_data_integrity_for_len(*data_frame: pd.DataFrame):
    data_frames = list(data_frame)

    for df in data_frames[1:]:
        assert len(df) == len(data_frame[0]), "Lengths of the DataFrame do not match."


def validate_data_dict_keys(data_dict):
    assert all([k in DATA_DICT_KEYS for k in data_dict.keys()])


def validate_data_dict_data_len(data_dict):
    validate_data_dict_keys(data_dict)
    first_key = next(iter(data_dict['data'].keys()))
    assert len(data_dict['data'][first_key]) == DATA_TUPLE_LEN, \
        "More elements in data tuple. Expected: {} found: {}".format(DATA_TUPLE_LEN,
                                                                     len(data_dict['data'][first_key]))
