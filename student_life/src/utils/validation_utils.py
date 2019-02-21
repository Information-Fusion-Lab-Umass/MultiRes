"""
Stores all validations required by the Lib.
"""
import pandas as pd


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
