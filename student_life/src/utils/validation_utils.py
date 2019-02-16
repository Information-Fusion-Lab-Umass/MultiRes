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
