"""
Script that
"""
import src.data_manager.student_life_var_binned_data_manager as data_manager
import src.utils.data_conversion_utils as conversions
import src.bin.tensorify as tensorify


def get_data_for_gru_d(*student_ids):
    """

    @param raw_data_dict:
    @param student_ids:
    @return: Performs conversions on data dict and return data in a form that can be used to train GRUD.
    """

    data = data_manager.get_data_for_training_in_dict_format(*student_ids)
    data = conversions.convert_logical_not_missing_flags(data)
    data = conversions.get_transposed_data(data)

    return data
