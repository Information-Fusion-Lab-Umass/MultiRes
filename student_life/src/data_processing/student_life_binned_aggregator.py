"""
Script to generate binned aggregates based on the configuration per feature.
"""
import os
import pandas as pd

from src.definitions import ROOT_DIR
from src.utils.read_utils import read_yaml
from src.utils.write_utils import df_to_csv
from src.utils.student_utils import getStudentistAfterIgnoring, getStudentsFromFolderNames
from src.data_processing import helper


MINIMAL_PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
BINNED_ON_VAR_FREQ_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_var_binned_data")
FEATURE_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configurations/feature_processing.yaml")
STUDENT_FOLDER_NAME_PREFIX = "student_"
ORIGINAL_WORKING_DIR = os.getcwd()

# Reading Configs.
FEATURE_CONFIG = read_yaml(FEATURE_CONFIG_FILE_PATH)['features']
AVAILABLE_FEATURE = FEATURE_CONFIG.keys()
STUDENT_CONFIG = read_yaml(FEATURE_CONFIG_FILE_PATH)['students']

# Student Processing.
STUDENTS_TO_BE_IGNORED = STUDENT_CONFIG["student_ignore_list"]
available_students_str = os.listdir(MINIMAL_PROCESSED_DATA_PATH)
available_students = getStudentsFromFolderNames(STUDENT_FOLDER_NAME_PREFIX, available_students_str)

# Ignoring the students.
available_students = getStudentistAfterIgnoring(available_students, STUDENTS_TO_BE_IGNORED)

############## Main Loop To Process Data ##################

for student_id in available_students:

    student_data = []

    for idx, feature in enumerate(AVAILABLE_FEATURE):
        feature_statistics = pd.DataFrame
        feature_data_path = os.path.join(MINIMAL_PROCESSED_DATA_PATH,
                                         STUDENT_FOLDER_NAME_PREFIX + str(student_id),
                                         feature + ".csv")
        feature_data = pd.read_csv(feature_data_path, index_col=[0])
        feature_data.index = pd.to_datetime(feature_data.index)
        aggregated_data = helper.get_resampled_aggregated_data(feature_data, FEATURE_CONFIG[feature], student_id)
        student_data.append(aggregated_data)

    student_data_flattened = helper.get_flattened_student_data_from_list(student_data, student_id)
    student_data_flattened = helper.replace_neg_one_with_nan(student_data_flattened)
    student_data_flattened_processed = helper.remove_days_with_no_stress_label(student_data_flattened)
    missing_value_mask = student_data_flattened_processed.notnull().astype(int)
    time_deltas_min = helper.get_time_deltas_min(student_data_flattened_processed)

    ############################### Writing the files to csv #############################
    student_binned_data_dir_path = os.path.join(
        BINNED_ON_VAR_FREQ_DATA_PATH,
        "student_{}".format(student_id)
    )
    df_to_csv(student_data_flattened_processed, file_name="var_binned_data.csv",
              path_to_folder=student_binned_data_dir_path)
    df_to_csv(missing_value_mask, file_name="missing_values_mask.csv",
              path_to_folder=student_binned_data_dir_path)
    df_to_csv(time_deltas_min, file_name="time_deltas_min.csv",
              path_to_folder=student_binned_data_dir_path)
