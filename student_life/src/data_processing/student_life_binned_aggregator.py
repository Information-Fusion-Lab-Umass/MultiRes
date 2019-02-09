"""
Script to generate binned aggregates based on the configuration per feature.
"""
import os
import pandas as pd
import numpy as np

from src.definitions import ROOT_DIR
from src.utils.read_utils import read_yaml
from src.utils.write_utils import df_to_csv
from src.utils.student_utils import getStudentistAfterIgnoring, getStudentsFromFolderNames
from src.data_processing.helpers import get_resampled_aggregated_data, get_flattened_student_data_from_list

kMinimalProcessedDataPath = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
kBinnedOnVarFreqDataPath = os.path.join(ROOT_DIR, "../data/student_life_var_binned_data")
kFeatureConfigFilePath = os.path.join(ROOT_DIR, "configurations/feature_processing.yaml")
kStudentFolderNamePrefix = "student_"
kDefaultResampleFreqKey = "resample_freq_min"
kOWD = os.getcwd()

# Reading Configs.
features_config = read_yaml(kFeatureConfigFilePath)['features']
available_features = features_config.keys()
student_config = read_yaml(kFeatureConfigFilePath)['students']

# Student Processing.
student_to_be_ignored = student_config["student_ignore_list"]
available_students_str = os.listdir(kMinimalProcessedDataPath)
available_students = getStudentsFromFolderNames(kStudentFolderNamePrefix, available_students_str)

# Ignoring the students.
available_students = getStudentistAfterIgnoring(available_students, student_to_be_ignored)

############## Main Loop To Process Data ##################

for student_id in available_students:

    student_data = []

    for idx, feature in enumerate(available_features):
        feature_statistics = pd.DataFrame
        feature_data_path = os.path.join(kMinimalProcessedDataPath,
                                         kStudentFolderNamePrefix + str(student_id),
                                         feature + ".csv")
        feature_data = pd.read_csv(feature_data_path, index_col=[0])
        feature_data.index = pd.to_datetime(feature_data.index)
        aggregated_data = get_resampled_aggregated_data(feature_data, features_config[feature], student_id)
        student_data.append(aggregated_data)

    student_data_flattened = get_flattened_student_data_from_list(student_data, student_id)
    # Converting any missing values to NaN.
    student_data_flattened.replace(to_replace={
        -1: np.nan,
        -1.0: np.nan},
        value=None, inplace=True)
    missing_value_mask = student_data_flattened.isnull()

    ############################### Writing the files to csv #############################
    student_binned_data_dir_path = os.path.join(
        kBinnedOnVarFreqDataPath,
        "student_{}".format(student_id)
    )
    df_to_csv(student_data_flattened, file_name="var_binned_data.csv",
              path_to_folder=student_binned_data_dir_path)

    df_to_csv(missing_value_mask, file_name="missing_values_mask.csv",
              path_to_folder=student_binned_data_dir_path)

    break
