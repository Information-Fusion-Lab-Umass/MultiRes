"""
Script to generate binned aggregates based on the configuration per feature.
"""
import os
import pandas as pd

from src.definitions import ROOT_DIR
from src.utils.read_utils import read_yaml
from src.data_processing.helpers import get_aggregation_rule, get_resampled_aggregated_data
from src.utils.student_utils import getStudentistAfterIgnoring, getStudentsFromFolderNames

kMinimalProcessedDataPath = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
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
    # Features are aggregated into this.
    stress_data_path = os.path.join(kMinimalProcessedDataPath,
                                    kStudentFolderNamePrefix + str(student_id),
                                    "stress_details.csv")
    stress_data = pd.read_csv(stress_data_path, index_col=[0])
    stress_data.index = pd.to_datetime(stress_data.index)

    student_data = []

    for idx, feature in enumerate(available_features):
        feature_statistics = pd.DataFrame
        feature_data_path = os.path.join(kMinimalProcessedDataPath,
                                         kStudentFolderNamePrefix+str(student_id),
                                         feature+".csv")
        feature_data = pd.read_csv(feature_data_path, index_col=[0])
        feature_data.index = pd.to_datetime(feature_data.index)
        aggregated_data = get_resampled_aggregated_data(feature_data, features_config[feature], student_id)
        student_data.append(aggregated_data)

    break
