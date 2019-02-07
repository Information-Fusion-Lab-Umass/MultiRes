"""
Script to generate binned aggregates based on the configuration per feature.
"""
import os
import pandas as pd

from definitions import ROOT_DIR
from src.utils.read_utils import read_yaml
from src.utils.student_utils import getStudentistAfterIgnoring, getStudentsFromFolderNames

kMinimalProcessedDataPath = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
kFeatureConfigFilePath = os.path.join(ROOT_DIR, "configurations/feature_processing.yaml")
kStudentFolderNamePrefix = "student_"
kOWD = os.getcwd()

# Reading Configs.
feature_config = read_yaml(kFeatureConfigFilePath)['features']
available_features = feature_config.keys()
student_config = read_yaml(kFeatureConfigFilePath)['students']

# Student Processing.
student_to_be_ignored = student_config["student_ignore_list"]
available_students_str = os.listdir(kMinimalProcessedDataPath)
available_students = getStudentsFromFolderNames(kStudentFolderNamePrefix, available_students_str)

# Ignoring the students.
available_students = getStudentistAfterIgnoring(available_students, student_to_be_ignored)

for student_id in available_students:

    for feature in available_features:

        feature_data_path = os.path.join(kMinimalProcessedDataPath,
                                         kStudentFolderNamePrefix+str(student_id),
                                         feature+".csv")
        feature_data = pd.read_csv(feature_data_path, index_col=[0])

        print(feature_data.head(5))


        break



