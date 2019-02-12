import os
import pathlib

# Defining Root Directory of the project.
ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
USER_HOME = pathlib.Path.home()

MINIMAL_PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
BINNED_ON_VAR_FREQ_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_var_binned_data")
FEATURE_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configurations/feature_processing.yaml")
STUDENT_FOLDER_NAME_PREFIX = "student_"
BINNED_DATA_FILE_NAME = "var_binned_data.csv"
BINNED_DATA_MISSING_VALES_FILE_NAME = "missing_values_mask.csv"
BINNED_DATA_TIME_DELTA_FILE_NAME = "time_deltas_min.csv"
