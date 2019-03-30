import os
import pathlib

from src.utils.read_utils import read_yaml

# Defining Root Directory of the project.
ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
USER_HOME = pathlib.Path.home()

# File and Key Names
STUDENT_FOLDER_NAME_PREFIX = "student_"
BINNED_DATA_FILE_NAME = "var_binned_data"
BINNED_DATA_MISSING_VALES_FILE_NAME = "missing_values_mask"
BINNED_DATA_TIME_DELTA_FILE_NAME = "time_deltas_min"

# Config File Path
FEATURE_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configurations/feature_processing.yaml")
DATA_MANAGER_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configurations/data_manager_config.yaml")
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configurations/model_config.yaml")

# Frequency constants
DEFAULT_BASE_FREQ = '1 min'
DEFAULT_EXPLODING_BASE_FREQ = '1 min'

# Data manager config Keys
VAR_BINNED_DATA_MANAGER_ROOT = "student_life_var_binned_data"

# Universal Config Keys.
STUDENT_LIST_CONFIG_KEY = "student_list"
FEATURE_LIST_CONFIG_KEY = "feature_list"
LABEL_LIST_CONFIG_KEY = "label_list"
COVARIATE_LIST_CONFIG_KEY = "covariate_list"
RESAMPLE_FREQ_CONFIG_KEY = "resample_freq_min"

# Data Folder Paths - LOCAL
MINIMAL_PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
BINNED_ON_VAR_FREQ_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_var_binned_data")
SURVEYS_AND_COVARIATES_DATA_PATH = os.path.join(ROOT_DIR, "../data/surveys_and_covariates")
STUDENT_RAW_DATA_ANALYSIS_ROOT = os.path.join(ROOT_DIR, "../data/raw_student_data_information")

# Data Tuple Indices
DATA_TUPLE_LEN = 5
ACTUAL_DATA_IDX = 0
MISSING_FLAGS_IDX = 1
TIME_DELTA_IDX = 2
COVARIATE_DATA_IDX = 3
LABELS_IDX = -1  # Always last!

# Data Folder Paths - CLUSTER
# Overwrite Global Constants when cluster mode on.
config = read_yaml(FEATURE_CONFIG_FILE_PATH)
if config['cluster_mode']:
    cluster_data_root = config['data_paths']['cluster_data_path']
    MINIMAL_PROCESSED_DATA_PATH = pathlib.Path(
        os.path.join(cluster_data_root, "student_life_minimal_processed_data"))
    BINNED_ON_VAR_FREQ_DATA_PATH = pathlib.Path(
        os.path.join(cluster_data_root, "student_life_var_binned_data"))
