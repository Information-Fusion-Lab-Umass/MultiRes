import os
import pathlib

from src.utils.read_utils import read_yaml

# Defining Root Directory of the project.
ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
USER_HOME = pathlib.Path.home()

# File and Key Names
STUDENT_FOLDER_NAME_PREFIX = "student_"
BINNED_DATA_FILE_NAME = "var_binned_data.csv"
BINNED_DATA_MISSING_VALES_FILE_NAME = "missing_values_mask.csv"
BINNED_DATA_TIME_DELTA_FILE_NAME = "time_deltas_min.csv"
RESAMPLE_FREQ_CONFIG_KEY = "resample_freq_min"

# Config File Path
FEATURE_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "configurations/feature_processing.yaml")

# Frequency constants
DEFAULT_BASE_FREQ = 'min'

# Data Folder Paths - LOCAL
MINIMAL_PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_minimal_processed_data")
BINNED_ON_VAR_FREQ_DATA_PATH = os.path.join(ROOT_DIR, "../data/student_life_var_binned_data")

# Data Folder Paths - CLUSTER
# Overwrite Global Constants when cluster mode on.
config = read_yaml(FEATURE_CONFIG_FILE_PATH)
if config['cluster_mode']:
    cluster_data_root = config['data_paths']['cluster_data_path']
    MINIMAL_PROCESSED_DATA_PATH = pathlib.Path(
        os.path.join(cluster_data_root, "student_life_minimal_processed_data"))
    BINNED_ON_VAR_FREQ_DATA_PATH = pathlib.Path(
        os.path.join(cluster_data_root, "student_life_var_binned_data"))
