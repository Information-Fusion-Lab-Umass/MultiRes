from src import definitions
from src.utils import student_utils
from src.utils import read_utils
from src.data_manager import splitter
from src.utils import set_utils
from src.bin import validations as validations
from src.utils import data_conversion_utils as conversions
from src.data_manager import helper as data_manager_helper

VAR_BINNED_DATA_CONFIG = read_utils.read_yaml(definitions.DATA_MANAGER_CONFIG_FILE_PATH)[
    definitions.VAR_BINNED_DATA_MANAGER_ROOT]
ADJUST_LABELS_WRT_MEDIAN = VAR_BINNED_DATA_CONFIG['adjust_labels_wrt_median']
FLATTEN_SEQUENCE_TO_COLS = VAR_BINNED_DATA_CONFIG['flatten_sequence_to_cols']

DEFAULT_STUDENT_LIST = VAR_BINNED_DATA_CONFIG[definitions.STUDENT_LIST_CONFIG_KEY]
available_students = student_utils.get_available_students(definitions.BINNED_ON_VAR_FREQ_DATA_PATH)
DEFAULT_STUDENT_LIST = list(set(DEFAULT_STUDENT_LIST).intersection(set(available_students)))

FEATURE_LIST = VAR_BINNED_DATA_CONFIG[definitions.FEATURE_LIST_CONFIG_KEY]
LABEL_LIST = VAR_BINNED_DATA_CONFIG[definitions.LABEL_LIST_CONFIG_KEY]
COVARIATE_LIST = VAR_BINNED_DATA_CONFIG[definitions.COVARIATE_LIST_CONFIG_KEY]

if VAR_BINNED_DATA_CONFIG['process_covariates_as_regular_features']:
    FEATURE_LIST = FEATURE_LIST + COVARIATE_LIST
else:
    assert len(set_utils.lists_intersection(FEATURE_LIST, COVARIATE_LIST)) == 0, \
        "Feature List and Covariate List cannot overlap."

# These sizes are in percent of data.
TRAIN_SET_SIZE = VAR_BINNED_DATA_CONFIG['train_set_size']
VAL_SET_SIZE = VAR_BINNED_DATA_CONFIG['val_set_size']
TEST_SET_SIZE = VAR_BINNED_DATA_CONFIG['test_set_size']

DEFAULT_SPLITTING_STRATEGY = VAR_BINNED_DATA_CONFIG['default_splitting_strategy']
SPLITTING_STRATEGY_FUNCTION_MAP = {
    'day': data_manager_helper.get_data_for_single_day,
    'time_delta': data_manager_helper.get_data_for_single_label_based_on_time_delta
}


def get_data_based_on_labels_and_splitting_strategy(training_values, covariate_values, missing_values, time_delta,
                                                    y_labels, splitting_strategy, flatten_sequence_to_cols):
    """

    @param training_values: Training values of students.
    @param covariate_values: Values that need to be processed as covariates.
    @param missing_values: Missing values for one student.
    @param time_delta: Time deltas for one student.
    @param y_labels: Labels for training. Can have null values.
    @param splitting_strategy: Splitting strategy for the data. Current support for
            1) days - Each label will have one day's worth of data.
            2) time_delta -  Each label will contain data x hours beihind and y hours ahead (configurable by data_manager.yaml)
    @param flatten_sequence_to_cols: If true, the sequences are flattened into columns.
    @return: Trimmed data based on time delta.
    """
    validations.validate_data_integrity_for_len(training_values, missing_values, time_delta, y_labels)
    assert splitting_strategy in SPLITTING_STRATEGY_FUNCTION_MAP.keys(), \
        "Invalid splitting strategy must be one of: {}".format(SPLITTING_STRATEGY_FUNCTION_MAP.keys())

    data_list = []
    # todo(abhinavshaw): make it general for all the labels.
    y_labels = y_labels[y_labels['stress_level_mode'].notnull()]

    # todo(abihnavshaw): Process on whole data once fixed issue with last label.
    # len(y_label) -1 to ignore the last label.
    for label_idx in range(len(y_labels) - 1):
        data = SPLITTING_STRATEGY_FUNCTION_MAP[splitting_strategy](training_values,
                                                                   covariate_values,
                                                                   missing_values,
                                                                   time_delta,
                                                                   y_labels,
                                                                   y_labels.index[label_idx])

        if data:
            month_day_hour = str(y_labels.index[label_idx].month) + '_' + str(y_labels.index[label_idx].day) + '_' \
                             + str(y_labels.index[label_idx].hour)
            data = conversions.flatten_data(data) if flatten_sequence_to_cols else data
            data_list.append((month_day_hour, data))

    return data_list


def process_student_data(raw_data, student_id: int, splitting_strategy, normalize: bool, fill_na: bool,
                         flatten_sequence: bool):
    """
    Processes student data from a large DF of all students. This data is then transformed to the kind
    acceptable by DBM and VDB.
    """
    assert len(LABEL_LIST) == 1, "Feature List greater than one, check logic to generate labels."
    validations.validate_student_id_in_data(*raw_data)
    validations.validate_data_integrity_for_len(*raw_data)

    student_data, missing_data, time_delta = conversions.extract_actual_missing_and_time_delta_from_raw_data_for_student(
        raw_data, student_id=student_id)

    validations.validate_all_columns_present_in_data_frame(student_data, missing_data, time_delta, columns=FEATURE_LIST)
    validations.validate_all_columns_present_in_data_frame(student_data, columns=LABEL_LIST)

    training_values = student_data.loc[:, FEATURE_LIST]
    covariate_values = student_data.loc[:, COVARIATE_LIST]
    missing_values = missing_data.loc[:, FEATURE_LIST]
    time_deltas = time_delta.loc[:, FEATURE_LIST]
    y_labels = student_data.loc[:, LABEL_LIST]

    # Additional flags for data processing.
    if ADJUST_LABELS_WRT_MEDIAN:
        y_labels['stress_level_mode'] = y_labels['stress_level_mode'].map(conversions.adjust_classes_wrt_median,
                                                                          na_action='ignore')
        if 'previous_stress_label' in FEATURE_LIST:
            training_values['previous_stress_label'] = training_values['previous_stress_label'].map(
                conversions.adjust_classes_wrt_median,
                na_action='ignore')

    if normalize:
        training_values = conversions.normalize(training_values)

    # Filling missing Values
    if fill_na:
        training_values.fillna(value=-1, inplace=True)

    data_list = get_data_based_on_labels_and_splitting_strategy(training_values,
                                                                covariate_values,
                                                                missing_values,
                                                                time_deltas,
                                                                y_labels,
                                                                splitting_strategy,
                                                                flatten_sequence)

    # Splitting data into Train, Val  and Test Split.
    train_set, end_idx = splitter.split_data_by_percentage(data_list, start_index=0, percent=25)
    val_set, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=15)
    test_set, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=1)

    train_set_2, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=25)
    val_set_2, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=15)
    test_set_2, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=1)

    train_set_3, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=10)
    val_set_3, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=1)
    test_set_3, end_idx = splitter.split_data_by_percentage(data_list, start_index=end_idx, percent=-1)

    train_set = train_set + train_set_2 + train_set_3
    val_set = val_set + val_set_2 + val_set_3
    test_set = test_set + test_set_2 + test_set_3

    train_set = [month_day for month_day, data in train_set]
    val_set = [month_day for month_day, data in val_set]
    test_set = [month_day for month_day, data in test_set]

    return data_list, train_set, val_set, test_set


def get_data_for_training_in_dict_format(*student_ids,
                                         splitting_strategy=DEFAULT_SPLITTING_STRATEGY,
                                         normalize=False,
                                         fill_na=True,
                                         flatten_sequence=False):
    """

    @attention: If no student_ids given to function the default students are returned.
    @return: The processed data for all the students in the config.
    """
    if not student_ids:
        student_ids = DEFAULT_STUDENT_LIST
    else:
        student_ids = list(student_ids)

    data = dict()
    data["train_ids"] = []
    data["val_ids"] = []
    data["test_ids"] = []

    data_dict = {}
    raw_data = student_utils.get_var_binned_data_for_students(*student_ids)

    for it, student_id in enumerate(student_ids):
        data_list, train_ids, val_ids, test_ids = process_student_data(raw_data,
                                                                       student_id,
                                                                       splitting_strategy=splitting_strategy,
                                                                       normalize=normalize,
                                                                       fill_na=fill_na,
                                                                       flatten_sequence=flatten_sequence)

        # Prefixing the IDs with student_id.
        for month_day, daily_data in data_list:
            data_key = str(student_id) + "_" + month_day
            data_dict[data_key] = daily_data

        train_ids, val_ids, test_ids = student_utils.prefix_list_of_strings_or_ids_with_student_id(train_ids,
                                                                                                   val_ids,
                                                                                                   test_ids,
                                                                                                   student_id=student_id)

        data['data'] = data_dict
        data['train_ids'] += train_ids
        data['val_ids'] += val_ids
        data['test_ids'] += test_ids

    return data
