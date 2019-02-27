from src import definitions
from src.utils import student_utils
from src.utils import read_utils
from src.bin import validations as validations

VAR_BINNED_DATA_CONFIG = read_utils.read_yaml(definitions.DATA_GETTER_CONFIG_FILE_PATH)[
    definitions.VAR_BINNED_DATA_GETTER_ROOT]

DEFAULT_STUDENT_LIST = VAR_BINNED_DATA_CONFIG[definitions.DATA_GETTER_STUDENT_LIST_CONFIG_KEY]
available_students = student_utils.get_available_students(definitions.BINNED_ON_VAR_FREQ_DATA_PATH)
DEFAULT_STUDENT_LIST = list(set(DEFAULT_STUDENT_LIST).intersection(set(available_students)))

FEATURE_LIST = VAR_BINNED_DATA_CONFIG[definitions.DATA_GETTER_FEATURE_LIST_CONFIG_KEY]
LABEL_LIST = VAR_BINNED_DATA_CONFIG[definitions.DATA_GETTER_LABEL_LIST_CONFIG_KEY]

# These sizes are in percent of data.
TRAIN_SET_SIZE = VAR_BINNED_DATA_CONFIG['train_set_size']
VAL_SET_SIZE = VAR_BINNED_DATA_CONFIG['val_set_size']
TEST_SET_SIZE = VAR_BINNED_DATA_CONFIG['test_set_size']


def convert_df_to_tuple(*data_frames):
    data_frames_as_list = []

    for df in data_frames:
        data_frames_as_list.append(df.values.tolist())

    return tuple(data_frames_as_list)


def get_data_for_single_day(training_values, missing_values, time_delta, y_labels, label_idx):
    """

    @param training_values:
    @param missing_values:
    @param time_delta:
    @param y_label:
    @return: Return split for a single day. i.e. One label corresponds to several data points,
             takes in raw data frame and the label for which the split has to be calculated.
    """
    day_string_format = '%Y-%m-%d'
    day_string = label_idx.to_pydatetime().strftime(day_string_format)

    return (training_values.loc[day_string, :].values.tolist(),
            missing_values.loc[day_string, :].values.tolist(),
            time_delta.loc[day_string, :].values.tolist(),
            int(y_labels.loc[day_string, :].values[0]))


def split_data_to_list_of_days(training_values, missing_values, time_deltas, y_labels):
    """
    @attention: Long code ahead!
    @return: Takes whole data tuple for a single student and converts the into Train, Test and Val.
             Will return three data tuples.
             Ratio -  Train : 60, Test: 20 and Val: 20.
    """

    validations.validate_data_integrity_for_len(training_values, missing_values, time_deltas, y_labels)

    data_list = []
    # todo(abhinavshaw): make it general for all the labels.
    y_labels = y_labels[y_labels['stress_level_mode'].notnull()]

    # len(y_label) -1 to ignore the last label.
    for label_idx in range(len(y_labels)-1):
        month_day = str(y_labels.index[label_idx].month) + '_' + str(y_labels.index[label_idx].day)
        data_list.append((month_day, get_data_for_single_day(training_values,
                                                             missing_values,
                                                             time_deltas,
                                                             y_labels,
                                                             y_labels.index[label_idx])))

    return data_list


def process_student_data(raw_data, student_id: int):
    """
    Processes student data from a large DF of all students. This data is then transformed to the kind
    acceptable by DBM and VDB.
    """
    assert len(LABEL_LIST) == 1, "Feature List greater than one, check logic to generate labels."

    validations.validate_student_id_in_data(*raw_data)
    validations.validate_data_integrity_for_len(*raw_data)

    (student_data, missing_data, time_delta) = raw_data
    student_data = student_data[student_data['student_id'] == student_id]
    missing_data = missing_data[missing_data['student_id'] == student_id]
    time_delta = time_delta[time_delta['student_id'] == student_id]

    training_values = student_data.loc[:, FEATURE_LIST]
    missing_values = missing_data.loc[:, FEATURE_LIST]
    time_deltas = time_delta.loc[:, FEATURE_LIST]
    y_labels = student_data.loc[:, LABEL_LIST]

    # Filling missing Values
    training_values.fillna(value=-1, inplace=True)
    data_list = split_data_to_list_of_days(training_values,
                                           missing_values,
                                           time_deltas,
                                           y_labels)

    # Splitting data into Train, Val  and Test Split.
    train_set, end_idx = split_data_by_percentage(data_list, start_index=0, percent=TRAIN_SET_SIZE)
    val_set, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=VAL_SET_SIZE)
    test_set, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=-1)

    train_set = [month_day for month_day, data in train_set]
    val_set = [month_day for month_day, data in val_set]
    test_set = [month_day for month_day, data in test_set]

    return data_list, train_set, val_set, test_set


def prefix_ids_with_student_id(index_list, student_id):
    return [str(student_id) + "_" + str(index) for index in index_list]


def get_test_train_split_split(all_ids):
    data_len = len(all_ids)
    percent_train_split = 60
    percent_val_split = 20
    train_index_end = int(data_len * (percent_train_split / 100))
    val_index_end = int(data_len * (percent_val_split / 100))

    train_ids = all_ids[:train_index_end]
    val_ids = all_ids[train_index_end: train_index_end + val_index_end]
    test_ids = all_ids[train_index_end + val_index_end:]

    return train_ids, val_ids, test_ids


def split_data_by_percentage(data_list, start_index: int = 0, percent: float = -1):
    """
    
    @param data_list: The data for which slice is required. 
    @param start_index: all indices before this are not considered for slicing. 
    @param percent: Percentage of data that contributes to the slice. If percent = -1,
           then everything from start_index to len(data) is returned.
    @return: 
    """
    data_len = len(data_list)
    slice_length = round(data_len*percent / 100)

    assert 0 < percent <= 100 or percent == -1, "Percent value must be between 1 and 100 but got {}".format(percent)
    assert 0 <= start_index < data_len
    assert start_index + slice_length < data_len, "Over flow of data list. " \
                                                  "Enter smaller percent value or reduce the start_index."

    if percent == -1:
        slice_data = data_list[start_index:]
        end_index = data_len - 1
    else:
        slice_data = data_list[start_index: start_index+slice_length]
        end_index = start_index + slice_length

    return slice_data, end_index


def get_data_for_pkl_file(*student_ids):
    """
    @todo(abhinavshaw): Make validation strategy general for all students and not whole data.
    @attention: The student list is controlled by the configuration in the config.
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
        data_list, train_ids, val_ids, test_ids = process_student_data(raw_data, student_id)

        # Prefixing the IDs with student_id.
        for month_day, daily_data in data_list:
            data_key = str(student_id) + "_" + month_day
            data_dict[data_key] = daily_data

        train_ids = prefix_ids_with_student_id(train_ids, student_id)
        val_ids = prefix_ids_with_student_id(val_ids, student_id)
        test_ids = prefix_ids_with_student_id(test_ids, student_id)

        data['data'] = data_dict
        data['train_ids'] += train_ids
        data['val_ids'] += val_ids
        data['test_ids'] += test_ids

    return data
