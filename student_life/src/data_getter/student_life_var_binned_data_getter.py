import src.definitions as definitions

from src.utils import student_utils
from src.utils import read_utils
from src.utils import validation_utils as validations


VAR_BINNED_DATA_CONFIG = read_utils.read_yaml(definitions.DATA_GETTER_CONFIG_FILE_PATH)[
    definitions.VAR_BINNED_DATA_GETTER_ROOT]

VAR_BINNED_DATA_STUDENTS = VAR_BINNED_DATA_CONFIG[definitions.DATA_GETTER_STUDENT_LIST_CONFIG_KEY]
available_students = student_utils.get_available_students(definitions.BINNED_ON_VAR_FREQ_DATA_PATH)
VAR_BINNED_DATA_STUDENTS = list(set(VAR_BINNED_DATA_STUDENTS).intersection(set(available_students)))

FEATURE_LIST = VAR_BINNED_DATA_CONFIG[definitions.DATA_GETTER_FEATURE_LIST_CONFIG_KEY]
LABEL_LIST = VAR_BINNED_DATA_CONFIG[definitions.DATA_GETTER_LABEL_LIST_CONFIG_KEY]


def get_var_binned_data_for_students():
    """
    @attention: The students returned in the data are controlled by the conflict DATA_GETTER_CONFIG.
    @return:
    """
    return student_utils.get_var_binned_data_for_students(*VAR_BINNED_DATA_STUDENTS)


def convert_df_to_tuple(*data_frames):
    data_frames_as_list = []

    for df in data_frames:
        data_frames_as_list.append(df.values.tolist())

    return tuple(data_frames_as_list)


def split_data_to_train_test_val(training_values, missing_values, time_deltas, y_labels):
    """
    @todo(abhinavshaw): Shorten this code by abstracting splitting operation to a function and call
                        that function for different splits.
    @attention: Long code ahead!
    @return: Takes whole data tuple for a single student and converts the into Train, Test and Val.
             Will return three data tuples.
             Ratio -  Train : 60, Test: 20 and Val: 20.
    """

    assert len(training_values) == len(missing_values) and len(training_values) == len(time_deltas) and len(
        training_values) == len(y_labels), "Lengths of the DataFrame do not match."

    data_len = len(training_values)

    percent_train_split = 60
    percent_val_split = 20
    # Rest is test split.

    train_index_end = int(data_len * (percent_train_split / 100))
    val_index_end = int(data_len * (percent_val_split / 100))

    train_split_values = training_values.iloc[:train_index_end, :]
    train_split_missing_values = missing_values.iloc[:train_index_end, :]
    train_split_time_deltas = time_deltas.iloc[:train_index_end, :]
    train_y_labels = y_labels.iloc[:train_index_end, :]

    val_split_values = training_values.iloc[train_index_end:train_index_end + val_index_end, :]
    val_split_missing_values = missing_values.iloc[train_index_end:train_index_end + val_index_end, :]
    val_split_time_deltas = time_deltas.iloc[train_index_end:train_index_end + val_index_end, :]
    val_y_labels = y_labels.iloc[train_index_end:train_index_end + val_index_end, :]

    test_split_values = training_values.iloc[train_index_end + val_index_end:, :]
    test_split_missing_values = missing_values.iloc[train_index_end + val_index_end:, :]
    test_split_time_deltas = time_deltas.iloc[train_index_end + val_index_end:, :]
    test_y_labels = y_labels.iloc[train_index_end + val_index_end:, :]

    train_tuple = convert_df_to_tuple(train_split_values,
                                      train_split_missing_values,
                                      train_split_time_deltas,
                                      train_y_labels)
    val_tuple = convert_df_to_tuple(val_split_values,
                                    val_split_missing_values,
                                    val_split_time_deltas,
                                    val_y_labels)
    test_tuple = convert_df_to_tuple(test_split_values,
                                     test_split_missing_values,
                                     test_split_time_deltas,
                                     test_y_labels)

    return train_tuple, val_tuple, test_tuple


def process_student_data(data_tuple, student_id: int):
    """
    Processes student data from a large DF of all students. This data is then transformed to the kind
    acceptable by DBM and VDB.
    """
    validations.validate_student_id_in_data(*data_tuple)

    (student_data, missing_data, time_delta) = data_tuple
    student_data = student_data[student_data['student_id'] == student_id]
    missing_data = missing_data[missing_data['student_id'] == student_id]
    time_delta = time_delta[time_delta['student_id'] == student_id]

    training_values = student_data.loc[:, FEATURE_LIST]
    y_labels = student_data.loc[:, LABEL_LIST]
    missing_values = missing_data.loc[:, FEATURE_LIST]
    time_deltas = time_delta.loc[:, FEATURE_LIST]

    # Filling missing Values
    training_values.fillna(value=-1, inplace=True)
    y_labels.fillna(method='ffill', inplace=True)
    result = split_data_to_train_test_val(training_values, missing_values, time_deltas, y_labels)

    return result


def get_data_for_pkl_file():
    """

    @return: The processed data for all the students in the config.
    """
    data = {}
    data_dict = {}
    train_ids, val_ids, test_ids = [], [], []

    raw_data = student_utils.get_var_binned_data_for_students(*VAR_BINNED_DATA_STUDENTS)

    train_split_key = str(1)
    val_split_key = str(2)
    test_split_key = str(3)

    for it, student_id in enumerate(VAR_BINNED_DATA_STUDENTS):
        train_tuple, val_tuple, test_tuple = process_student_data(raw_data, student_id)
        student_key = str(student_id)
        train_id = student_key + "_" + train_split_key
        val_id = student_key + "_" + val_split_key
        test_id = student_key + "_" + test_split_key

        data_dict[train_id] = train_tuple
        data_dict[val_id] = val_tuple
        data_dict[test_id] = test_tuple
        train_ids.append(train_id)
        val_ids.append(val_id)
        test_ids.append(test_id)

        data['data'] = data_dict

    data["train_ids"] = train_ids
    data["val_ids"] = val_ids
    data["test_ids"] = test_ids

    return data
