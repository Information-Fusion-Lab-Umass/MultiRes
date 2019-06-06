import os
import pandas as pd

from src import definitions
from src.utils.set_utils import lists_intersection
from src.utils.read_utils import read_yaml

STUDENT_DATA_TYPES = [definitions.BINNED_DATA_FILE_NAME,
                      definitions.BINNED_DATA_MISSING_VALES_FILE_NAME,
                      definitions.BINNED_DATA_TIME_DELTA_FILE_NAME]


def get_student_list_after_ignoring(students, students_to_be_ignored):
    """
    @param students: A list of students that are to be considered. Has to be integer IDs
    @param students_to_be_ignored: List of students to be ignored.
    @return: Final processed list of students.
    """
    for e in students_to_be_ignored:
        if e in students:
            students.remove(e)

    return students


def get_students_from_folder_names(prefix, folder_names):
    """

    @param prefix: Prefix to be removed
    @param folder_names: Folder name of the students whos prefix are to be removed.
    @return: Student Names as int.
    """
    result = []
    for folder_name in folder_names:
        if folder_name.startswith(prefix):
            val = None
            try:
                val = folder_name[len(prefix):]
                result.append(int(val))
            except ValueError as e:
                print("Student ID couldn't be converted to Integer: {}".format(val))

    return result


def get_students_to_be_ignored():
    """

    @return: Reads the ignore list from feature config.
    """
    config = read_yaml(definitions.FEATURE_CONFIG_FILE_PATH)
    return config['students']['student_ignore_list']


def get_available_students(path):
    """

    @return: Returns available students after ignoring the ones in the ignore list.
    """
    students = get_students_from_folder_names(definitions.STUDENT_FOLDER_NAME_PREFIX, os.listdir(path))
    return get_student_list_after_ignoring(students, get_students_to_be_ignored())


def get_helper_data_dict(keys):
    """

    @return: Return a helper dictionary with keys as given in the input and values as data frames.
    """
    helper_dict = dict()

    for key in keys:
        helper_dict[key] = pd.DataFrame()

    return helper_dict


def get_binned_data_for_student(student_id: int, data_type: str):
    """

    @param student_id: Student id for whom the data is require.
    @param data_type: Type of data for eg : missing values, actual data or Time Deltas. Must be in STUDENT_DATATYPE
    @return: Takes in student ids as `args and returns data for that student for the requested data type.
    """
    assert data_type in STUDENT_DATA_TYPES, "Invalid type entered: {}".format(data_type)

    assert student_id in get_available_students(definitions.BINNED_ON_VAR_FREQ_DATA_PATH), \
        "Data for student_id: {} is not available. Check student or ignore list.".format(student_id)

    data_file_path = os.path.join(definitions.BINNED_ON_VAR_FREQ_DATA_PATH,
                                  definitions.STUDENT_FOLDER_NAME_PREFIX +
                                  str(student_id), data_type + ".csv")
    data = pd.read_csv(data_file_path, index_col=[0])
    data.index = pd.to_datetime(data.index)

    return data


def get_var_binned_data_for_students(*student_id: int):
    """
    @attention: Primary API to query Var Binned Data for students.
    @param student_id: The student id(s) for which binned data is required.
           Can pass multiple student id.
    @return: Tuple of DataFrames containing actual data, missing flags and time delta.
    """
    students = get_available_students(definitions.BINNED_ON_VAR_FREQ_DATA_PATH)
    students = lists_intersection(students, list(student_id))
    helper_data_dict = get_helper_data_dict(STUDENT_DATA_TYPES)

    assert len(students) > 0, "None of the students entered is valid. Please enter valid students"

    for student in students:
        for data_type in STUDENT_DATA_TYPES:
            student_data = get_binned_data_for_student(student, data_type)
            helper_data_dict[data_type] = helper_data_dict[data_type].append(student_data)

    return helper_data_dict[definitions.BINNED_DATA_FILE_NAME], helper_data_dict[
        definitions.BINNED_DATA_MISSING_VALES_FILE_NAME], helper_data_dict[
        definitions.BINNED_DATA_TIME_DELTA_FILE_NAME]


def prefix_list_of_strings_or_ids_with_student_id(*string_or_id_list, student_id):
    prefixed_list = []
    for strings_or_ids in list(string_or_id_list):
        prefixed_list.append([str(student_id) + "_" + str(str_or_id) for str_or_id in strings_or_ids])

    return tuple(prefixed_list)
