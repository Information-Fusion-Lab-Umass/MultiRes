import os
import pandas as pd

from src import definitions
from src.utils.read_utils import read_yaml


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
            try:
                result.append(int(folder_name[len(prefix):]))
            except ValueError:
                print("Student ID couldn't be converted to Integer!")

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


def get_binned_data_for_students(*student_id: int):
    """

    @param student_id: The student id(s) for which binned data is required.
           Can pass multiple student id.
    @return: List of DataFrames containing actual data, missing flags and time delta.
    """
    students = set(get_available_students(definitions.BINNED_ON_VAR_FREQ_DATA_PATH))
    student_ids = set(list(student_id))
    students = student_ids & students

    students_data_binned_data = pd.DataFrame()
    students_data_missing_values_mask = pd.DataFrame()
    students_data_time_deltas = pd.DataFrame()

    assert len(students) > 0, "None of the students entered is valid. Please enter valid students"

    for student in students:
        binned_data_fp = os.path.join(definitions.BINNED_ON_VAR_FREQ_DATA_PATH,
                                      definitions.STUDENT_FOLDER_NAME_PREFIX +
                                      str(student), definitions.BINNED_DATA_FILE_NAME)
        missing_values_fp = os.path.join(definitions.BINNED_ON_VAR_FREQ_DATA_PATH,
                                         definitions.STUDENT_FOLDER_NAME_PREFIX +
                                         str(student), definitions.BINNED_DATA_MISSING_VALES_FILE_NAME)
        time_delta_fp = os.path.join(definitions.BINNED_ON_VAR_FREQ_DATA_PATH,
                                     definitions.STUDENT_FOLDER_NAME_PREFIX +
                                     str(student), definitions.BINNED_DATA_TIME_DELTA_FILE_NAME)

        binned_data = pd.read_csv(binned_data_fp, index_col=[0])
        missing_values = pd.read_csv(missing_values_fp, index_col=[0])
        time_deltas = pd.read_csv(time_delta_fp, index_col=[0])
        binned_data.index = pd.to_datetime(binned_data.index)
        missing_values.index = pd.to_datetime(missing_values.index)
        time_deltas.index = pd.to_datetime(time_deltas.index)
        students_data_binned_data = students_data_binned_data.append(binned_data)
        students_data_missing_values_mask = students_data_missing_values_mask.append(missing_values)
        students_data_time_deltas = students_data_time_deltas.append(time_deltas)

    return students_data_binned_data, students_data_missing_values_mask, students_data_time_deltas
