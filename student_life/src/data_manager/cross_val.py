import numpy as np

from random import shuffle
from sklearn.model_selection import train_test_split
from src.utils import data_conversion_utils as conversions_utils


def random_stratified_splits(data: dict, stratify_by="labels"):
    """

    @param data: Data in classic dict format.
    @param stratify_by: By what the splits need to be stratified.
           Accepts - `labels` and `students`.
    @return: Return splits which are randomize and stratified by labels.
    """
    keys, labels = conversions_utils.extract_keys_and_labels_from_dict(data)
    keys, labels = np.array(keys), np.array(labels)

    if stratify_by == "students":
        student_ids = conversions_utils.extract_student_ids_from_keys(keys)
        stratification_array = np.array(student_ids)
    else:
        stratification_array = labels

    (X_train, X_test,
     y_train, y_test,
     stratification_array, new_stratification_array) = train_test_split(keys,
                                                                        labels,
                                                                        stratification_array,
                                                                        test_size=0.40,
                                                                        shuffle=True,
                                                                        stratify=stratification_array)
    X_val, X_test = train_test_split(X_test,
                                     test_size=0.40,
                                     shuffle=True,
                                     stratify=new_stratification_array)

    return X_train.tolist(), X_val.tolist(), X_test.tolist()


def leave_one_subject_out_split(data: dict):
    """

    @param data: data for which the splits are needed to be generated.
    @return: Return generator object with different data split,
             with test and val belonging to the left out student.
    """

    keys, labels = conversions_utils.extract_keys_and_labels_from_dict(data)
    student_ids = conversions_utils.extract_distinct_student_idsfrom_keys(keys)

    for idx, left_out_student in enumerate(student_ids):
        students_for_training = student_ids[:idx] + student_ids[idx + 1:]
        train_ids = conversions_utils.get_filtered_keys_for_these_students(*students_for_training, keys=keys)
        val_test_ids = conversions_utils.get_filtered_keys_for_these_students(left_out_student, keys=keys)

        shuffle(train_ids)
        shuffle(val_test_ids)

        data['train_ids'] = train_ids
        val_len = len(val_test_ids)
        data['val_ids'] = val_test_ids
        data['test_ids'] = []

        yield data, left_out_student
