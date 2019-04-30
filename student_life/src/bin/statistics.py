import pandas as pd
import numpy as np
import torch

from collections import Counter
from tabulate import tabulate
from src import definitions
from src.bin import validations
from src.data_manager import student_life_var_binned_data_manager as data_manager

LABEL_COUNT_HEADERS = ['Train', 'Val', 'Test']


def get_statistics_on_data_dict(data: dict, feature_list: list):
    """
    @attention  Statistics returned are ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].
    @param data: Data in classic dictionary format.
    @param feature_list: Feature list for the data.
    @return: Statistics on whole data and raw appended data.
    """
    validations.validate_data_dict_keys(data)
    validations.validate_all_data_present_in_data_dict(data)
    df_for_statistics = pd.DataFrame()

    for key in data['data']:
        unit_sequence = data['data'][key][0]
        df_for_statistics = df_for_statistics.append(pd.DataFrame(unit_sequence),
                                                     ignore_index=True)

    if not data_manager.FLATTEN_SEQUENCE_TO_COLS:
        df_for_statistics.columns = feature_list
    df_for_statistics.replace(to_replace=-1, value=np.nan, inplace=True)
    return df_for_statistics.describe(percentiles=[0.25, 0.5, 0.75]), df_for_statistics


def get_train_test_val_label_counts_from_raw_data(data: dict):
    """

    @param data: Data in dictionary format.
    @return: Counts.
    """
    train_counter = get_label_count_in_split(data, 'train')
    val_counter = get_label_count_in_split(data, 'val')
    test_counter = get_label_count_in_split(data, 'test')
    overall_counts = convert_label_counters_to_list(train_counter, val_counter, test_counter)

    return tabulate(overall_counts, headers=LABEL_COUNT_HEADERS)


def get_label_count_in_split(data: dict, split: str):
    """

    @param data: Data in dictionary format.
    @param split: Split for which label counts are required.
                  Accepts 'test', 'train' and 'val'.
    @return: Label count for the given split.
    """
    assert split in ['train', 'test', 'val']

    labels = []
    for split_id in data[split + "_ids"]:
        label = data['data'][split_id][definitions.LABELS_IDX]
        labels.append(label)

    counters = Counter(labels)
    return counters


def convert_label_counters_to_list(*counters):
    counters = list(counters)
    overall_counts = []

    def append_count_for_label(label):
        counts_per_element = [label]
        for count in counters:
            counts_per_element.append(count[label])
        overall_counts.append(counts_per_element)

    if data_manager.ADJUST_LABELS_WRT_MEDIAN:
        for label in range(3):
            append_count_for_label(label)
    else:
        for label in range(5):
            append_count_for_label(label)

    return overall_counts


def tensor_preds_to_scalar_counter(tensor_preds):
    preds_list = []
    if isinstance(tensor_preds, torch.Tensor):
        for pred in tensor_preds:
            preds_list.append(pred.item())
    else:
        pred_list = tensor_preds

    return Counter(preds_list)


def get_train_test_val_label_counts_from_predictions(*predictions):
    predictions = list(predictions)
    counters = []
    for preds in predictions:
        counters.append(tensor_preds_to_scalar_counter(preds))
    overall_counts = convert_label_counters_to_list(*counters)

    return tabulate(overall_counts, LABEL_COUNT_HEADERS)


def get_class_weights_in_inverse_proportion(data: dict):

    train_label_counts = get_label_count_in_split(data, 'train')
    # todo(abhinavshaw): Make it general for 3 or 5 classes.
    train_label_counts = [train_label_counts[label] for label in range(3)]
    class_weights = [x / max(train_label_counts) for x in train_label_counts]
    class_weights = [1 / x for x in class_weights]
    class_weights = [x / max(class_weights) for x in class_weights]

    return class_weights
