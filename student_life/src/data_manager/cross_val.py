import numpy as np

from sklearn.model_selection import train_test_split
from src.utils import data_conversion_utils as conversions_utils


def random_stratified_splits(data: dict):
    """

    @param data: Data in classic dict format.
    @return: Return splits which are randomize and stratified by labels.
    """
    keys, labels = conversions_utils.extract_keys_and_labels_from_dict(data)
    keys, labels = np.array(keys), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(keys,labels,
                                                        test_size=0.40,
                                                        shuffle=True,
                                                        stratify=labels)

    X_val, X_test = train_test_split(X_test,
                                           test_size=0.40,
                                           shuffle=True,
                                           stratify=y_test)

    return X_train.tolist(), X_val.tolist(), X_test.tolist()
