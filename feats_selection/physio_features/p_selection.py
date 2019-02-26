import csv

import numpy as np


def read_corr_csv(file_name):
    """
    :param file_name: csv matrix file path
    :return: for osaka it returns a 24 * 24 matrix. The first 3 physio_features are aborted so their values are all zeros
    """
    with open(file_name, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')

        next(csvreader)

        data_ = []
        for row in csvreader:
            data_.append([float(row[i]) for i in range(1, len(row))])

    return np.array(data_)


def p_selection(corr_path):
    corr = read_corr_csv(corr_path)
    columns = np.full((corr.shape[0],), True, dtype=bool)

    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    print(columns)


if __name__ == '__main__':
    p_selection('../../data/osaka_features_corr.csv')

