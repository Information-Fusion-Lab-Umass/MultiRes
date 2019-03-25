import cPickle as pickle
import csv

import matplotlib
import numpy as np

matplotlib.use('agg')


def read_corr_csv(file_name):
    """
    :param file_name: csv matrix file path
    :return: 21 * 21 matrix. The first 3 features are aborted so we do not store their correlation value
    """
    with open(file_name, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')

        next(csvreader)

        data_ = []
        for row in csvreader:
            data_.append([float(row[i]) for i in range(1, len(row))])

    return np.array(data_)


if __name__ == '__main__':
    # ax = sns.heatmap(read_corr_csv('../../data/intercorr_physionet.csv'))
    # plt.show()
    dummy_cluster = [list(range(0, 10)), list(range(10, 20)), list(range(20, 37))]
    pickle.dump(dummy_cluster, open('./pre/dummy_cluster.pkl', 'wb'))
