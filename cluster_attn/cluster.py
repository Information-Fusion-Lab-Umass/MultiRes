import csv
import numpy as np

import cPickle as pickle

import seaborn as sns
import matplotlib.pyplot as plt


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


def oppo_dot(data):
    """
    :param data: a list of np (T, 37) matrix. Has list len = B. If the data is training data, B = 2500
                 For each T-dim vector in the matrix, it is a T dim missing flag vector
    :return: a (37, 37) dim matrix. We use that matrix to do clustering
    """
    B = len(data)
    num_feats = len(data[0][0])
    scores = np.zeros((num_feats, num_feats))
    for b in range(B):
        local_score = np.zeros((num_feats, num_feats))
        m = data[b]  # (T, 37)
        T = m.shape[0]
        for f in range(num_feats):
            main_vec = 1 - m[:, f]  # (37)
            for sf in range(num_feats):
                local_score[f, sf] = main_vec.dot(m[:, sf]) / T
        scores += local_score
    np.savetxt("physio_feats_score.csv", scores, delimiter=",")
    return scores / B


if __name__ == '__main__':
#    ax = sns.heatmap(read_corr_csv('../../data/intercorr_physionet.csv'))
#    plt.show()

     dummy_cluster = [list(range(0, 10)), list(range(10, 20)), list(range(20, 37))]
     pickle.dump(dummy_cluster, open('./data/dummy_cluster.pkl', 'wb'))

    # data = pickle.load(open('../../data/imputed_physionet.pkl', 'rb'))
    # train = data['train']['data']  # [[time_seq0, 37, 4]]
    # flag_data = []  # [[time_seq0, 37]]
    # for b in range(len(train)):
    #     batch = np.array(train[b])  # [time_seq0, 37, 4]
    #     flag_data.append(batch[:, :, 2])
    # print(oppo_dot(flag_data))

#    ax = sns.heatmap(read_corr_csv('physio_feats_score.csv'))
#    plt.show()

