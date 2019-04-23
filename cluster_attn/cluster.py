import csv
import numpy as np

# import cPickle as pickle
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx


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


def mst_corr_cluster(corr_path, cluster_num=None):
    corr = np.abs(read_corr_csv(corr_path))
    feats = list(range(0, 37))

    print(corr[23][36])
    print(corr[24][36])

    rank = np.argsort(corr, axis=0)
    clusters = []

    G = nx.Graph()

    G.add_nodes_from(feats)

    # for a in feats:
    #     for b in range(a, 37):
    #         G.add_edge(a, b, weight=corr[a, b])

    for a in feats:
        for b in range(a, 37):
            if corr[a, b] > 0.3:
                G.add_edge(a, b, weight=corr[a, b])
    T = nx.maximum_spanning_tree(G)

    nx.draw(T, with_labels=True, font_weight='bold')

    sorted_edges = sorted(T.edges.data('weight'), key=lambda x: x[2])
    # if cluster_num is None:
    #     for e in sorted_edges:
    #         # print(e)
    #         if e[2] < 0.1:
    #             T.remove_edge(e[0], e[1])
    # else:
    #     for eid in range(cluster_num-1):
    #         T.remove_edge(sorted_edges[eid][0], sorted_edges[eid][1])
    nx.draw(T, with_labels=True, font_weight='bold')

    comp = nx.connected_components(T)
    for i in comp:
        clusters.append(list(i))
    print(clusters)

    return clusters


if __name__ == '__main__':
    mst_cluster = mst_corr_cluster('./data/intercorr_physionet.csv')
    print(len(mst_cluster))
    count = 0
    l_cluster = []
    for c in mst_cluster:
        if len(c) > 1:
            count += 1
            l_cluster.append(c)
    print(count)
    print(l_cluster)
    pickle.dump(mst_cluster, open('./data/mst_cluster.pkl', 'wb'), protocol=2)
    # cluster = pickle.load(open('./data/mst_cluster.pkl', 'rb'))
    # print(cluster)
    # print(len(cluster))

   # ax = sns.heatmap(read_corr_csv('../../data/intercorr_physionet.csv'))
   # plt.show()
#      dummy_cluster = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20)),
#                       list(range(20, 25)), list(range(25, 30)), list(range(30, 37))]
#
#      # dummy_cluster = [list(range(i, i+1)) for i in range(0, 37)]
#      pickle.dump(dummy_cluster, open('./data/dummy_cluster.pkl', 'wb'))

    # data = pickle.load(open('../../data/imputed_physionet.pkl', 'rb'))
    # train = data['train']['data']  # [[time_seq0, 37, 4]]
    # flag_data = []  # [[time_seq0, 37]]
    # for b in range(len(train)):
    #     batch = np.array(train[b])  # [time_seq0, 37, 4]
    #     flag_data.append(batch[:, :, 2])
    # print(oppo_dot(flag_data))

#    ax = sns.heatmap(read_corr_csv('physio_feats_score.csv'))
#    plt.show()

