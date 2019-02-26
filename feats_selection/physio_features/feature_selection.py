import csv

import numpy as np
import cPickle as pickle


def read_corr_csv(file_name):
    """
    :param file_name: csv matrix file path
    :return: 21 * 21 matrix. The first 3 physio_features are aborted so we do not store their correlation value
    """
    with open(file_name, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')

        next(csvreader)
        next(csvreader)
        next(csvreader)
        next(csvreader)

        data_ = []
        for row in csvreader:
            data_.append([float(row[i]) for i in range(4, len(row))])

    return np.array(data_)


def incremental_select(corr_path, file_path):
    corr_mtrx = (read_corr_csv(corr_path))

    abs_corr_mtrx = np.abs(corr_mtrx)
    idx_corr = np.argsort(np.sum(abs_corr_mtrx, axis=0))

    data = pickle.load(open(file_path, 'rb'))

    # train, test, validation: A list of [[[21 dim], [21 dim], ..., [21 dim]], label]
    # train = [data['data'][i] for i in data['train_ids']]
    # test = [data['data'][i] for i in data['test_ids']]
    # validation = [data['data'][i] for i in data['val_ids']]

    select_l = {}

    for idx in data['data'].keys():
        content = data['data'][idx]
        temp = [[], [], content[1]]
        for t in content[0]:
            zero_feats = [0 for i in range(21)]
            labels = [0 for i in range(21)]
            zero_feats[idx_corr[0]] = t[idx_corr[0]]
            labels[idx_corr[0]] = 1
            temp[0].append(zero_feats[:])
            temp[1].append(labels[:])
        select_l[idx] = temp

    pickle.dump({'train_ids': data['train_ids'],
                 'test_ids': data['test_ids'],
                 'val_ids': data['val_ids'],
                 'data': select_l.copy()}, open('../../data/physio_features/osaka_1_fl.pkl', 'wb'))

    for i in range(1, len(idx_corr)):
        for idx in data['data'].keys():
            content = data['data'][idx]
            temp = select_l[idx]
            for j in range(len(content[0])):
                temp[0][j][idx_corr[i]] = (content[0][j][idx_corr[i]])
                temp[1][j][idx_corr[i]] = 1
            print(temp)

        pickle.dump({'train_ids': data['train_ids'],
                     'test_ids': data['test_ids'],
                     'val_ids': data['val_ids'],
                     'data': select_l.copy()}, open('../../data/physio_features/osaka_'+str(i+1)+'_fl.pkl', 'wb'))


if __name__ == '__main__':
    incremental_select('../../data/osaka_features_corr.csv', '../../data/osaka_py27.pkl')
    # data = pickle.load(open('../../data/osaka_py27.pkl'))
    # print(data['data']['T0_ID221321_Walk2.csv'])