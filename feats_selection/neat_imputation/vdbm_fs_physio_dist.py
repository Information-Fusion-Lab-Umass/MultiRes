import sys
import csv

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from tqdm import tqdm
import pandas as pd

import cPickle as pickle


from sklearn.metrics import precision_recall_fscore_support

import os

import batchify
import evaluate_plot as eval_plot
import imputation as vdbm


is_cuda = torch.cuda.is_available()


def read_corr_csv(file_name):
    """
    :param file_name: csv matrix file path
    :return: 37 * 37 matrix. The first 3 features are aborted so we do not store their correlation value
    """
    with open(file_name, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')

        next(csvreader)

        data_ = []
        for row in csvreader:
            data_.append([float(row[i]) for i in range(1, len(row))])

    return np.array(data_)


def feats_select(corr_path, corr_num):
    corr = read_corr_csv(corr_path)
    abs_corr = np.abs(corr)
    idx_rank = np.argsort(abs_corr, axis=1)
    idx_rank = idx_rank[:, : -1]

    feat_select_dict = {}
    for i in range(idx_rank.shape[0]):
        feat_select_dict[i] = list(idx_rank[i, -corr_num:])
    pickle.dump(feat_select_dict, open('/home/sidongzhang/code/fl/data/dict_selected_feats_physionet'+str(corr_num) + '.pkl', 'wb'))


# data_type: train_ids, test_ids, val_ids
def pre_imputation(data_path, data_type, num_features=37):
    # return a dict {'data_id': {data: {feat0: [seq_len, 4]matrix, feat1: [seq_len, 4]matrix, ...}, label: 0}, ...}
    imputation = {}
    data = pickle.load(open(data_path, 'rb'))
    for i in data[data_type]:
        d = data['data'][i]
        feats = np.asarray(d[0])
        # print(feats)
        flags = np.asarray(d[1])
        label = d[3]
        all_features = []
        # num_features = self.num_features
        input_ = {}

        # for every 24 features
        for feat_ind in range(num_features):
            input_[feat_ind] = []
            feat = feats[:, feat_ind]  # historical data for ith feats
            feat_flag = flags[:, feat_ind]  # corresponding missing labels for ith feats
            ind_keep = feat_flag == 0
            ind_missing = feat_flag == 1
            if (sum(ind_keep) > 0):  # if in the whole historical data there exists at least one missing point
                avg_val = np.mean(feat[ind_keep])  # we calculate the mean feats value for ith feat
            else:
                avg_val = 0.0
            last_val_observed = avg_val
            delta_t = -1
            for ind, each_flag in enumerate(feat_flag):
                # we visit all the historical data point for ith feat
                # and insert all the imputed historical data [feat_value, avg_value, is_missing, delta_time] into a dict
                if (each_flag == 1):
                    imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                    input_[feat_ind].append(imputation_feat)
                #                     input_[feat_ind][ind] = autograd.Variable(torch.cuda.FloatTensor(imputation_feat))
                #                     f_ = self.imputation_layer_in[feat_ind](input_)
                elif (each_flag == 0):
                    delta_t = 0
                    last_val_observed = feat[ind]
                    imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                    input_[feat_ind].append(imputation_feat)
                delta_t += 1
        imputation[i] = {'data': input_, 'label': label}

    return imputation


def v_fit(params, corr_num, data_path):
    # data = pickle.load(open(data_path, 'rb'))

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model_RNN = vdbm.Imputation(params).cuda()
    else:
        model_RNN = vdbm.Imputation(params)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model_RNN.parameters(), lr=0.0001, weight_decay=0.00000000002)
    mode = 'normal'

    if (mode == 'normal'):
        feature_ind = 0
        label_ind = -1
        print "NORMAL mode with Flags"

    batch_size = 1
    epochs = 45
    save_flag = True
    dict_df_prf_mod = {}

    start_epoch = 0
    end_epoch = 60
    model_name = params['model_name']

    train_imputation = pre_imputation(data_path, 'train_ids')
    test_imputation = pre_imputation(data_path, 'test_ids')
    val_imputation = pre_imputation(data_path, 'val_ids')

    print "==x==" * 20
    print "Data Statistics"
    print "Train Data: " + str(len(train_imputation.keys()))
    print "Val Data: " + str(len(test_imputation.keys()))
    print "Test Data: " + str(len(val_imputation.keys()))
    print "==x==" * 20

    accuracy_dict = {'prf_tr': [], 'prf_val': [], 'prf_test': []}

    for iter_ in range(start_epoch, end_epoch):
        print "=#=" * 5 + str(iter_) + "=#=" * 5
        total_loss = 0
        preds_train = []
        actual_train = []
        for each_ID in tqdm(train_imputation.keys()):
            model_RNN.zero_grad()
            tag_scores = model_RNN(train_imputation[each_ID]['data'])

            _, ind_ = torch.max(tag_scores, dim=1)
            preds_train += ind_.tolist()
            # For this dataset the label is in -2
            curr_labels = [train_imputation[each_ID]['label']]
            curr_labels = [batchify.label_mapping[x] for x in curr_labels]
            actual_train += curr_labels
            if is_cuda:
                curr_labels = torch.cuda.LongTensor(curr_labels)
            else:
                curr_labels = torch.LongTensor(curr_labels)
            curr_labels = autograd.Variable(curr_labels)

            loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train,
                                                                  labels=[0, 1])),
                             columns=[0, 1])
        df_tr.index = ['Precision', 'Recall', 'F-score', 'Count']
        prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
        #     prf_tr, df_tr = evaluate_(model_RNN, data, 'train_ids')
        prf_test, df_test = eval_plot.evaluate_dbm(model_RNN, test_imputation)
        prf_val, df_val = eval_plot.evaluate_dbm(model_RNN, val_imputation)

        df_all = pd.concat([df_tr, df_val, df_test], axis=1)
        dict_df_prf_mod['Epoch' + str(iter_)] = df_all

        print '==' * 5 + "Epoch No:" + str(iter_) + "==" * 5
        print "Training Loss: " + str(total_loss)
        print "==" * 4
        print "Train: " + str(prf_tr)
        print df_tr
        print "--" * 4
        print "Val: " + str(prf_val)
        print df_val
        print "--" * 4
        print "Test: " + str(prf_test)
        print df_test
        print '==' * 40
        print '\n'

        if (save_flag):
            torch.save(model_RNN, '/home/sidongzhang/code/fl/models/' + model_name + str(iter_) + '.pt')
            pickle.dump(dict_df_prf_mod, open('/home/sidongzhang/code/fl/results/dict_prf_' + model_name + str(iter_) + '.pkl', 'wb'))
            eval_plot.plot_graphs(dict_df_prf_mod, 'F-score',
                                 '/home/sidongzhang/code/fl/plots/' + model_name + str(iter_) + '.png',
                                 0, iter_ + 1,
                                 model_name)
        accuracy_dict['prf_tr'].append(prf_tr)
        accuracy_dict['prf_test'].append(prf_test)
        accuracy_dict['prf_val'].append(prf_val)

    pickle.dump(accuracy_dict, open('/home/sidongzhang/code/fl/results/neo_phy_' + str(corr_num) + '.pkl', 'wb'))


if __name__ == '__main__':
    params = sys.argv[1:]
    if not params:
        i = 1
    else:
        i = int(params[0])
    feats_select('/home/sidongzhang/code/fl/data/intercorr_physionet.csv', i)
    params = {'bilstm_flag': True,
              'dropout': 0.9,
              'layers': 1,
              'tagset_size': 3,
              'attn_category': 'dot',
              'num_features': 37,
              'imputation_layer_dim_op': 20,
              'selected_feats': 3,
              'batch_size': 1,
              'same_device': False,
              'same_feat_other_device': False,
              'model_name': 'NEO-Phy'+str(i)+'-',
              'feats_provided_flag': True,
              'path_selected_feats': '/home/sidongzhang/code/fl/data/dict_selected_feats_physionet'+str(i) + '.pkl'}
    v_fit(params, i, '/home/sidongzhang/code/fl/data/final_Physionet_avg_new.pkl')
