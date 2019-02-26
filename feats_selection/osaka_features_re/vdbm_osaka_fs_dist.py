import csv
import sys

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
import vdbm as dbm


is_cuda = torch.cuda.is_available()

path_pre = '/home/sidongzhang/code/fl'
# path_pre = '../..'


def read_corr_csv(file_name):
    """
    :param file_name: csv matrix file path
    :return: 21 * 21 matrix. The first 3 physio_features are aborted so we do not store their correlation value
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
    pickle.dump(feat_select_dict, open(path_pre + '/data/dict_selected_feats_osaka'+str(corr_num) + '.pkl', 'wb'))


def v_fit(params, corr_num, data_path):
    data = pickle.load(open(data_path, 'rb'))

    model_RNN = dbm.RNN_osaka(params).cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model_RNN.parameters(), lr=0.007, weight_decay=0.00000000002)

    mode = 'normal'
    if (mode == 'normal'):
        feature_ind = 0
        label_ind = -1
        print "NORMAL mode with Flags"

    batch_size = 1
    epochs = 45
    save_flag = True
    dict_df_prf_mod = {}
    print "==x==" * 20
    print "Data Statistics"
    print "Train Data: " + str(len(data['train_ids']))
    print "Val Data: " + str(len(data['val_ids']))
    print "Test Data: " + str(len(data['test_ids']))
    print "==x==" * 20

    accuracy_dict = {'prf_tr': [], 'prf_val': [], 'prf_test': []}

    start_epoch = 0
    end_epoch = 60
    model_name = params['model_name']
    for iter_ in range(start_epoch, end_epoch):
        print "=#=" * 5 + str(iter_) + "=#=" * 5
        total_loss = 0
        preds_train = []
        actual_train = []
        for each_ID in tqdm(data['train_ids']):
            model_RNN.zero_grad()
            tag_scores = model_RNN(data['data'], each_ID)

            _, ind_ = torch.max(tag_scores, dim=1)
            preds_train += ind_.tolist()

            curr_labels = [data['data'][each_ID][label_ind]]
            curr_labels = [batchify.label_mapping[x] for x in curr_labels]
            actual_train += curr_labels
            curr_labels = torch.cuda.LongTensor(curr_labels)
            curr_labels = autograd.Variable(curr_labels)

            loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train,
                                                                  labels=[0, 1, 2])),
                             columns=[0, 1, 2])
        df_tr.index = ['Precision', 'Recall', 'F-score', 'Count']
        prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
        #     prf_tr, df_tr = evaluate_(model_RNN, data, 'train_ids')
        prf_test, df_test = eval_plot.evaluate_dbm(model_RNN, data, 'test_ids')
        prf_val, df_val = eval_plot.evaluate_dbm(model_RNN, data, 'val_ids')

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
            torch.save(model_RNN, path_pre + '/models/' + model_name + str(iter_) + '.pt')
            pickle.dump(dict_df_prf_mod, open(path_pre + '/results/dict_prf_' + model_name + str(iter_) + '.pkl', 'wb'))
            eval_plot.plot_graphs(dict_df_prf_mod, 'F-score',
                                  path_pre + '/plots/' + model_name + str(iter_) + '.png',
                                  0, iter_ + 1,
                                  model_name)

        accuracy_dict['prf_tr'].append(prf_tr)
        accuracy_dict['prf_test'].append(prf_test)
        accuracy_dict['prf_val'].append(prf_val)

    pickle.dump(accuracy_dict, open(path_pre + '/results/prf' + str(corr_num) + '.pkl', 'wb'))


if __name__ == '__main__':
    params = sys.argv[1:]
    if not params:
        i = 5
    else:
        i = int(params[0])
    feats_select(path_pre + '/data/osaka_features_corr.csv', i)
    print('num of select physio_features:', str(i))

    params = {'bilstm_flag': True,
              'hidden_dim': 200,
              'dropout': 0.9,
              'layers': 1,
              'tagset_size': 3,
              'attn_category': 'dot',
              'num_features': 24,
              'imputation_layer_dim_op': 20,
              'selected_feats': i,
              'batch_size': 1,
              'same_device': False,
              'same_feat_other_device': False,
              'model_name': 'VDBM-Osaka-Full-Oracle' + str(i) + '-',
              'feats_provided_flag': True,
              'path_selected_feats': path_pre + '/data/dict_selected_feats_osaka'+str(i) + '.pkl'}

    v_fit(params, i, path_pre + '/data/final_osaka_data_avg_60_flags.pkl')
