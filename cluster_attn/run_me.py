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

import evaluate_plot as eval_plot
import imputation
import cluster_vertical_lstm as cvl

label_mapping = {0: 0, 1: 1}


def fit(params, data_path, lr=0.0001):
    # print('#'*10 + 'imputing ...')
    # imputated = imputation.get_imputation(data_path)
    # print('#' * 10 + 'end of imputing ...')
    imputated = pickle.load(open(data_path, 'rb'))
    #
    train = imputated['train']
    test = imputated['test']
    val = imputated['val']

    model = cvl.CVL(params).cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00000000002)
    mode = 'normal'
    #
    if (mode == 'normal'):
        feature_ind = 0
        label_ind = -1
        print "NORMAL mode with Flags"

    batch_size = params['batch_size']
    save_flag = True
    dict_df_prf_mod = {}

    print "==x==" * 20
    print "Data Statistics"
    print "Train Data: " + str(len(train['label']))
    print "Val Data: " + str(len(test['label']))
    print "Test Data: " + str(len(val['label']))
    print "==x==" * 20
    #
    start_epoch = 0
    end_epoch = 60
    model_name = params['model_name']

    train_size = len(train['label'])
    #
    accuracy_dict = {'prf_tr': [], 'prf_val': [], 'prf_test': []}
    #
    for iter_ in range(start_epoch, end_epoch):
        print "=#=" * 5 + str(iter_) + "=#=" * 5
        total_loss = 0
        preds_train = []
        actual_train = []

        for i in tqdm(range(train_size / batch_size + 1)):
            start_id = i * batch_size
            if i < train_size / batch_size:
                end_id = (i + 1) * batch_size
            else:
                end_id = train_size

            train_block = train['data'][start_id: end_id]
            train_label_block = train['label'][start_id: end_id]
            train_len_block = train['lens'][start_id: end_id]

    #     for each_ID in tqdm(range(len(train['label']))):
            model.zero_grad()
            tag_scores = model(train_block)
    #
            _, ind_ = torch.max(tag_scores, dim=1)
            preds_train += ind_.tolist()
    #         curr_label = train['label'][each_ID]
            curr_labels = [label_mapping[l] for l in train_label_block]
            actual_train += curr_labels
    #
    #         # print('#' * 50)
    #         # print(preds_train)
    #         # print(actual_train)
    #
            curr_labels = torch.cuda.LongTensor(curr_labels)
            curr_labels = autograd.Variable(curr_labels)
    #
            loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
            total_loss += loss.item()
    #
            loss.backward()
            optimizer.step()
    #
        df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train,
                                                                  labels=[0, 1])),
                             columns=[0, 1])
        df_tr.index = ['Precision', 'Recall', 'F-score', 'Count']
        prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
    #     #     prf_tr, df_tr = evaluate_(model_RNN, data, 'train_ids')
        prf_test, df_test = eval_plot.evaluate_dbm(model, test, batch_size)
        prf_val, df_val = eval_plot.evaluate_dbm(model, val, batch_size)
    #
        df_all = pd.concat([df_tr, df_val, df_test], axis=1)
        dict_df_prf_mod['Epoch' + str(iter_)] = df_all
    #
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
            torch.save(model, './models/' + model_name + str(iter_) + '.pt')
            pickle.dump(dict_df_prf_mod,
                        open('./results/prf_' + model_name + str(iter_) + '.pkl', 'wb'))
            eval_plot.plot_graphs(dict_df_prf_mod, 'F-score',
                                  './plots/' + model_name + str(iter_) + '.png',
                                  0, iter_ + 1,
                                  model_name)
        accuracy_dict['prf_tr'].append(prf_tr)
        accuracy_dict['prf_test'].append(prf_test)
        accuracy_dict['prf_val'].append(prf_val)

    pickle.dump(accuracy_dict, open('./results/physio_final_prf_' + model_name + '.pkl', 'wb'))


if __name__ == '__main__':
    p = sys.argv[1:]
    if not p:
        lr = 1e-6
    else:
        lr = float(p[0])
    params = {'bilstm_flag': True,
              'dropout': 0.9,
              'layers': 1,
              'tagset_size': 2,
              'attn_category': 'dot',
              'num_features': 37,
              'input_dim': 120,
              'hidden_dim': 150,
              # 'hidden_dim': 150,
              # 'input_dim': 50,
              'max_len': 100,
              'batch_size': 5,
              'same_device': False,
              'same_feat_other_device': False,
              'model_name': 'CVL-Phy',
              'cluster_path': './data/dummy_cluster.pkl'}
    # fit(params, '/home/sidongzhang/code/fl/data/final_Physionet_avg_new.pkl')
    fit(params, './data/pc_physionet.pkl', lr=lr)
    # fit(params)
