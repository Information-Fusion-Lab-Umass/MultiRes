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
# import cluster_vertical_lstm as cvl
import neo_cvl as cvl
label_mapping = {0: 0, 1: 1}


def fit(params, data_path, lr=0.0001):
    imputated = pickle.load(open(data_path, 'rb'))

    train = imputated['train']

    test = imputated['test']
    val = imputated['val']

    model = cvl.CVL(params).cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mode = 'normal'
    #
    if (mode == 'normal'):
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
    #
    accuracy_dict = {'prf_tr': [], 'prf_val': [], 'prf_test': []}
    #
    for iter_ in range(start_epoch, end_epoch):
        print "=#=" * 5 + str(iter_) + "=#=" * 5
        total_loss = 0
        preds_train = []
        actual_train = []

        for each_ID in tqdm(range(len(train['label']))):
            model.zero_grad()
            tag_scores = model([train['data'][each_ID]])

            _, ind_ = torch.max(tag_scores, dim=1)
            preds_train += ind_.tolist()
            curr_labels = [label_mapping[train['label'][each_ID]]]
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


def small_fit(params, data_path, start_idx, end_idx, lr):
    train = pickle.load(open(data_path, 'rb'))

    model = cvl.CVL(params).cuda()
    loss_function = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00000000002)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    end_epoch = 60
    model_name = params['model_name']

    for iter_ in range(start_epoch, end_epoch):
        print "=#=" * 5 + str(iter_) + "=#=" * 5
        total_loss = 0
        preds_train = []
        actual_train = []

        for each_ID in tqdm(range(start_idx, end_idx)):
            model.zero_grad()
            tag_scores = model([train['data'][each_ID]])

            _, ind_ = torch.max(tag_scores, dim=1)
            preds_train += ind_.tolist()
            curr_labels = [label_mapping[train['label'][each_ID]]]
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
            # print(curr_labels.reshape(tag_scores.shape[0]), tag_scores)
            total_loss += loss.item()
            #
            loss.backward()
            optimizer.step()
        #
        df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train,
                                                                  labels=[0, 1])),
                             columns=[0, 1])
        prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
        print '==' * 5 + "Epoch No:" + str(iter_) + "==" * 5
        print "Training Loss: " + str(total_loss)
        print "==" * 4
        print "Train: " + str(prf_tr)
        print df_tr


def tri_fit(params, lr=0.0001):
    imputated1 = pickle.load(open('./data/origin_physionet_1.pkl', 'rb'))

    train1 = imputated1['train']

    test1 = imputated1['test']
    val1 = imputated1['val']

    imputated2 = pickle.load(open('./data/origin_physionet_2.pkl', 'rb'))

    train2 = imputated2['train']

    test2 = imputated2['test']
    val2 = imputated2['val']

    imputated3 = pickle.load(open('./data/origin_physionet_3.pkl', 'rb'))

    train3 = imputated3['train']

    test3 = imputated3['test']
    val3 = imputated3['val']

    model1 = cvl.CVL(params).cuda()
    loss_function1 = nn.NLLLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=lr)

    model2 = cvl.CVL(params).cuda()
    loss_function2 = nn.NLLLoss()
    optimizer2 = optim.Adam(model2.parameters(), lr=lr)

    model3 = cvl.CVL(params).cuda()
    loss_function3 = nn.NLLLoss()
    optimizer3 = optim.Adam(model3.parameters(), lr=lr)

    mode = 'normal'
    #
    if (mode == 'normal'):
        print "NORMAL mode with Flags"

    batch_size = params['batch_size']
    save_flag = True
    dict_df_prf_mod = {}

    print "==x==" * 20
    print "Data Statistics"
    print "Train Data: " + str(len(train1['label']))
    print "Val Data: " + str(len(test1['label']))
    print "Test Data: " + str(len(val1['label']))
    print "==x==" * 20
    #
    start_epoch = 0
    end_epoch = 60
    model_name = params['model_name']
    #
    accuracy_dict = {'prf_tr': [], 'prf_val': [], 'prf_test': []}
    #
    for iter_ in range(start_epoch, end_epoch):
        print "=#=" * 5 + str(iter_) + "=#=" * 5
        total_loss = 0
        preds_train1 = []
        actual_train1 = []

        preds_train2 = []
        actual_train2 = []

        preds_train3 = []
        actual_train3 = []

        for each_ID in tqdm(range(len(train1['label']))):
            model1.zero_grad()
            tag_scores1 = model1([train1['data'][each_ID]])

            _, ind1_ = torch.max(tag_scores1, dim=1)
            preds_train1 += ind1_.tolist()
            curr_labels1 = [label_mapping[train1['label'][each_ID]]]
            actual_train1 += curr_labels1
    #
    #         # print('#' * 50)
    #         # print(preds_train)
    #         # print(actual_train)
    #
            curr_labels1 = torch.cuda.LongTensor(curr_labels1)
            curr_labels1 = autograd.Variable(curr_labels1)
    #
            loss1 = loss_function1(tag_scores1, curr_labels1.reshape(tag_scores1.shape[0]))
            total_loss += loss1.item()
    #
            loss1.backward()
            optimizer1.step()

            model2.zero_grad()
            tag_scores2 = model2([train2['data'][each_ID]])

            _, ind2_ = torch.max(tag_scores2, dim=1)
            preds_train2 += ind2_.tolist()
            curr_labels2 = [label_mapping[train2['label'][each_ID]]]
            actual_train2 += curr_labels2
            #
            #         # print('#' * 50)
            #         # print(preds_train)
            #         # print(actual_train)
            #
            curr_labels2 = torch.cuda.LongTensor(curr_labels2)
            curr_labels2 = autograd.Variable(curr_labels2)
            #
            loss2 = loss_function2(tag_scores2, curr_labels2.reshape(tag_scores2.shape[0]))
            total_loss += loss2.item()
            #
            loss2.backward()
            optimizer2.step()

            model3.zero_grad()
            tag_scores3 = model3([train3['data'][each_ID]])

            _, ind3_ = torch.max(tag_scores3, dim=1)
            preds_train3 += ind3_.tolist()
            curr_labels3 = [label_mapping[train3['label'][each_ID]]]
            actual_train3 += curr_labels3
            #
            #         # print('#' * 50)
            #         # print(preds_train)
            #         # print(actual_train)
            #
            curr_labels3 = torch.cuda.LongTensor(curr_labels3)
            curr_labels3 = autograd.Variable(curr_labels3)
            #
            loss3 = loss_function3(tag_scores3, curr_labels3.reshape(tag_scores3.shape[0]))
            total_loss += loss3.item()
            #
            loss3.backward()
            optimizer3.step()

            total_loss /= 3.0
        #
        #
        df_tr1 = pd.DataFrame(list(precision_recall_fscore_support(actual_train1, preds_train1,
                                                                  labels=[0, 1])),
                             columns=[0, 1])
        df_tr1.index = ['Precision', 'Recall', 'F-score', 'Count']

        prf_tr1 = precision_recall_fscore_support(actual_train1, preds_train1, average='weighted')
        prf_test1, df_test1 = eval_plot.evaluate_dbm(model1, test1, batch_size)
        prf_val1, df_val1 = eval_plot.evaluate_dbm(model1, val1, batch_size)

        df_tr2 = pd.DataFrame(list(precision_recall_fscore_support(actual_train2, preds_train2,
                                                                   labels=[0, 1])),
                              columns=[0, 1])
        df_tr2.index = ['Precision', 'Recall', 'F-score', 'Count']

        prf_tr2 = precision_recall_fscore_support(actual_train2, preds_train2, average='weighted')
        prf_test2, df_test2 = eval_plot.evaluate_dbm(model2, test2, batch_size)
        prf_val2, df_val2 = eval_plot.evaluate_dbm(model2, val2, batch_size)

        df_tr3 = pd.DataFrame(list(precision_recall_fscore_support(actual_train3, preds_train3,
                                                                   labels=[0, 1])),
                              columns=[0, 1])
        df_tr3.index = ['Precision', 'Recall', 'F-score', 'Count']

        prf_tr3 = precision_recall_fscore_support(actual_train3, preds_train3, average='weighted')
        prf_test3, df_test3 = eval_plot.evaluate_dbm(model3, test3, batch_size)
        prf_val3, df_val3 = eval_plot.evaluate_dbm(model3, val3, batch_size)

        df_tr = (df_tr1 + df_tr2 + df_tr3) / 3.0
        df_val = (df_val1 + df_val2 + df_val3) / 3.0
        df_test = (df_test1 + df_test2 + df_test3) / 3.0

        prf_tr = ((prf_tr1[0] + prf_tr2[0] + prf_tr3[0]) / 3.0,
                  (prf_tr1[1] + prf_tr2[1] + prf_tr3[1]) / 3.0,
                  (prf_tr1[2] + prf_tr2[2] + prf_tr3[2]) / 3.0, None)
        
        prf_val = ((prf_val1[0] + prf_val2[0] + prf_val3[0]) / 3.0,
                  (prf_val1[1] + prf_val2[1] + prf_val3[1]) / 3.0,
                  (prf_val1[2] + prf_val2[2] + prf_val3[2]) / 3.0, None)

        prf_test = ((prf_test1[0] + prf_test2[0] + prf_test3[0]) / 3.0,
                   (prf_test1[1] + prf_test2[1] + prf_test3[1]) / 3.0,
                   (prf_test1[2] + prf_test2[2] + prf_test3[2]) / 3.0, None)
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
            # torch.save(model, './models/' + model_name + str(iter_) + '.pt')
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
    # params = {'bilstm_flag': True,
    #           'dropout': 0.9,
    #           'layers': 1,
    #           'tagset_size': 2,
    #           'attn_category': 'dot',
    #           'num_features': 37,
    #           'input_dim': 30,
    #           'hidden_dim': 50,
    #           # 'hidden_dim': 150,
    #           # 'input_dim': 50,
    #           'max_len': 116,
    #           'batch_size': 1,
    #           'same_device': False,
    #           'same_feat_other_device': False,
    #           'model_name': 'CVL-Phy-4-',
    #           'cluster_path': './data/mst_cluster.pkl'}
    # # fit(params, '/home/sidongzhang/code/fl/data/final_Physionet_avg_new.pkl')
    # fit(params, './data/pc_physionet.pkl', lr=lr)

    params = {'bilstm_flag': True,
              'dropout': 0.9,
              'layers': 1,
              'tagset_size': 2,
              'attn_category': 'dot',
              'num_features': 37,
              'input_dim': 60,
              'hidden_dim': 100,
              # 'hidden_dim': 150,
              # 'input_dim': 50,
              # 'max_len': 116,
              'batch_size': 1,
              'same_device': False,
              'same_feat_other_device': False,
              'model_name': 'CVL-Phy-7-',
              'cluster_path': './data/mst_cluster.pkl'}
    # fit(params, '/home/sidongzhang/code/fl/data/final_Physionet_avg_new.pkl')

    # fit(params, './data/origin_physionet_1.pkl', lr=lr)

    tri_fit(params, lr=lr)

    # small_fit(params, './data/small_train.pkl', 0, 10, lr=lr)
    # fit(params)


"""
1) do dot attention for every main features using all the other features as a suuport feature set
2) go back to where we start: generate a support features set for every main feature based on correlation 
"""