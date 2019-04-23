import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import pandas as pd

import cPickle as pickle


from sklearn.metrics import precision_recall_fscore_support

import evaluate_plot as eval_plot
# import cluster_vertical_lstm as cvl
import tbm as tbm

label_mapping = {0: 0, 1: 1}


def tri_fit(params, lr=0.0001):
    imputed = [pickle.load(open('./data/tbm_physionet_%d.pkl' % i, 'rb')) for i in range(1, 4)]
    trains = [imputed[i]['train'] for i in range(0, 3)]
    tests = [imputed[i]['test'] for i in range(0, 3)]
    vals = [imputed[i]['val'] for i in range(0, 3)]

    models = [tbm.tbm(params).cuda() for _ in range(0, 3)]
    optimizers = [optim.Adam(models[i].parameters(), lr=lr) for i in range(0, 3)]
    loss_functions = [nn.NLLLoss() for _ in range(0, 3)]

    mode = 'normal'
    #
    if (mode == 'normal'):
        print "NORMAL mode with Flags"

    batch_size = params['batch_size']
    save_flag = True
    dict_df_prf_mod = {}

    dict_df_prf_mod_sp = [{}, {}, {}]

    assert (len(trains[0]['label']) == len(trains[1]['label']))
    assert (len(trains[0]['label']) == len(trains[2]['label']))

    assert (len(tests[0]['label']) == len(tests[1]['label']))
    assert (len(tests[0]['label']) == len(tests[2]['label']))

    assert (len(vals[0]['label']) == len(vals[1]['label']))
    assert (len(vals[0]['label']) == len(vals[2]['label']))

    print "==x==" * 20
    print "Data Statistics"
    print "Train Data: " + str(len(trains[0]['label']))
    print "Val Data: " + str(len(tests[0]['label']))
    print "Test Data: " + str(len(vals[0]['label']))
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

        preds_trains = [[], [], []]
        actual_trains = [[], [], []]

        for each_ID in tqdm(range(len(trains[0]['label']))):
            for i in range(0, 3):
                models[i].zero_grad()
                tag_scores = models[i]([trains[i]['data'][each_ID]])

                _, ind_ = torch.max(tag_scores, dim=1)
                preds_trains[i] += ind_.tolist()
                curr_labels = [label_mapping[trains[i]['label'][each_ID]]]
                actual_trains[i] += curr_labels

                curr_labels = torch.cuda.LongTensor(curr_labels)
                curr_labels = autograd.Variable(curr_labels)
                #
                loss = loss_functions[i](tag_scores, curr_labels.reshape(tag_scores.shape[0]))
                total_loss += loss.item() / 3.0
                #
                loss.backward()
                optimizers[i].step()
        #
        #
        df_trs = [pd.DataFrame(list(precision_recall_fscore_support(actual_trains[i], preds_trains[i],
                                                                  labels=[0, 1])),
                             columns=[0, 1]) for i in range(0, 3)]
        for i in range(0, 3):
            df_trs[i].index = ['Precision', 'Recall', 'F-score', 'Count']

        prf_trs = [precision_recall_fscore_support(actual_trains[i], preds_trains[i], average='weighted') for i in range(0, 3)]

        prf_tests = []
        prf_vals = []
        df_tests = []
        df_vals = []
        for i in range(0, 3):
            prf_test, df_test = eval_plot.evaluate_dbm(models[i], tests[i], batch_size)
            prf_val, df_val = eval_plot.evaluate_dbm(models[i], vals[i], batch_size)
            prf_tests.append(prf_test)
            prf_vals.append(prf_val)
            df_tests.append(df_test)
            df_vals.append(df_val)

        df_tr = (df_trs[0] + df_trs[1] + df_trs[2]) / 3.0
        df_val = (df_vals[0] + df_vals[1] + df_vals[2]) / 3.0
        df_test = (df_tests[0] + df_tests[1] + df_tests[2]) / 3.0

        prf_tr = ((prf_trs[0][0] + prf_trs[1][0] + prf_trs[2][0]) / 3.0,
                  (prf_trs[0][1] + prf_trs[1][1] + prf_trs[2][1]) / 3.0,
                  (prf_trs[0][2] + prf_trs[1][2] + prf_trs[2][2]) / 3.0, None)

        prf_val = ((prf_vals[0][0] + prf_vals[1][0] + prf_vals[2][0]) / 3.0,
                  (prf_vals[0][1] + prf_vals[1][1] + prf_vals[2][1]) / 3.0,
                  (prf_vals[0][2] + prf_vals[1][2] + prf_vals[2][2]) / 3.0, None)

        prf_test = ((prf_tests[0][0] + prf_tests[1][0] + prf_tests[2][0]) / 3.0,
                   (prf_tests[0][1] + prf_tests[1][1] + prf_tests[2][1]) / 3.0,
                   (prf_tests[0][2] + prf_tests[1][2] + prf_tests[2][2]) / 3.0, None)
    #
        df_all = pd.concat([df_tr, df_val, df_test], axis=1)
        dict_df_prf_mod['Epoch' + str(iter_)] = df_all

        df_sp = [pd.concat([df_trs[i], df_vals[i], df_tests[i]], axis=1) for i in range(0, 3)]
        for i in range(0, 3):
            dict_df_prf_mod_sp[i]['Epoch' + str(iter_)] = df_sp[i]

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

            for i in range(0, 3):
                torch.save(models[i], './models/' + model_name + str(iter_) + '_%d.pt' % i)
                pickle.dump(dict_df_prf_mod_sp[i],
                            open('./results/prf_%d_' % i + model_name + str(iter_) + '.pkl', 'wb'))
                eval_plot.plot_graphs(dict_df_prf_mod_sp[i], 'F-score',
                                      './plots/' + model_name + str(iter_) + '_%d.png' % i,
                                      0, iter_ + 1,
                                      model_name)
        accuracy_dict['prf_tr'].append(prf_tr)
        accuracy_dict['prf_test'].append(prf_test)
        accuracy_dict['prf_val'].append(prf_val)

    pickle.dump(accuracy_dict, open('./results/physio_final_prf_' + model_name + '.pkl', 'wb'))


if __name__ == '__main__':
    p = sys.argv[1:]
    if not p:
        lr = 5e-5
        name = 'CVL-Phy-'
    else:
        lr = float(p[0])
        name = p[1]
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
              'input_dim': 50,
              'hidden_dim': 100,
              # 'input_dim': 60,
              # 'hidden_dim': 100,
              # 'hidden_dim': 150,
              # 'input_dim': 50,
              # 'max_len': 116,
              'batch_size': 1,
              'same_device': False,
              'same_feat_other_device': False,
              'model_name': name}
    # fit(params, '/home/sidongzhang/code/fl/data/final_Physionet_avg_new.pkl')

    # fit(params, './data/origin_physionet_1.pkl', lr=lr)

    tri_fit(params, lr=lr)