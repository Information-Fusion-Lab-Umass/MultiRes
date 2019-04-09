#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import cPickle as pickle
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
#get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import evaluate_plot as eval_plot
import batchify as batchify
import dbm_multires as dbm
from sklearn.metrics import precision_recall_fscore_support

import os
# 1 starts the process on GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.__version__




# Data split by fast and slow Format: (fast_data, fast_missing, fast_timesteps, fast_zero_flag ,slow_data, slow_missing, slow_timesteps ,slow_zero_flag, label)
data = pickle.load(open('../../Data/final_Physionet_avg_new_split.pkl','rb'))
keep_slow_data = True



# This is just for testing
#data['train_ids'] = data['train_ids'][:10]
#data['val_ids'] = data['val_ids'][:10]
#data['test_ids'] = data['test_ids'][:10]




params = {'bilstm_flag':True,
        'hidden_dim' : 500,
        'dropout' : 0.9,
        'layers' : 1,
        'tagset_size' : 2,
        'bilstm_flag' : True,
        'attn_category' : 'dot',
        'num_features' : 37,
        'imputation_layer_dim_op':10,
        'selected_feats' : 0,
        'batch_size':1,
        'same_device':True,
        'same_feat_other_device':False,
        'model_name':'DBM-Phy-3rd-',
        'slow_features_indexes': [0, 1, 2, 3, 5, 6, 16, 27, 28, 31, 32],
        'fast_features_indexes': [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 33, 34, 35,36]}
pickle.dump(params, open('../../Models/config_'+params['model_name']+'.pt','wb'))




model_RNN = dbm.RNN_osaka(params).cuda()
loss_function = nn.NLLLoss()
# optimizer = optim.Adam(model_RNN.parameters(), lr=0.01, weight_decay=0.00005)
optimizer = optim.SGD(model_RNN.parameters(), lr=0.0001, weight_decay=0.000000002)




mode = 'normal'
if(mode=='normal'):
    feature_ind = 0
    label_ind = -1
    print "NORMAL mode with Flags"




batch_size = 1
epochs = 45
save_flag = True
dict_df_prf_mod = {}
print "==x=="*20
print "Data Statistics"
print "Train Data: "+str(len(data['train_ids']))
print "Val Data: "+str(len(data['val_ids']))
print "Test Data: "+str(len(data['test_ids']))
print "==x=="*20



start_epoch = 0
end_epoch = 60
model_name = params['model_name']
for iter_ in range(start_epoch, end_epoch):
    print "=#="*5+str(iter_)+"=#="*5
    total_loss = 0
    preds_train = []
    actual_train = []
    for each_ID in tqdm(data['train_ids']):
        model_RNN.zero_grad()

        # Skip patients for which there is no slow data
        if keep_slow_data == False:
            if data['data'][each_ID][7] == 1:
                continue
        tag_scores = model_RNN(data['data'], each_ID)

        _, ind_ = torch.max(tag_scores, dim=1)
        preds_train+=ind_.tolist()

        # For this dataset the label is in -2
        curr_labels = [data['data'][each_ID][label_ind]]
        curr_labels = [batchify.label_mapping[x] for x in curr_labels]
        actual_train+=curr_labels
        curr_labels = torch.cuda.LongTensor(curr_labels)
        curr_labels = autograd.Variable(curr_labels)

        loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
        total_loss+=loss.item()

        loss.backward()
        optimizer.step()

    df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train,
                                                              labels = [0,1])),
                                                             columns = [0,1])
    df_tr.index = ['Precision','Recall','F-score','Count']
    prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
    prf_test, df_test = eval_plot.evaluate_dbm(model_RNN, data, 'test_ids')
    prf_val, df_val = eval_plot.evaluate_dbm(model_RNN, data, 'val_ids')

    df_all = pd.concat([df_tr, df_val, df_test],axis=1)
    dict_df_prf_mod['Epoch'+str(iter_)] = df_all

    print '=='*5 + "Epoch No:"+str(iter_) +"=="*5
    print "Training Loss: "+str(total_loss)
    print "=="*4
    print "Train: " + str(prf_tr)
    print df_tr
    print "--"*4
    print "Val: " + str(prf_val)
    print df_val
    print "--"*4
    print "Test: " + str(prf_test)
    print df_test
    print '=='*40
    print '\n'
    if(save_flag):
        torch.save(model_RNN, '../../Models/'+model_name+str(iter_)+'.pt')
        pickle.dump(dict_df_prf_mod, open('../../Results/dict_prf_'+model_name+str(iter_)+'.pkl','wb'))
        eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
                              '../../Plots/'+model_name+str(iter_)+'.png',
                              0, iter_+1, 
                              model_name)

eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
                              '../../Plots/'+model_name+str(iter_)+'.png',
                              0, iter_, 
                              model_name)

