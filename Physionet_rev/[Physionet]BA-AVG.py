#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cPickle as pickle
from random import shuffle
from tqdm import tqdm
# import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import batchify as batchify
import evaluate_plot as eval_plot
import rnn

import os
# 1 starts the process on GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print torch.__version__


# In[2]:


data = pickle.load(open('../../Data/final_Physionet_avg_new.pkl','rb'))
# data['data_headers']


# In[3]:


# # This is just for testing
# data['train_ids'] = data['train_ids'][:60]
# data['val_ids'] = data['val_ids'][:60]
# data['test_ids'] = data['test_ids'][:60]


# In[4]:


params = {'hidden_dim' : 150,
        'dropout' : 0.9,
        'layers' : 1,
        'input_dim' : 50,
        'tagset_size' : 2,
        'bilstm_flag': True,
        'attn_category':'dot',
         'num_of_features': 37,
         'batch_size':3,
         'model_name':'BA-Avg-Phy-4th-'}
pickle.dump(params, open('../../Models/config_'+params['model_name']+'.pt','wb'))


# In[5]:


model_RNN = rnn.RNN_osaka(params).cuda()
loss_function = nn.NLLLoss()
# optimizer = optim.Adam(model_RNN.parameters(), lr=0.01, weight_decay=0.00005)
optimizer = optim.SGD(model_RNN.parameters(), lr=0.0001, weight_decay=0.000000002)


# In[6]:


save_flag = True
dict_df_prf_mod = {}
print "=x="*20
print "Data Statistics"
print "Train Data: "+str(len(data['train_ids']))
print "Val Data: "+str(len(data['val_ids']))
print "Test Data: "+str(len(data['test_ids']))
print "=x="*20


# In[7]:


mode = 'normal'
if(mode=='normal'):
    feature_ind = 0
    label_ind = -1
    print "NORMAL mode with NO missing values"
elif(mode=='average'):
    feature_ind = 3
    label_ind = -1
    print "AVERAGE mode where missing data replaced with average values"


# In[8]:


start_epoch = 0
end_epoch = 90
model_name = params['model_name']
for iter_ in range(start_epoch, end_epoch):
    total_loss = 0
    for ind in tqdm(range(0,len(data['train_ids']),params['batch_size'])):
        model_RNN.zero_grad()
        curr_set_ids = data['train_ids'][ind:ind+params['batch_size']]
        curr_feats, curr_labels, curr_lens = batchify.prepare_batch(data['data'],
                                                                    curr_set_ids,
                                                                   params['num_of_features'],
                                                                       feature_ind, 
                                                                           label_ind)
#         print curr_set_ids
#         print curr_lens
        tag_scores = model_RNN(curr_feats, curr_lens)

        loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
        total_loss+=loss.item()

        loss.backward()
        optimizer.step()
    prf_tr, df_tr = eval_plot.evaluate_(model_RNN, data, 'train_ids', 
                                        params['batch_size'],params['num_of_features'],
                                       feature_ind, label_ind)
    prf_test, df_test = eval_plot.evaluate_(model_RNN, data, 'test_ids', 
                                            params['batch_size'],params['num_of_features'],
                                           feature_ind, label_ind)
    prf_val, df_val = eval_plot.evaluate_(model_RNN, data, 'val_ids', 
                                          params['batch_size'],params['num_of_features'],
                                         feature_ind, label_ind)
    
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


# In[ ]:





# In[ ]:





# In[9]:


eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
                      '../../Plots/'+model_name+str(iter_)+'.png',
                      0, iter_+1, 
                      model_name)


# In[ ]:





# In[ ]:




