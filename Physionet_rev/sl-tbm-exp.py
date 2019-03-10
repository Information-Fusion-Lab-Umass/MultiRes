#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
cwd = os.getcwd()
module_path = os.path.abspath(os.path.join('..'))
student_life_path = module_path + "/student_life"
physnet_path = module_path + "/Physionet_rev"
print(student_life_path)
print(physnet_path)

sys.path.append(physnet_path)
sys.path.append(student_life_path)

import pandas as pd
import numpy as np
import pickle
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
print(torch.cuda.is_available())

import tbm as tbm
from net_dbm import RNN_osaka


import evaluate_plot as eval_plot
import batchify as batchify
from sklearn.metrics import precision_recall_fscore_support
import importlib
import src.utils.student_utils as student_utils
import src.definitions as definitions
from IPython.display import display
importlib.reload(student_utils)

# 1 starts the process on GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.__version__
get_ipython().magic(u'matplotlib inline')
print(eval_plot)
print(sys.version_info)


# In[2]:


torch.cuda.is_available()


# In[3]:


data = None
pickle_path = student_life_path + '/data/training_data/student_life_pickle.pkl'
model_path = student_life_path + '/models'
print('pickle_path', pickle_path)
print('model_config_path', model_path)


# In[4]:


with open(pickle_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file)
print(data.keys())


# In[5]:


params = {'bilstm_flag':True,
        'hidden_dim' : 32,
        'input_dim':50,
        'dropout' : 0.9,
        'layers' : 1,
        'tagset_size' : 5,
        'bilstm_flag' : True,
        'attn_category' : 'dot',
        'num_features' : 10,
        'batch_size':1,
        'model_name':'TBM-SL-'}

model_config_path = model_path + "/" + params['model_name']+'.pt'
print(model_config_path)
with open(model_config_path, 'wb') as config_file:
    pickle.dump(params, config_file)
    


# In[6]:


model_RNN = tbm.RNN_osaka(params)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model_RNN.parameters(), lr=0.0001, weight_decay=0.000000002)


# In[7]:


mode = 'normal'
if(mode=='normal'):
    feature_ind = 0
    label_ind = -1
    print("NORMAL mode with Flags")


# In[8]:


batch_size = 1
epochs = 45
save_flag = True
dict_df_prf_mod = {}
print("==x=="*20)
print("Data Statistics")
print("Train Data: "+str(len(data['train_ids'])))
print("Val Data: "+str(len(data['val_ids'])))
print("Test Data: "+str(len(data['test_ids'])))
print("==x=="*20)


# In[9]:


start_epoch = 0
end_epoch = 90
model_name = params['model_name']
target_classes = [0,1,2,3,4]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("DEVICE: {}".format(device))
for iter_ in range(start_epoch, end_epoch):
    print ("=#="*5+str(iter_)+"=#="*5)
    total_loss = 0
    preds_train = []
    actual_train = []
    for each_ID in tqdm(data['train_ids']):
        model_RNN.zero_grad()
        tag_scores = model_RNN(data['data'], each_ID)
        
        _, ind_ = torch.max(tag_scores, dim=1)
        preds_train+=ind_.tolist()
        
        curr_labels = [data['data'][each_ID][-1]]
        actual_train+=curr_labels
        curr_labels = torch.LongTensor(curr_labels).to(device)
        curr_labels = autograd.Variable(curr_labels)
        
        loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
    
    df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train, labels = target_classes)),
                       columns = list(range(len(target_classes))))
    df_tr.index = ['Precision','Recall','F-score','Count']
    prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
    prf_test, df_test = eval_plot.evaluate_dbm(model_RNN, data, 'test_ids', target_n=target_classes)

    prf_val, df_val = eval_plot.evaluate_dbm(model_RNN, data, 'val_ids', target_n=target_classes)
    
    df_all = pd.concat([df_tr, df_val, df_test],axis=1)
    dict_df_prf_mod['Epoch'+str(iter_)] = df_all
    
    print('=='*5 + "Epoch No:"+str(iter_) +"=="*5)
    print("Training Loss: "+str(total_loss))
    print("=="*4)
    print("Train: " + str(prf_tr))
    print(df_tr)
    print("--"*4)
    print("Val: " + str(prf_val))
    print(df_val)
    print("--"*4)
    print("Test: " + str(prf_test))
    print(df_test)
    print('=='*40)
    print('\n')

    eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
                  model_path + "/" +model_name+str(iter_)+'.png',
                  0, iter_+1, 
                  model_name, target_n=len(target_classes))

#     if(save_flag):
#         torch.save(model_RNN, '../../Models/'+model_name+str(iter_)+'.pt')
#         pickle.dump(dict_df_prf_mod, open('../../Results/dict_prf_'+model_name+str(iter_)+'.pkl','wb'))
#         eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
#                               '../../Plots/'+model_name+str(iter_)+'.png',
#                               0, iter_+1, 
#                               model_name)


# In[ ]:


print(data)


# In[ ]:





# In[ ]:





# In[ ]:




