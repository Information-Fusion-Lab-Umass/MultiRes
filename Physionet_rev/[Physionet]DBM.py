
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from random import shuffle
from tqdm import tqdm
# import matplotlib.pyplot as plt
import random
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import evaluate_plot as eval_plot
import batchify as batchify
import dbm as dbm
from sklearn.metrics import precision_recall_fscore_support

import os

print('CUDA_AVAIL: ' + str(torch.cuda.is_available()))
print(torch.cuda.device_count())
# 1 starts the process on GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.__version__


# In[2]:


data = pickle.load(open('data/final_Physionet_avg_new.pkl','rb'))


# In[3]:


def get_fast_slow_features_per_ID(data, key, num_feats = 37):
    """
    Calculates the normalized sum of the delta_t of each feature PER ID and saves it to a csv file
    """
    import csv
    with open("output.csv", "a", newline='') as fp:
        for ID in data[key]:
           # print('Running ID: '+ str(ID))
            data_matrix = np.matrix(data['data'][ID][0])
            missing_matrix = np.matrix(data['data'][ID][1])
            t = np.array(data['data'][ID][2])
            delta_matrix= np.full_like(missing_matrix, -1)
            #iterate through each feature
            for feat_ind in range(num_feats):
                current_feature = missing_matrix[:,feat_ind]
                delta_t = 0
                #iterate through current feature. if the missing_matrix says the value is missing, then add 1 to delta_t. if the value is not missing, reset delta_t to 0
                for i, ind in enumerate(current_feature):
                        flag = int(ind)
                        if(flag==0):
                            delta_t = 0 
                        delta_matrix[i,feat_ind] = delta_t
                        delta_t+=1

             #get sum of deltas
            delta_sums = delta_matrix.sum(axis = 0)
            #normalize this because in the future there will be variable length t's, so having a scale 0 to 1 is ideal
            delta_sums = delta_sums / np.sum(np.arange(len(t)))
            delta_sums[delta_sums > 0.5] = 1
            delta_sums[delta_sums <= 0.5] = 0
            y = delta_sums.tolist()[0]
            y.append(ID)
            print(y)
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(y)
                #f.write( str(ID) + ": " + str(delta_sums) + "\n" )
    


# In[7]:


def get_fast_slow_features_entire_dataset(data, key, num_feats = 37):
    """
    Calculates the normalized sum of the delta_t of each feature to classify fast/slow features. Closer to 0 is a fast feature. Closer to 1 is a slow feature.
    """
    data_matrix = None
    for ID, data_tuple in data['data'].items():
        print('Running ID: '+ str(ID))
        if data_matrix is None:
            data_matrix = np.matrix(data_tuple[0])
            missing_matrix = np.matrix(data_tuple[1])
        else:
            current_data = np.matrix(data_tuple[0])
            current_missing = np.matrix(data_tuple[1])
            data_matrix = np.append(data_matrix,current_data,axis = 0)
            missing_matrix = np.append(missing_matrix,current_missing, axis = 0)
            
    print(data_matrix.shape)
    print(missing_matrix.shape)
            
        #t = np.array(data['data'][ID][2])
        
    
    delta_matrix= np.full_like(missing_matrix, -1)
    #iterate through each feature
    for feat_ind in range(num_feats):
        current_feature = missing_matrix[:,feat_ind]
        delta_t = 0
        #iterate through current feature. if the missing_matrix says the value is missing, then add 1 to delta_t. if the value is not missing, reset delta_t to 0
        for i, ind in enumerate(current_feature):
                flag = int(ind)
                if(flag==0):
                    delta_t = 0 
                delta_matrix[i,feat_ind] = delta_t
                delta_t+=1

     #get sum of deltas
    delta_sums = delta_matrix.sum(axis = 0)
    #normalize it
    delta_sums = delta_sums / np.sum(np.arange(missing_matrix.shape[0]))
    print(delta_sums)
        #slow_indexes = 
        #fast_indexes =
    


# In[8]:


#uncomment this to get the scores for each feature over the entire dataset
#get_fast_slow_features_entire_dataset(data,'train_ids')


# In[24]:


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
        'slow_features_indexes': [0,1,2,3,5,6,16,27,28,31,32], #there were calculated by using get_fast_slow_features_entire_dataset()
         'fast_features_indexes': [4,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,29,30,33,34,35,36] }



#pickle.dump(params, open('../../Models/config_'+params['model_name']+'.pt','wb'))


# In[25]:


model_RNN = dbm.RNN_osaka(params).cuda()
loss_function = nn.NLLLoss()
# optimizer = optim.Adam(model_RNN.parameters(), lr=0.01, weight_decay=0.00005)
optimizer = optim.SGD(model_RNN.parameters(), lr=0.0001, weight_decay=0.000000002)


# In[26]:


mode = 'normal'
if(mode=='normal'):
    feature_ind = 0
    label_ind = -1
    print("NORMAL mode with Flags")


# In[27]:


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


# In[28]:


start_epoch = 0
end_epoch = 60
model_name = params['model_name']
for iter_ in range(start_epoch, end_epoch):
    print("=#="*5+str(iter_)+"=#="*5)
    total_loss = 0
    preds_train = []
    actual_train = []
    for each_ID in tqdm(data['train_ids']):
        model_RNN.zero_grad()
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
#     prf_tr, df_tr = evaluate_(model_RNN, data, 'train_ids')
    prf_test, df_test = eval_plot.evaluate_dbm(model_RNN, data, 'test_ids')
    prf_val, df_val = eval_plot.evaluate_dbm(model_RNN, data, 'val_ids')
    
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
    if(save_flag):
        torch.save(model_RNN, '../../Models/'+model_name+str(iter_)+'.pt')
        pickle.dump(dict_df_prf_mod, open('../../Results/dict_prf_'+model_name+str(iter_)+'.pkl','wb'))
        eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
                              '../../Plots/'+model_name+str(iter_)+'.png',
                              0, iter_+1, 
                              model_name)


# In[ ]:


eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', 
                              '../../Plots/'+model_name+str(iter_)+'.png',
                              0, iter_, 
                              model_name)

