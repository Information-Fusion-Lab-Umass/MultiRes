import pickle
import os
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import pandas as pd
import evaluate_plot as eval_plot
import batchify as batchify
import signal_split as icml_workshop
from sklearn.metrics import precision_recall_fscore_support

# Set random seeds
#import random
#import numpy as np
#random.seed(1)
#np.random.seed(1)
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--data', required=True, help='Training Data Filename')
arg_parser.add_argument('--use_second_attention', required=True, help='True or False')
arg_parser.add_argument('--hidden_dim_slow', help='Size of output representation of Slow block. To be used when lstm_output_type is "different"')
arg_parser.add_argument('--hidden_dim_moderate', help='Size of output representation of Moderate block. To be used when lstm_output_type is "different"')
arg_parser.add_argument('--hidden_dim_fast', help='Size of output representation of Fast block. To be used when lstm_output_type is "different"')

arg_parser.add_argument('--model_name', help='Learning Rate for Training')
arg_parser.add_argument('--layers', help='Number of layers for the LSTM')
arg_parser.add_argument('--lr', help='Learning Rate for Training')
arg_parser.add_argument('--dropout', help='Dropout value')
arg_parser.add_argument('--epochs', help='Number of training Epochs')
arg_parser.add_argument('--opt', help='Optimization Algorithm; Options are: "SGD", "adam"')
arg_parser.add_argument('--lstm_output_type', help='Options: "same", "different"')
arg_parser.add_argument('--lstm_hidden_dim', help='Only used if lstm_output_type is same; Val: 120')
arg_parser.set_defaults(lr=0.0001, opt='SGD', dropout=0.9, epochs=200, model_name='default_model', layers=2, lstm_output_type='different', lstm_hidden_dim=120)

args, leftovers = arg_parser.parse_known_args()
print (args)

if args.use_second_attention == 'True':
    use_second_attention = True
elif args.use_second_attention == 'False':
    use_second_attention = False

# Process arguments
filename = args.data
datapath = os.path.join("..", "..", "..", "Data","signal_splits", "Physionet_Easy", filename)
learning_rate = float(args.lr)
model_name=args.model_name
# model_name = filename[:-4] + '_' + opt + '_LR-' + str(learning_rate) + '_hidden_dim-' + str(lstm_hidden_dim) + '_output_type-' + str(lstm_output_type) + '_second_attention-' + str(use_second_attention) + '_'


# 1 starts the process on GPU-0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.__version__

# Load Data
data = pickle.load(open(datapath,'rb'))
train_data = data[0]
val_data = data[1]
test_data = data[2]

print ("==x=="*20)
print ("Data Statistics")
print ("Train Data: "+str(len(train_data)))
print ("Val Data: "+str(len(val_data)))
print ("Test Data: "+str(len(test_data)))
print ("==x=="*20)


# This is just for testing
#train_data = train_data[:10]
#val_data = val_data[:10]
#test_data = test_data[:10]

params = {'bilstm_flag':True,
        'dropout' : float(args.dropout),
        'tagset_size' : 2,
        'attn_category' : 'dot',
        'layers' : int(args.layers),
        'num_features' : 37,
        'hidden_dim_slow': int(args.hidden_dim_slow),
        'hidden_dim_moderate': int(args.hidden_dim_moderate),
        'hidden_dim_fast': int(args.hidden_dim_fast),
        'lstm_output_type': args.lstm_output_type,
        'use_second_attention': use_second_attention,
        'lstm_hidden_dim': int(args.lstm_hidden_dim),
        'batch_size':1,
        'same_device':True,
        'same_feat_other_device':False,
        'model_name':model_name,
        'fast_features_indexes': [15, 34, 31, 11],
        'moderate_features_indexes': [14, 8, 5, 13, 16, 27, 24, 20, 35, 30, 9, 18, 12, 23, 21, 22, 36, 37, 10, 26, 25, 19],
        'slow_features_indexes': [17, 28, 29, 6, 3, 2, 1, 4, 33, 7, 32]}

save_flag = True
savepath = os.path.join("..", "..","..")
pickle.dump(params, open(os.path.join(savepath,"Models/config_")+params['model_name']+'.pt','wb'))

dict_df_prf_mod = {}
model = icml_workshop.Model(params).cuda()
loss_function = nn.NLLLoss(weight=torch.cuda.FloatTensor([1.0, 6.2]))

if args.opt == 'adam':
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005)
elif args.opt == 'SGD':
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00000002)

epochs = int(args.epochs)
model_name = params['model_name']
total_loss = 0
preds_train = []
actual_train = []
 
for iter_ in range(epochs):
    print ("=#="*5+str(iter_)+"=#="*5)

    for datapoint in tqdm(train_data):
        model.zero_grad()

        # Get prediction
        tag_scores = model(datapoint)   # Datapoint is tuple with: (id, slow_data, moderate_data, fast_data, label)

        # Get label
        curr_labels = [datapoint[4]]
        curr_labels = [batchify.label_mapping[x] for x in curr_labels]
        actual_train+=curr_labels
        curr_labels = torch.cuda.LongTensor(curr_labels)
        curr_labels = autograd.Variable(curr_labels)
        
        loss = loss_function(tag_scores, curr_labels)
        total_loss+=loss.item()

        _, ind_ = torch.max(tag_scores, dim=1)
        preds_train+=ind_.tolist()
       
        loss.backward()
        optimizer.step()


    df_tr = pd.DataFrame(list(precision_recall_fscore_support(actual_train, preds_train,
                                                              labels = [0,1])),
                                                             columns = [0,1])
    df_tr.index = ['Precision','Recall','F-score','Count']
    prf_tr = precision_recall_fscore_support(actual_train, preds_train, average='weighted')
    prf_test, df_test = eval_plot.evaluate_mlhc(model, test_data)
    prf_val, df_val = eval_plot.evaluate_mlhc(model, val_data)

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
        torch.save(model, savepath+"/Models/"+model_name+str(iter_)+'.pt')
        pickle.dump(dict_df_prf_mod, open(savepath+'/Results/dict_prf_'+model_name+str(iter_)+'.pkl','wb'))
        # eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', savepath+'/Plots/'+model_name+str(iter_)+'.png', 0, iter_+1, model_name)

eval_plot.plot_graphs(dict_df_prf_mod, 'F-score', savepath+'/Plots/'+model_name+str(iter_)+'.png',0, iter_, model_name)

# f.close()
