import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

import cPickle as pickle

label_mapping = {0: 0, 1: 1}


def evaluate_dbm(model, value_dict, batch_size):
    preds = []
    actual = []

    data = value_dict['data']
    labels = value_dict['label']
    lens = value_dict['lens']
    size = len(labels)
    # print(size)

    for i in tqdm(range(size)):
        
        label_scores = model([data[i]])
        _, ind = torch.max(label_scores, dim=1)
        preds += ind.tolist()
        # print(preds)
        # print('#' * 10)

    labels = [label_mapping[x] for x in labels]
    actual += labels

    df_ = pd.DataFrame(list(precision_recall_fscore_support(actual, preds, labels = [0,1])),
                       columns = [0,1])
    df_.index = ['Precision','Recall','F-score','Count']
    return precision_recall_fscore_support(actual, preds, average='weighted'), df_


def plot_graphs(data, metric, fig_name, start_epoch, end_epoch, title):
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    # Train
    train_metric_vals, _, _ = get_metric_values(data, 'train', metric, start_epoch, end_epoch)
    ax1.plot(train_metric_vals, color='red',label='Train')
    # Val
    val_metric_vals, _, max_key = get_metric_values(data, 'val', metric, start_epoch, end_epoch)
    ax1.plot(val_metric_vals, color='blue',label='Val')
    # Test
    test_metric_vals, _, _ = get_metric_values(data, 'test', metric, start_epoch, end_epoch)
    ax1.plot(test_metric_vals, color='green',label='Test')

    fig.set_size_inches(11.5, 6.5)
    
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()
    # print "=="*5+max_key+"=="*4
    # print "TRAIN: "+str(get_prf_metrics(data[max_key],'train'))
    # print "VAL: "+str(get_prf_metrics(data[max_key],'val'))
    # print "TEST: "+str(get_prf_metrics(data[max_key],'test'))
    # print "=="*4+" Detailed Results "+"=="*4
    # print data[max_key]
    
def get_prf_metrics(data, key):
    if(key=='train'):
        start_ind = 0
        end_ind = 2
    elif(key=='val'):
        start_ind = 2
        end_ind = 4
    elif(key=='test'):
        start_ind = 4
        end_ind = 6
    f_score = data.loc['F-score'][start_ind:end_ind]
    precision = data.loc['Precision'][start_ind:end_ind]
    recall = data.loc['Recall'][start_ind:end_ind]
    count_ = data.loc['Count'][start_ind:end_ind]
    # F-score
    f_score = f_score*count_
    f_score = sum(f_score)/sum(count_)
    # Precision
    precision = precision*count_
    precision = sum(precision)/sum(count_)
    #Recall
    recall = recall*count_
    recall = sum(recall)/sum(count_)
    return (precision, recall, f_score)

def get_metric_values(data, key, metric, start_epoch, end_epoch):
    val_f_score = []
    max_ = 0.0
    max_ind = 0
    max_key = ''
    for ind in range(start_epoch, end_epoch):
        key_ = 'Epoch'+str(ind)
        if(key=='train'):
            start_ind = 0
            end_ind = 2
        elif(key=='val'):
            start_ind = 2
            end_ind = 4
        elif(key=='test'):
            start_ind = 4
            end_ind = 6
        # metrics: Precision, Recall and F-score
        f_score = data[key_].loc[metric][start_ind:end_ind]
        count_ = data[key_].loc['Count'][start_ind:end_ind]
        
        weighted_sum = f_score*count_
        weighted_sum = sum(weighted_sum)/sum(count_)
        if(weighted_sum>max_):
            max_ = weighted_sum
            max_ind = ind
            max_key = key_
        val_f_score.append(weighted_sum)
    return val_f_score, max_, max_key
