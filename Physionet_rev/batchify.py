import torch
import torch.autograd as autograd
import numpy as np

label_mapping = {'SlopeDown.csv':0, 
                 'SlopeUp.csv':1, 
                 'Walk1.csv':2, 
                 'Walk2.csv':2}
# label_mapping = {0:0,1:1}   
# num_of_features = 8

def prepare_batch(dict_data, ids, num_of_features, feature_ind, label_ind):
    labels = []
    for each_id in ids:
        t_label = label_mapping[dict_data[each_id][label_ind]]
        labels.append(t_label)
    features = []
    max_len = 0
    actual_lens = []
    for each_id in ids:
        t_features = dict_data[each_id][feature_ind]
        features.append(t_features)
        if(len(t_features)>max_len):
            max_len = len(t_features)
        actual_lens.append(len(t_features))
    
    for ind in range(len(features)):
        features[ind] = features[ind]+[[0 for x in range(num_of_features)] for y in range(max_len-len(features[ind]))]
    
    sorted_inds = np.argsort(actual_lens)
    sorted_inds = sorted_inds[::-1]
    
    sorted_lens = []
    sorted_features = []
    sorted_labels = []
    ind_cnt = 0
    for ind in sorted_inds:
        sorted_lens.append(actual_lens[ind])
        sorted_features.append(features[ind])
        sorted_labels.append(labels[ind])
    
    sorted_features = torch.cuda.FloatTensor(sorted_features)
    sorted_features = autograd.Variable(sorted_features)
    
    sorted_labels = torch.cuda.LongTensor(sorted_labels)
    sorted_labels = autograd.Variable(sorted_labels)
    
    return sorted_features, sorted_labels, sorted_lens