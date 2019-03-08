import csv

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

import cPickle as pickle

from imputation import *


# path_pre = '/home/sidongzhang/code/fl'
path_pre = '../..'


def toy_corr():
    a = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]])
    print(a)
    print(np.corrcoef(a))


def interval_corr(interval_len):
    data = pickle.load(open(path_pre + '/data/final_osaka_data_avg_60_flags.pkl', 'rb'))
    imputation = Imputation({'tagset_size': 3, 'num_features': 24})

    imputed = imputation.forward(data['data'], 'T0_ID400836_SlopeDown.csv')

    for i in range(imputed.shape[1] % interval_len):
        if i < imputed.shape[1] % interval_len-1:
            local_value_corr = np.corrcoef(imputed[:, i * interval_len: (i+1) * interval_len, 0])
        else:
            local_value_corr = np.corrcoef(imputed[:, i * interval_len:, 0])
        print('######'*20)
        print(local_value_corr)
    print(imputed.shape)


if __name__ == '__main__':
    # toy_corr()
    interval_corr(10)