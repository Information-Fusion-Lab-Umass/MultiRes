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

import imputation
import cluster_vertical_lstm as cvl
import evaluate_plot as eval_plot


if __name__ == '__main__':
    params = {'bilstm_flag': True,
              'dropout': 0.9,
              'layers': 1,
              'tagset_size': 2,
              'attn_category': 'dot',
              'num_features': 37,
              'input_dim': 10,
              'hidden_dim': 50,
              'same_device': False,
              'same_feat_other_device': False,
              'model_name': 'CVL-Phy',
              'cluster_path': '/home/sidongzhang/code/fl/data/dummy_cluster.pkl'}
    imputated = pickle.load(open('/home/sidongzhang/code/fl/data/imputed_physionet.pkl', 'rb'))
    test = imputated['test']
    val = imputated['val']

    model = cvl.CVL(params).cuda()
    prf_test, df_test = eval_plot.evaluate_dbm(model, test)