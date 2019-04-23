import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

import cPickle as pickle

import tools

is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0")

label_mapping = {0: 0, 1: 1}

path_pre = '../..'


class CVL(nn.Module):
    def __init__(self, params):
        super(CVL, self).__init__()

        self.bilstm_flag = params['bilstm_flag']
        self.dropout = params['dropout']
        self.layers = params['layers']
        self.tagset_size = params['tagset_size']
        self.attn_category = params['attn_category']
        self.num_features = params['num_features']
        # self.max_len = params['max_len']
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']

        self.LL = nn.Linear(self.num_features, self.input_dim)

        if (self.bilstm_flag):
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim / 2, num_layers=self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layers,
                                bidirectional=False, batch_first=True, dropout=self.dropout)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        if (self.attn_category == 'dot'):
            print "Dot Attention is being used!"
            self.attn = DotAttentionLayer(self.hidden_dim).cuda()

    def init_hidden(self, batch_size):
        # num_layes, minibatch size, hidden_dim
        if self.bilstm_flag:
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers * 2,
                                                             batch_size,
                                                             self.hidden_dim / 2).fill_(0)),
                    autograd.Variable(torch.cuda.FloatTensor(self.layers * 2,
                                                             batch_size,
                                                             self.hidden_dim / 2).fill_(0)))
        else:
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                             batch_size,
                                                             self.hidden_dim).fill_(0)),
                    autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                             batch_size,
                                                             self.hidden_dim).fill_(0)))

    def forward(self, data):
        """
        :param data: list: batch data: [B, 37, T]
               lens: list of int. Has length B. Saves the actual length for every data point.
        :return: tag_score: [B, 2]
        """
        batch_size = len(data)
        T = len(data[0][0])
        input_ = torch.cuda.FloatTensor(data)

        # transpose the data into shape (B, T, 37)
        # input_ = torch.transpose(input_, 1, 2)
        input_ = autograd.Variable(input_)

        # pass the original data to ll
        densed_data = self.LL(input_)  # (B, T, input_dim)

        # vertical attn over clusters by bilstm-attn
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(densed_data, self.hidden)  # (B, T, hidden_dim)

        tools.validate_no_nans_in_tensor(lstm_out)

        if self.attn_category == 'dot':
            pad_attn = self.attn((lstm_out, torch.cuda.LongTensor([T for _ in range(batch_size)])))
            #             print pad_attn
            #             print "TAG SPACE"
            tag_space = self.hidden2tag(pad_attn)  # (B, 2)
        #             print tag_space
        else:
            tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_score = F.log_softmax(tag_space, dim=1)  # (B, 2)

        return tag_score


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        torch.nn.init.xavier_uniform(self.W.weight)

    def forward(self, input):
        """
        input: a tuple:
        tuple[0]: unpacked_padded_output: B x cluster_len x Max_len (T for short)
        tuple[1]: a list of B integers (actual time_seq len))
        """
        inputs, lengths = input  # (B, Cl, ML); (B)
        batch_size, T,  _ = inputs.size()

        flat_input = inputs.contiguous().view(-1, self.hidden_size)  # (B * CL, ML)
        logits = self.W(flat_input).view(batch_size, T)  # (B * Cl, ML) ->  (B, CL)

        # computing mask
        # idxes = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len)).unsqueeze(0).cuda()  # (1, T)
        # masked = []
        # print('idxes, ', idxes)
        # print('lengths, ', lengths)
        #
        # for l in lengths:
        #     print(torch.cuda.LongTensor(l).unsqueeze(1))
        #     one_batch = autograd.Variable(torch.cuda.ByteTensor(idxes < torch.cuda.LongTensor(l)))
        #     print(one_batch)
        #     masked.append(one_batch)  # (1, T)
        # # mask = autograd.Variable(torch.cuda.ByteTensor(idxes<lengths.unsqueeze(1)))
        # mask = torch.cat(masked, dim=0)  # (B, T)
        #
        # # mask the padded part to -inf so they contribute 0 in the softmax
        # # for the cut part we just cut them off, their masks will be all 1
        # logits[~mask] = float('-inf')

        alphas = F.softmax(logits, dim=1)  # (B, Cl)

        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)  # (B, 1, Cl) dot (B, Cl, ML) -> (B, 1, ML) -> (B, ML)
        return output  # (batch_size, max_len)
