import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

import cPickle as pickle

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
        self.max_len = params['max_len']
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.cluster = pickle.load(open(params['cluster_path'], 'rb'))  # [[cluster one feats], [cluster 2 feats], ...]

        # vertical dot attns: [B, one_cluster_feats, T] -> [B, 1, T]
        # initial strategy: xavier
        if self.attn_category == 'dot':
            print "Dot Attention is being used!"
            self.inner_attns = nn.ModuleList([])
            for _ in self.cluster:
                self.inner_attns.append(DotAttentionLayer(self.max_len).cuda())

        # dense layer: [B, T, num_clusters] -> [B, T, input_dims]
        # initial strategy: xavier
        self.LL = nn.Linear(len(self.cluster), self.input_dim)
        torch.nn.init.xavier_uniform(self.LL.weight)

        # Bilistm: [B, T, input_dims] -> [B, T, hidden_dims]
        # initial strategy: zero hiddens
        if self.bilstm_flag:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim / 2, num_layers=self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layers,
                                bidirectional=False, batch_first=True, dropout=self.dropout)

        # Final dot attention: [B, T, hidden_dims] -> [B, hidden_dims]
        # initial strategy: xavier
        if self.attn_category == 'dot':
            print "Dot Attention is being used!"
            self.attn = DotAttentionLayer(self.hidden_dim).cuda()

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

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
        :param data: list: batch data: [B, 37, T, 4]
               lens: list of int. Has length B. Saves the actual length for every data point.
        :return: tag_score: [B, 2]
        """

        features = self.vertical_attn(data)  # (B, cluster_num, T)
        features = torch.transpose(features, 1, 2)
        #         print "FEAT"
        #         print features
        features = self.LL(features)  # (B, cluster_num, input_dim)
        #         print "Features"
        #         print features
        #         print features.shape

        batch_size = len(data)

        packed = pack_padded_sequence(features,
                                      torch.cuda.LongTensor([len(self.cluster) for _ in range(batch_size)]),
                                      batch_first=True)

        self.hidden = self.init_hidden(batch_size)

        packed_output, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]  # (B, cluster_num, hidden_dim)

        if self.attn_category == 'dot':
            pad_attn = self.attn((lstm_out, torch.cuda.LongTensor([len(self.cluster) for _ in range(batch_size)])))  # (B, cluster_num, hidden_dim) -> B,H
            #             print pad_attn
            #             print "TAG SPACE"
            tag_space = self.hidden2tag(pad_attn)  # (B, 2)
        #             print tag_space
        else:
            tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_score = F.log_softmax(tag_space, dim=1)  # (B, 2)

        return tag_score

    def vertical_attn(self, data):
        """
        :param data: list: (B, 37, time_seq, 4)
                     time_seq here is a fixed num. We preprocess the data so that for every data point it has the same
                     time_seq. We call it max_time_len or T for short
               lens: list of int. Has length B. Saves the actual length for every data point.

        :return: imputation_attn: pytorch tensor: (B, cluster_num, T)
        """

        B = len(data)
        F = len(data[0])
        T = len(data[0][0])

        # tbm
        raw_tbm = []  # (B, F, T)
        for b in range(B):
            one_batch = data[b]
            local_b = []
            for f in range(F):
                one_feat = one_batch[f]
                local_f = []
                for t in range(T):
                    curr_feat = one_feat[t]
                    # if(curr_feat[2]==1):
                    # TBM parameters
                    beta_val = 0.75
                    tau_val = 2
                    h_val = 0.4
                    m_t = curr_feat[2]
                    x_l = curr_feat[0]
                    x_m = curr_feat[1]
                    curr_delta_t = curr_feat[3]
                    b_t_dash = np.exp(-beta_val * curr_delta_t * 1.0 / tau_val)
                    if b_t_dash > h_val:
                        b_t = 1
                    else:
                        b_t = 0
                    feat_val = (1 - m_t) * x_l + m_t * (b_t * x_l + (1 - b_t) * x_m)
                    local_f.append(feat_val)
                local_b.append(local_f)
            raw_tbm.append(local_b)

        raw_tbm = np.array(raw_tbm)  # (B, F, T)
        # cluster attention
        stack_data = []  # a list of (B, T), list len = num of clusters
        for cid, c in enumerate(self.cluster):
            local_cluster = []  # cluster_len (B, T) tensor lists
            for b in range(B):
                for f in c:
                    local_cluster.append(torch.from_numpy(raw_tbm[:, f, :]).float().to(device))
            stacked_local = torch.stack(local_cluster, dim=1)  # (B, cluster_len, T)
            attn = self.inner_attns[cid]((stacked_local,  torch.cuda.LongTensor([len(c) for _ in range(B)])))  # (B, T)
            stack_data.append(attn)
        stack_data = torch.stack(stack_data, dim=1)  # (B, cluster_num, T)
        stack_data = autograd.Variable(stack_data)
        return stack_data


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
