import cPickle as pickle

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

is_cuda = torch.cuda.is_available()

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
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.cluster = pickle.load(open(params['cluster_path'], 'rb'))  # [[cluster one feats], [cluster 2 feats], ...]

        # self.LL = nn.Linear(self.num_features, self.input_dim)
        self.LL = nn.Linear(len(self.cluster), self.input_dim)

        if self.attn_category == 'dot':
            print "Dot Attention is being used!"
            self.inner_attns = nn.ModuleList([])
            for _ in self.cluster:
                # self.inner_attns.append(DotAttentionLayer(4).cuda())
                self.inner_attns.append(DotAttentionLayer(1).cuda())
            # self.outer_attn = DotAttentionLayer(4).cuda()

        if self.bilstm_flag:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim / 2, num_layers=self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layers,
                                bidirectional=False, batch_first=True, dropout=self.dropout)

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
        :param data: list: one batch data: [time_seq0, 37, 4]
        :return: tag_score: [1, 2]
        """
        features = self.vertical_attn(data)  # (1, time_seq, cluster_num)    # (B, 3, T)
        #         print "FEAT"
        #         print features
        # features = self.LL(features)            # (1, time_seq, input_dim)  # (B, 3, input_dim)
        #         print "Features"
        #         print features
        #         print features.shape
        lenghts = [features.shape[1]]
        lengths = torch.cuda.LongTensor(lenghts)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first=True)

        batch_size = 1
        self.hidden = self.init_hidden(batch_size)

        packed_output, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]  # Bx3xH

        if self.attn_category == 'dot':
            pad_attn = self.attn((lstm_out, torch.cuda.LongTensor(lengths)))  # (B,3,H) -> B,H
            #             print pad_attn
            #             print "TAG SPACE"
            tag_space = self.hidden2tag(pad_attn)  # (B, 2)
        #             print tag_space
        else:
            tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score

    def vertical_attn(self, data):
        """
        :param data: list: (B, 37, time_seq, 4)
                     time_seq here is a fixed num. We preprocess the data so that for every datapoint it has the same
                     time_seq. We call it max_time_len or T for short
        :return: imputation_attn: pytorch tensor: (1, time_seq, cluster_num)
        How to do this: We first visit all time_seq points,
                        in every time seq we have a [37, 4] tensor.
                        We use tbm to transfer it to a [37, 1] tensor and stack all those tensors to get a
                        [time_seq, 37, 1] tensor. We then apply cluster attention and get a [time_seq, cluster_num, 1]
                        tensor. Finally we squeeze the 3rd dim and unsqueeze the 1st dim to get a
                        [1, time_seq, cluster_num] tensor.

                        # We expand it to [1, 37, 4] (1 for one batch).
                        # On each time point, we then do cluster level attention to transfer it
                        # to [1, cluster_ num, 4] array. We use tbm
                        # on this [1, cluster_num, 4] tensor and get a [1, cluster_num] output.

                        # We then do 2 cluster attention to transfer it
                        # to [1, 4] array and squeeze it to [4] array. We concat all those [4] array and get a
                        # [time_seq, 4] tensor. We use tbm
                        # on this [time_seq, 4] tensor and get a [time_seq, 1] output.
        """

        # #Start
        # BxFxTx4
        # #TBM
        # BxFxTx1
        #
        # #Clustering
        # BxF1Xt
        # BxF2Xt
        # BxF3Xt
        #
        # #Pass to DotAttention Layer
        # Bxt
        # Bxt
        # Bxt
        #
        # #Stack to get
        # Bx3xt

        # Pass to bilstm
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

        # cluster attention
        stack_data = []  # a list of (B, T), list len = num of clusters
        for cid, c in enumerate(self.cluster):
            local_cluster = []
            for f in c:
                local_cluster.append(torch.cuda.FloatTensor(raw_tbm[:, ]))

        # time_seq = len(data)
        # raw_tbm = []  # (time_seq, num_features, 1)
        # for t in range(time_seq):
        #     one_time = np.array(data[t])  # (37, 4)
        #
        #     local_tbm = []  # (37, 1)
        #
        #     for f in range(self.num_features):
        #         curr_feat = one_time[f]
        #         # if(curr_feat[2]==1):
        #         # TBM parameters
        #         beta_val = 0.75
        #         tau_val = 2
        #         h_val = 0.4
        #         m_t = curr_feat[2]
        #         x_l = curr_feat[0]
        #         x_m = curr_feat[1]
        #         curr_delta_t = curr_feat[3]
        #         b_t_dash = np.exp(-beta_val * curr_delta_t * 1.0 / tau_val)
        #         if b_t_dash > h_val:
        #             b_t = 1
        #         else:
        #             b_t = 0
        #         feat_val = (1 - m_t) * x_l + m_t * (b_t * x_l + (1 - b_t) * x_m)
        #         local_tbm.append([feat_val])
        #     raw_tbm.append(local_tbm)
        #
        # raw_tbm = list((np.array(raw_tbm).transpose((1, 0, 2))).tolist())  # (num_features, time_seq, 1)
        # attn_clustered_tbm = []
        # for cid, c in enumerate(self.cluster):
        #     local = []
        #     for f in c:
        #         local.append(torch.cuda.FloatTensor(raw_tbm[f]))
        #     local = torch.stack(local, dim=1)  # (time_seq, cluster_len, 1)
        #     attn = self.inner_attns[cid]((local, torch.cuda.LongTensor([len(c)])))  # (time_seq, 1)
        #     attn_clustered_tbm.append(attn)
        # attn_clustered_tbm = torch.cat(tuple(attn_clustered_tbm), 1)  # (time_seq, cluster_num)
        #
        # attn_clustered_tbm.unsqueeze_(0)  # (1, time_seq, cluster_num)
        # attn_clustered_tbm = autograd.Variable(attn_clustered_tbm)
        #
        # return attn_clustered_tbm

        # cvlised = []  # (time_seq, cluster_num)

        # for t in range(time_seq):
        #     one_time = np.array([data[t]])  # (1, 37, 4)
        #     # print(one_time.shape)
        #     clustered = []  # (cluster_num)
        #
        #     for cidx, c in enumerate(self.cluster):
        #         array = []
        #         for f in c:
        #             array.append(one_time[:, f, :])
        #         stacked = np.stack(array, axis=1)
        #         attended = self.inner_attns[cidx](stacked)  # (1, 4)
        #
        #         # tbm
        #         curr_feat = attended[0]
        #         # if(curr_feat[2]==1):
        #         # TBM parameters
        #         beta_val = 0.75
        #         tau_val = 2
        #         h_val = 0.4
        #         m_t = curr_feat[2]
        #         x_l = curr_feat[0]
        #         x_m = curr_feat[1]
        #         curr_delta_t = curr_feat[3]
        #         b_t_dash = np.exp(-beta_val * curr_delta_t * 1.0 / tau_val)
        #         if b_t_dash > h_val:
        #             b_t = 1
        #         else:
        #             b_t = 0
        #         feat_val = (1 - m_t) * x_l + m_t * (b_t * x_l + (1 - b_t) * x_m)
        #         # print(feat_val)
        #         # print(feat_val.shape)
        #
        #         clustered.append(feat_val)
        #
        #     cvlised.append(clustered)
        #
        # cvlised = torch.cuda.FloatTensor(cvlised)
        # cvlised = cvlised.unsqueeze(0)  # (1, time_seq, cluster_num)
        # cvlised = autograd.Variable(cvlised)
        # return cvlised

        # final_feat_list = []
        # for t in range(time_seq):
        #     curr_feat = cvlised[t]
        #     # if(curr_feat[2]==1):
        #     # TBM parameters
        #     beta_val = 0.75
        #     tau_val = 2
        #     h_val = 0.4
        #     m_t = curr_feat[2]
        #     x_l = curr_feat[0]
        #     x_m = curr_feat[1]
        #     curr_delta_t = curr_feat[3]
        #     b_t_dash = np.exp(-beta_val * curr_delta_t * 1.0 / tau_val)
        #     if b_t_dash > h_val:
        #         b_t = 1
        #     else:
        #         b_t = 0
        #     feat_val = (1 - m_t) * x_l + m_t * (b_t * x_l + (1 - b_t) * x_m)
        #     # print(feat_val)
        #     # print(feat_val.shape)
        #     final_feat_list.append(feat_val)
        #
        # final_feat_list = torch.cuda.FloatTensor(final_feat_list)
        #
        # final_feat_list = final_feat_list.unsqueeze(0)
        # final_feat_list = autograd.Variable(final_feat_list)
        #
        # return final_feat_list

        # for ind, each_flag in enumerate(feat_flag):
        #     final_feat_list = []
        #     for feat_ind in range(num_features):
        #         curr_feat = imputed_input[feat_ind][ind]
        #         # if(curr_feat[2]==1):
        #         # TBM parameters
        #         beta_val = 0.75
        #         tau_val = 2
        #         h_val = 0.4
        #         m_t = curr_feat[2]
        #         x_l = curr_feat[0]
        #         x_m = curr_feat[1]
        #         curr_delta_t = curr_feat[3]
        #         b_t_dash = np.exp(-beta_val * curr_delta_t * 1.0 / tau_val)
        #         if (b_t_dash > h_val):
        #             b_t = 1
        #         else:
        #             b_t = 0
        #         feat_val = (1 - m_t) * x_l + m_t * (b_t * x_l + (1 - b_t) * x_m)
        #         # print(feat_val)
        #         # print(feat_val.shape)
        #         final_feat_list.append(feat_val)
        #     all_features.append(final_feat_list)
        # all_features = torch.cuda.FloatTensor(all_features)
        #
        # all_features = all_features.unsqueeze(0)
        # all_features = autograd.Variable(all_features)
        # # print(all_features)
        # # print(all_features.shape)
        # # all_features (batch, seq_len, 37)
        # return all_features


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        torch.nn.init.xavier_uniform(self.W.weight)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: time_seq)
        """
        inputs, lengths = input  # (B, T, H)
        batch_size, max_len, _ = inputs.size()

        flat_input = inputs.contiguous().view(-1, self.hidden_size)  # (B * T, H)
        logits = self.W(flat_input).view(batch_size, max_len)  # (B * T, 1) ->  (B, T)

        print('dot attn logits:', logits)

        # computing mask
        if is_cuda:
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()  # (1, T)
        else:
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)

        mask = autograd.Variable(torch.cuda.ByteTensor(idxes < lengths.unsqueeze(1)))  # (1, T)
        mask = torch.cat([mask] * batch_size, dim=0)
        print('dot attn mask: ', mask)
        logits[~mask] = float('-inf')
        alphas = F.softmax(logits, dim=1)  # (B, T)

        print('dot attn alpha: ', alphas)

        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)  # (B, 1, T) dot (B, T, H) -> (B, 1, H) -> (B, H)
        return output  # (batch_size, hidden_size)
