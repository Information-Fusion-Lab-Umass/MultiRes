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

label_mapping = {0: 0, 1: 1}

path_pre = '../..'

"""
params: tagset_size(3), num_features(24), imputation_layer_dim_op(20), feats_provided_flag
This is for one single record (corresponding to one action label)

Change the dense layer into one small linear layer and put the potential support features into a dot attention net

The selection of support features can be done by Opposite-Missing strategy (or correlation, but this is not promising)
"""


class Imputation(nn.Module):
    def __init__(self, params):
        super(Imputation, self).__init__()
        self.bilstm_flag = params['bilstm_flag']
        self.tagset_size = params['tagset_size']
        self.num_features = params['num_features']

        self.selected_feats = params['selected_feats']
        self.attn_category = params['attn_category']
        self.layers = params['layers']
        self.dropout = params['dropout']

        self.imputation_layer_dim_in = 4
        self.imputation_layer_dim_op = params['imputation_layer_dim_op']

        self.input_dim = self.num_features * self.imputation_layer_dim_op * 2

        self.hidden_dim = 2 * self.input_dim
        # if is_cuda:
        #     self.imputation_layer_in = nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op).cuda()
        # else:
        #     self.imputation_layer_in = nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op)

        # self.imputation_layer_dim_op = params['imputation_layer_dim_op']
        # self.imputation_layer_dim_in = (self.selected_feats + 1) * 4

        # self.dict_selected_feats = {}
        # if params['feats_provided_flag']:
        #     print "Oooh! the physio_features are provided!"
        self.dict_selected_feats = pickle.load(open(params['path_selected_feats'], 'rb'))
        # for each_ind in range(self.num_features):
        #     self.dict_selected_feats[each_ind] = [each_ind] + self.dict_selected_feats[each_ind]
        self.dense_layers = nn.ModuleList([])

        for i in range(self.num_features):
            if is_cuda:
                self.dense_layers.append(nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op).cuda())
            else:
                self.dense_layers.append(nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op))

        self.init_weight()

        if self.attn_category == 'dot':
            print "Dot Attention is being used!"
            self.sp_attns = nn.ModuleList([])
            # for all 37 features we allocate a attention layer to their potential support feats
            for i in range(self.num_features):
                if is_cuda:
                    self.sp_attns.append(DotAttentionLayer(self.imputation_layer_dim_op).cuda())
                else:
                    self.sp_attns.append(DotAttentionLayer(self.imputation_layer_dim_op))

        if self.bilstm_flag:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim / 2, num_layers=self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layers,
                                bidirectional=False, batch_first=True, dropout=self.dropout)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        if self.attn_category == 'dot':
            print "Dot Attention is being used!"
            if is_cuda:
                self.attn = DotAttentionLayer(self.hidden_dim).cuda()
            else:
                self.attn = DotAttentionLayer(self.hidden_dim)

    def init_weight(self):
        for layer in self.dense_layers:
            torch.nn.init.xavier_uniform(layer.weight)

    def init_hidden(self, batch_size):
        if self.bilstm_flag:
            if is_cuda:
                return (autograd.Variable(torch.cuda.FloatTensor(self.layers * 2,
                                                                 batch_size,
                                                                 self.hidden_dim / 2).fill_(0)),
                        autograd.Variable(torch.cuda.FloatTensor(self.layers * 2,
                                                                 batch_size,
                                                                 self.hidden_dim / 2).fill_(0)))
            else:
                return (autograd.Variable(torch.FloatTensor(self.layers * 2,
                                                                 batch_size,
                                                                 self.hidden_dim / 2).fill_(0)),
                        autograd.Variable(torch.FloatTensor(self.layers * 2,
                                                                 batch_size,
                                                                 self.hidden_dim / 2).fill_(0)))
        else:
            if is_cuda:
                return (autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                                 batch_size,
                                                                 self.hidden_dim).fill_(0)),
                        autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                                 batch_size,
                                                                 self.hidden_dim).fill_(0)))
            else:
                return (autograd.Variable(torch.FloatTensor(self.layers,
                                                                 batch_size,
                                                                 self.hidden_dim).fill_(0)),
                        autograd.Variable(torch.FloatTensor(self.layers,
                                                                 batch_size,
                                                                 self.hidden_dim).fill_(0)))

    def forward(self, input_):
        # input_ should be a dict that contains information about one data point (one batch)
        # this datapoint has structure: {feat_id : [seq_len, 4] matrix}
        features = self.get_imputed_feats(input_, self.dict_selected_feats)
        # print(features)
        lengths = [features.shape[1]]
        if is_cuda:
            lengths = torch.cuda.LongTensor(lengths)
        else:
            lengths = torch.LongTensor(lengths)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first=True)

        batch_size = 1
        self.hidden = self.init_hidden(batch_size)

        packed_output, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]

        if self.attn_category == 'dot':
            if is_cuda:
                pad_attn = self.attn((lstm_out, torch.cuda.LongTensor(lengths)))
            else:
                pad_attn = self.attn((lstm_out, torch.LongTensor(lengths)))
            tag_space = self.hidden2tag(pad_attn)
        else:
            tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score

    def padding(self, data):
        '''
        :param data:
            A dict of [[4 dim], [4 dim], ..., [4 dim]], seq_len unfixed. The dict contains 24 keys,
            representing 24 time series data from 24 sensors.
        :returns:
            a padded 24 * max_seq_len * 4 tensor
            and its corresponding actual seq_lens
        '''
        seq_lens = []
        padded = []
        for k in data.keys():
            seq_lens.append(len(data[k]))
            padded.append(data[k])

        sorted_idx = list(np.argsort(seq_lens))
        sorted_idx.reverse()

        sorted_lens = torch.autograd.Variable(torch.tensor([seq_lens[i] for i in sorted_idx]))

        sorted_padded = [torch.autograd.Variable(torch.tensor(padded[i])) for i in sorted_idx]
        sorted_padded = torch.nn.utils.rnn.pad_sequence(sorted_padded, batch_first=True)

        return sorted_padded, sorted_lens

    def get_imputed_feats(self, input_, dict_selected_feats):
        # input_ should be a dict that contains information about one data point (one batch)
        # this datapoint has structure: {feat_id : [seq_len, 4] matrix}

        # feats: actual values
        # flags: indicate whether the corresponding record in feats is missing; 1: missing; 0: not missing;

        num_features = self.num_features
        all_features = []
        # input_ = {}

        # we again visit the 24 feats and append their support feats
        for feat_ind in range(num_features):
            final_feat_list = []
            for ind, imputed in enumerate(input_[feat_ind]):
                # main feats -> dense layer -> main dense feats
                if is_cuda:
                    main_feats = torch.cuda.FloatTensor(input_[feat_ind][ind])
                else:
                    main_feats = torch.FloatTensor(input_[feat_ind][ind])

                densed_feats = self.dense_layers[feat_ind](main_feats)

                # we visit all the historical data point for ith feat by ind
                imputation_feat = []
                for each_selected_feat in dict_selected_feats[feat_ind]:
                    if is_cuda:
                        imputation_feat.append(self.dense_layers[each_selected_feat](torch.cuda.FloatTensor(input_[each_selected_feat][ind])))
                    else:
                        imputation_feat.append(self.dense_layers[each_selected_feat](
                            torch.FloatTensor(input_[each_selected_feat][ind])))

                # shape: (support_feats, 4) -> (support_feats, imputation_dim_op)

                imputation_feat = torch.stack(imputation_feat)
                imputation_feat.unsqueeze_(0)  # (1, support_feats, imputation_dim_op)
                # print('imputation_shape: ', imputation_feat.shape)

                if is_cuda:
                    attn_imputation = self.sp_attns[feat_ind]((imputation_feat, torch.cuda.LongTensor([1])))  # (1, imputation_dim_op)
                else:
                    attn_imputation = self.sp_attns[feat_ind](
                        (imputation_feat, torch.LongTensor([1])))  # (1, imputation_dim_op)

                attn_imputation.squeeze_(0)  # (imputation_dim_op)

                '''
                Here we should determine how to combine the attn_imputation with the linear-went-through main features
                '''

                if is_cuda:
                    attn_imputation = torch.cat((torch.cuda.FloatTensor(densed_feats), attn_imputation))
                else:
                    attn_imputation = torch.cat((torch.FloatTensor(densed_feats), attn_imputation))
                    #  (support_feats+1) * 4
                    # print(attn_imputation.shape)
                # print(attn_imputation)
                # print(imputation_feat.shape)
                # f_ = self.attn((imputation_feat, torch.LongTensor([1])))
                # final_feat_list.append(imputation_feat)
                final_feat_list.append(attn_imputation)

            final_feat_list = torch.stack(final_feat_list)
            #  print(final_feat_list)
            #  print(final_feat_list.shape)
            all_features.append(final_feat_list)
        all_features = torch.cat(all_features, 1)
        all_features = all_features.unsqueeze(0)
        all_features = autograd.Variable(all_features)
        # print(all_features.shape)
        # print(all_features)
        return all_features  # (batch, seq_len,  num_features * 2 * imputation_dim_op)


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)
        torch.nn.init.xavier_uniform(self.W.weight)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = F.softmax(logits, dim=1)

        # computing mask
        if is_cuda:
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
        else:
            idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)

        mask = autograd.Variable((idxes<lengths.unsqueeze(1)).float())

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output


# if __name__ == '__main__':
#     params = {'bilstm_flag': True,
#               'dropout': 0.9,
#               'layers': 1,
#               'tagset_size': 3,
#               'attn_category': 'dot',
#               'num_features': 37,
#               'imputation_layer_dim_op': 20,
#               'selected_feats': 3,
#               'batch_size': 1,
#               'same_device': False,
#               'same_feat_other_device': False,
#               'model_name': 'physio3-',
#               'feats_provided_flag': True,
#               'path_selected_feats': path_pre + '/data/dict_selected_feats_physionet3.pkl'}
#     imputation = Imputation(params)
#
#     data = pickle.load(open('../../data/final_Physionet_avg_new.pkl'))
#
#     imputation(data['data'], '141309')

