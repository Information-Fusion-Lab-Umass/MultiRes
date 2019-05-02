import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class RNN_osaka(nn.Module):
    def __init__(self, params):
        super(RNN_osaka, self).__init__()

        self.bilstm_flag = params['bilstm_flag']
        self.hidden_dim = params['hidden_dim']
        self.dropout = params['dropout']
        self.layers = params['layers']
        self.tagset_size = params['tagset_size']
        self.attn_category = params['attn_category']
        self.num_features = params['num_features']
        self.imputation_layer_dim_op = params['imputation_layer_dim_op']
        self.selected_feats = params['selected_feats']
        self.imputation_layer_dim_in = (self.selected_feats+1)*4
        self.input_dim = self.num_features * self.imputation_layer_dim_op
        self.hidden_dim = 2*self.input_dim

        self.dict_selected_feats = {}
        for each_ind in range(self.num_features):
            all_feats = list(range(self.num_features))
            all_feats.remove(each_ind)
            random.shuffle(all_feats)
            self.dict_selected_feats[each_ind] = [each_ind] + all_feats[:self.selected_feats]


#         self.LL = nn.Linear(self.len_features, self.input_dim)

        # self.imputation_layer_in = nn.ModuleList([nn.Linear(self.imputation_layer_dim_in,self.imputation_layer_dim_op) for x in range(self.num_features)])
        self.imputation_layer_in = [nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op) for x in range(self.num_features)]
#         self.imputation_layer_op = nn.Linear(self.imputation_layer_dim, 1)

        if(self.bilstm_flag):
            self.lstm = nn.LSTM(self.input_dim, int(self.hidden_dim/2), num_layers = self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = self.layers,
                                bidirectional=False, batch_first=True, dropout=self.dropout)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        if(self.attn_category == 'dot'):
            print ("Dot Attention is being used!")
            self.attn = DotAttentionLayer(self.hidden_dim)


    def init_hidden(self, batch_size):
    # num_layes, minibatch size, hidden_dim

        if(self.bilstm_flag):
            return (autograd.Variable(torch.FloatTensor(self.layers*2,
                                                             batch_size,
                                                             int(self.hidden_dim/2)).fill_(0)),
                   autograd.Variable(torch.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            int(self.hidden_dim/2)).fill_(0)))
        else:
            return (autograd.Variable(torch.FloatTensor(self.layers,
                                                             batch_size,
                                                             self.hidden_dim).fill_(0)),
                   autograd.Variable(torch.FloatTensor(self.layers,
                                                            batch_size,
                                                            self.hidden_dim).fill_(0)))

    def forward(self, data, id_):
#         features = self.LL(features)
        features = self.get_imputed_feats(data[id_][0], data[id_][1], self.dict_selected_feats)
        lenghts = [features.shape[1]]
        lengths = torch.LongTensor(lenghts)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first = True)

        batch_size = 1
        self.hidden = self.init_hidden(batch_size)

        packed_output, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]

        if(self.attn_category=='dot'):
            pad_attn = self.attn((lstm_out, torch.LongTensor(lengths)))
            tag_space = self.hidden2tag(pad_attn)
        else:
            tag_space = self.hidden2tag(lstm_out[:,-1,:])
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score

    def get_imputed_feats(self, feats, flags, dict_selected_feats):
        feats = np.asarray(feats)
        flags = np.asarray(flags)
        all_features = []
        num_features = self.num_features
        input_ = {}
        for feat_ind in range(num_features):
            input_[feat_ind] = []
            feat = feats[:,feat_ind]
            feat_flag = flags[:,feat_ind]
            ind_keep = feat_flag==0
            ind_missing = feat_flag==1
            if(sum(ind_keep)>0):
                avg_val = np.mean(feat[ind_keep])
            else:
                avg_val = 0.0
            last_val_observed = avg_val
            delta_t = -1
            for ind, each_flag in enumerate(feat_flag):
                if(each_flag==1):
                    imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                    input_[feat_ind].append(imputation_feat)
#                     input_[feat_ind][ind] = autograd.Variable(torch.FloatTensor(imputation_feat))
#                     f_ = self.imputation_layer_in[feat_ind](input_)
                elif(each_flag==0):
                    delta_t = 0
                    last_val_observed = feat[ind]
                    imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                    input_[feat_ind].append(imputation_feat)
#                     input_[feat_ind][ind] = autograd.Variable(torch.FloatTensor(imputation_feat))
#                     input_ = input_
#                     f_ = self.imputation_layer_in[feat_ind](input_)
                delta_t+=1
#                 final_feat_list.append(f_)
#             final_feat_list = torch.stack(final_feat_list)
#             all_features.append(final_feat_list)

        for feat_ind in range(num_features):
            final_feat_list = []

            for ind, each_flag in enumerate(feat_flag):
                imputation_feat = []


                for each_selected_feat in dict_selected_feats[feat_ind]:
                    imputation_feat+=input_[each_selected_feat][ind]
                imputation_feat = autograd.Variable(torch.FloatTensor(imputation_feat))
                f_= self.imputation_layer_in[feat_ind](imputation_feat)
                final_feat_list.append(f_)
            final_feat_list = torch.stack(final_feat_list)
            all_features.append(final_feat_list)
        all_features = torch.cat(all_features,1)
        all_features = all_features.unsqueeze(0)
        all_features = autograd.Variable(all_features)
        return all_features

    def prepare_batch(self, dict_data, ids):
        labels = []
        for each_id in ids:
            t_label = label_mapping[dict_data[each_id][1]]
            labels.append(t_label)
        features = []
        max_len = 0
        actual_lens = []

        for each_id in ids:
            t_features = dict_data[each_id][0]
            features.append(t_features)
            if(len(t_features)>max_len):
                max_len = len(t_features)
            actual_lens.append(len(t_features))

        for ind in range(len(features)):
            features[ind] = features[ind]+[[0 for x in range(21)] for y in range(max_len-len(features[ind]))]

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

        sorted_features = torch.FloatTensor(sorted_features)
        sorted_features = autograd.Variable(sorted_features)

        sorted_labels = torch.LongTensor(sorted_labels)
        sorted_labels = autograd.Variable(sorted_labels)

        return sorted_features, sorted_labels, sorted_lens


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

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
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = autograd.Variable((idxes<lengths.unsqueeze(1)).float())

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output
