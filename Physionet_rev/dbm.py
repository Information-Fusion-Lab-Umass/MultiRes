import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import copy

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

        # Frequency based feature splitting
        self.fast_indexes = params['fast_features_indexes']
        self.slow_indexes = params['slow_features_indexes']

        self.num_features_fast = len(self.fast_indexes)
        self.num_features_slow = len(self.slow_indexes)

        self.input_dim = self.num_features * self.imputation_layer_dim_op
        self.input_dim_fast = self.num_features_fast * self.imputation_layer_dim_op
        self.input_dim_slow = self.num_features_slow * self.imputation_layer_dim_op

        self.hidden_dim = 2*self.input_dim
        self.hidden_dim_fast = 2 * self.input_dim_fast
        self.hidden_dim_slow = 2 * self.input_dim_slow
        
        self.dict_selected_feats = {}
        for each_ind in range(self.num_features):
            all_feats = range(self.num_features)
            all_feats.remove(each_ind)
            random.shuffle(all_feats)
            self.dict_selected_feats[each_ind] = [each_ind] + all_feats[:self.selected_feats]

        # First hidden layers in DBM; One per feature
        self.imputation_layer_in_fast = [nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op).cuda() for x in range(self.num_features_fast)]
        self.imputation_layer_in_slow = [nn.Linear(self.imputation_layer_dim_in, self.imputation_layer_dim_op).cuda() for x in range(self.num_features_slow)]
        
        if(self.bilstm_flag):
            self.lstm_slow = nn.LSTM(self.input_dim_slow, self.hidden_dim_slow/2, num_layers = self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
            self.lstm_fast = nn.LSTM(self.input_dim_fast, self.hidden_dim_fast/2, num_layers=self.layers,
                                     bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm_slow = nn.LSTM(self.input_dim_slow, self.hidden_dim_slow, num_layers = self.layers,
                                bidirectional=False, batch_first=True, dropout=self.dropout)
            self.lstm_fast = nn.LSTM(self.input_dim_fast, self.hidden_dim_fast, num_layers=self.layers,
                                     bidirectional=False, batch_first=True, dropout=self.dropout)

        # Final FC layer of the DBM
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        
        if(self.attn_category == 'dot'):
            print "Dot Attention is being used!"
            self.attn_slow = DotAttentionLayer(self.hidden_dim_slow).cuda()
            self.attn_fast = DotAttentionLayer(self.hidden_dim_fast).cuda()

    
    def init_hidden(self, batch_size, frequency_flag=None):
    # num_layes, minibatch size, hidden_dim
        if frequency_flag == 'fast':
            hidden_dim = self.hidden_dim_fast
        elif frequency_flag == 'slow':
            hidden_dim = self.hidden_dim_slow
        else:
            hidden_dim = self.hidden_dim

        if(self.bilstm_flag):
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                             batch_size,
                                                             hidden_dim/2).fill_(0)),
                   autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            hidden_dim/2).fill_(0)))
        else:
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                             batch_size,
                                                             hidden_dim).fill_(0)),
                   autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                            batch_size,
                                                            hidden_dim).fill_(0)))


    def get_imputed_feats(self, feats, flags, dict_selected_feats, frequency_flag):
        feats = np.asarray(feats)     # t_n x d
        flags = np.asarray(flags)     # t_n x d
        all_features = []

        # num_features = self.num_features
        # Replaced above line with,
        if frequency_flag == 'fast':
            num_features = len(self.fast_indexes)
        else:
            num_features = len(self.slow_indexes)

        input_ = {}

        # Fill in missing value flags
        for feat_ind in range(num_features):
            input_[feat_ind] = []
            feat = feats[:,feat_ind]      # (t_n, )
            feat_flag = flags[:,feat_ind]
            ind_keep = feat_flag==0
            ind_missing = feat_flag==1
            if(sum(ind_keep)>0):
                avg_val = np.mean(feat[ind_keep])
            else:
                avg_val = 0.0
            last_val_observed = avg_val
            delta_t = -1

            # Create four valued tuple for each feature
            for ind, each_flag in enumerate(feat_flag):
                if(each_flag==1):
                    imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                    input_[feat_ind].append(imputation_feat)
#                     input_[feat_ind][ind] = autograd.Variable(torch.cuda.FloatTensor(imputation_feat))
#                     f_ = self.imputation_layer_in[feat_ind](input_)
                elif(each_flag==0):
                    delta_t = 0
                    last_val_observed = feat[ind]
                    imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                    input_[feat_ind].append(imputation_feat)
#                     input_[feat_ind][ind] = autograd.Variable(torch.cuda.FloatTensor(imputation_feat)) 
#                     input_ = input_.cuda()
#                     f_ = self.imputation_layer_in[feat_ind](input_)
                delta_t+=1
#                 final_feat_list.append(f_)
#             final_feat_list = torch.stack(final_feat_list)
#             all_features.append(final_feat_list)

        # Pass each feature's tuple into a seperate FC Layer (d FC layers, running vertically)
        for feat_ind in range(num_features):
            final_feat_list = []
            for ind, each_flag in enumerate(feat_flag):
                imputation_feat = []
                for each_selected_feat in dict_selected_feats[feat_ind]:
                    imputation_feat+=input_[each_selected_feat][ind]
                
                # For each of the 37 features, one FC layer maps the 1x4 vector to a 1x10 representation
                imputation_feat = autograd.Variable(torch.cuda.FloatTensor(imputation_feat))     # (4,)
                if frequency_flag == 'fast':
                    f_= self.imputation_layer_in_fast[feat_ind](imputation_feat)                 # (10, )
                else:
                    f_ = self.imputation_layer_in_slow[feat_ind](imputation_feat)                # (10, )
                final_feat_list.append(f_)                                                       
            final_feat_list = torch.stack(final_feat_list)                                       # List len of t (timesteps), each element is (10,). This line converts it to matrix of (t, 10)
            all_features.append(final_feat_list)                                                 # List of len d, each element is (t, 10)
        all_features = torch.cat(all_features,1)                                                 # (t, d*10)
        all_features = all_features.unsqueeze(0)                                                 # (1, t, d*10)
        all_features = autograd.Variable(all_features)

        return all_features

    def forward(self, data, id_):

            # Split the features based on their frequency of occurrence
            data_fast, data_slow = self._split_data_fast_slow(data)
            data_fast = self._remove_missing_rows(data_fast)
            data_slow = self._remove_missing_rows(data_slow)

            # Pass features for both frequencies through model
            pad_attn_slow = self.forward_slow(data_slow, id_)
            pad_attn_fast = self.forward_fast(data_fast, id_)
            # print "ATTN", pad_attn_slow.shape, pad_attn_fast.shape
            # print "hidden2tag", self.hidden2tag

            # Concatenate outputs from both 'legs' of the model, and run through the FC layer hidden2tag
            if (self.attn_category == 'dot'):
                pad_attn = torch.cat((pad_attn_slow, pad_attn_fast), 1)
                # print "CAT ATTN", pad_attn.shape
                tag_space = self.hidden2tag(pad_attn)
            else:
                # Ignoring modifying code in this case, since attention is always being used
                tag_space = self.hidden2tag(lstm_out[:, -1, :])

            # Get final Confidence Scores
            tag_score = F.log_softmax(tag_space, dim=1)                                                      # Size: (1, 2)  (Confidence scores for binary classification)

            return tag_score

    def forward_slow(self, data, id_):
        """
        Runs the forward pass of the model for low frequency features, till the attention mechanism
        """
        features = self.get_imputed_feats(data[id_][0], data[id_][1], self.dict_selected_feats, 'slow')      # Size: (1, timesteps, d*10)
        # print "features", features.shape

        lenghts = [features.shape[1]]                                                                        # timesteps
        lengths = torch.cuda.LongTensor(lenghts)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first=True)
        batch_size = 1

        # LSTM Layers
        self.hidden = self.init_hidden(batch_size, frequency_flag='slow')                                    # Tuple, length 2  Size of [0] and [1]: (2,1,d*10)
        # print "hidden", self.hidden[0].shape, self.hidden[1].shape
        # print "lstm", self.lstm_slow
        # print "packed", type(packed), len(packed)
        packed_output, self.hidden = self.lstm_slow(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]                                   # Size: (1, timesteps, 2*d*10)

        # Attention
        if(self.attn_category=='dot'):
            pad_attn = self.attn_slow((lstm_out, torch.cuda.LongTensor(lengths)))                            # Size: (1, 2*d*10) (timestep independent)

        return pad_attn


    def forward_fast(self, data, id_):
        """
        Runs the forward pass of the model for high frequency features, till the attention mechanism
        """
        features = self.get_imputed_feats(data[id_][0], data[id_][1], self.dict_selected_feats, 'fast')      # Size: (1, timesteps, d*10)

        lenghts = [features.shape[1]]  # timesteps
        lengths = torch.cuda.LongTensor(lenghts)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first=True)
        batch_size = 1

        # LSTM Layers
        self.hidden = self.init_hidden(batch_size, frequency_flag='fast')                                     # Tuple, length 2  Size of [0] and [1]: (2,1,d*10)
        packed_output, self.hidden = self.lstm_fast(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]                                    # Size: (1, timesteps, 2*d*10)

        # Attention
        if(self.attn_category=='dot'):
            pad_attn = self.attn_fast((lstm_out, torch.cuda.LongTensor(lengths)))                             # Size: (1, 2*d*10) (timestep independent)

        return pad_attn

    
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

        sorted_features = torch.cuda.FloatTensor(sorted_features)
        sorted_features = autograd.Variable(sorted_features)

        sorted_labels = torch.cuda.LongTensor(sorted_labels)
        sorted_labels = autograd.Variable(sorted_labels)

        return sorted_features, sorted_labels, sorted_lens


    def _split_data_fast_slow(self, data):
        """
        Returns fast/slow data the same structure as data
        """
        fast_data = copy.deepcopy(data)
        slow_data = copy.deepcopy(data)

        # NOTE: PYTHON2 is
        for ID, current_data in data.iteritems():
        # NOTE PYTHON3 is
        # for ID, current_data in data.items():
            list_data = current_data[0]  # get the data as a list
            numpy_data = np.matrix(list_data)  # convert it to numpy matrix
            fast_feats = numpy_data[:, self.fast_indexes]
            slow_feats = numpy_data[:, self.slow_indexes]

            list_missing = current_data[1]  # get the missing matrix
            numpy_missing = np.matrix(list_missing)
            fast_missing = numpy_missing[:, self.fast_indexes]
            slow_missing = numpy_missing[:, self.slow_indexes]

            #print "Fast features: ", fast_feats.shape
            #print "Slow features: ", slow_feats.shape

            # some sanity checks
            assert fast_feats.shape[1] == len(self.fast_indexes)
            assert slow_feats.shape[1] == len(self.slow_indexes)
            assert self.num_features == len(self.fast_indexes) + len(self.slow_indexes)

            # A tuple is immutable, so you need to create a new one
            # index 2 (timestamps) and index 3 (label) are unchanged
            fast_data[ID] = (fast_feats.tolist(), fast_missing.tolist(), fast_data[ID][2], fast_data[ID][3])
            slow_data[ID] = (slow_feats.tolist(), slow_missing.tolist(), slow_data[ID][2], slow_data[ID][3])

        return fast_data, slow_data


    def _remove_missing_rows(self, data):
        """
        Removes rows that are all missing (missing flag is 1 for each feature)
        """
        for ID, current_data in data.items():
            numpy_data = np.matrix(current_data[0])
            numpy_missing = np.matrix(current_data[1])
            timestamps = np.array(data[ID][2])
            num_timestamps = numpy_data.shape[0]
            rows_to_remove = []
            for row in range(num_timestamps):
                current_row = numpy_missing[row, :]
                equal = np.array_equal(current_row, np.ones((1, numpy_data.shape[1])))
                if equal:
                    rows_to_remove.append(row)

            numpy_data = np.delete(numpy_data, (rows_to_remove), axis=0)
            numpy_missing = np.delete(numpy_missing, (rows_to_remove), axis=0)
            timestamps = np.delete(timestamps, rows_to_remove)

            assert numpy_data.shape[0] == numpy_missing.shape[0]
            assert len(timestamps) == numpy_data.shape[0]

            data[ID] = (numpy_data, numpy_missing, timestamps, data[ID][3])

        return data


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
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
        mask = autograd.Variable((idxes<lengths.unsqueeze(1)).float())

        alphas = alphas * mask

        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)

        return output

