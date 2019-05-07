import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd

# Set random seeds
#import numpy as np
#import random
#random.seed(1)
#np.random.seed(1)
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.layers = params['layers']
        self.dropout = params['dropout']
        self.tagset_size = params['tagset_size']
        self.batch_size = params['batch_size']

        self.num_slow_feats = len(params['slow_features_indexes']) + len(params['moderate_features_indexes']) + len(params['fast_features_indexes'])
        self.num_moderate_feats = len(params['moderate_features_indexes']) + len(params['fast_features_indexes'])
        self.num_fast_feats = len(params['fast_features_indexes'])
        self.lstm_output_type = params['lstm_output_type']    
        self.use_second_attention = params['use_second_attention']

        if params['lstm_output_type']  == 'different':      
            self.hidden_dim_slow = params['hidden_dim_slow']                # params['hidden_constant']*self.num_slow_feats
            self.hidden_dim_moderate = params['hidden_dim_moderate']        # params['hidden_constant']*self.num_moderate_feats
            self.hidden_dim_fast = params['hidden_dim_fast']                # params['hidden_constant']*self.num_fast_feats
    
            self.lstm_slow = nn.LSTM(self.num_slow_feats, int(self.hidden_dim_slow / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
            self.lstm_moderate = nn.LSTM(self.num_moderate_feats, int(self.hidden_dim_moderate / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
            self.lstm_fast = nn.LSTM(self.num_fast_feats, int(self.hidden_dim_fast / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
            
            self.attn_slow = DotAttentionLayer( self.hidden_dim_slow ).cuda()
            self.attn_moderate = DotAttentionLayer(self.hidden_dim_moderate ).cuda()
            self.attn_fast = DotAttentionLayer( self.hidden_dim_fast).cuda()
            if params['use_second_attention']:
                self.final_attn = DotAttentionLayer(self.hidden_dim_slow).cuda() # we will pad to be same size as largest output (which is slow)
                self.hidden2tag = nn.Linear(self.hidden_dim_slow  ,self.tagset_size)
            else:
                self.hidden2tag = nn.Linear(self.hidden_dim_slow+self.hidden_dim_moderate+self.hidden_dim_fast, self.tagset_size)

        elif params['lstm_output_type'] == 'same':
            
            self.lstm_hidden_dim = params['lstm_hidden_dim']
            
            self.lstm_slow = nn.LSTM(self.num_slow_feats, int(self.lstm_hidden_dim / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
            self.lstm_moderate = nn.LSTM(self.num_moderate_feats, int(self.lstm_hidden_dim / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
            self.lstm_fast = nn.LSTM(self.num_fast_feats, int(self.lstm_hidden_dim / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
    
    
            self.attn_slow = DotAttentionLayer( self.lstm_hidden_dim ).cuda()
            self.attn_moderate = DotAttentionLayer(self.lstm_hidden_dim ).cuda()
            self.attn_fast = DotAttentionLayer( self.lstm_hidden_dim).cuda()
            if params['use_second_attention']:
                self.final_attn = DotAttentionLayer(self.lstm_hidden_dim).cuda()
                self.hidden2tag = nn.Linear(self.lstm_hidden_dim,self.tagset_size)
            else:
                self.hidden2tag = nn.Linear(self.lstm_hidden_dim*3, self.tagset_size)

  
    def forward(self, data):
        slow_feats = Variable(torch.cuda.FloatTensor(data[1]))
        moderate_feats = Variable(torch.cuda.FloatTensor(data[2]))
        fast_feats = Variable(torch.cuda.FloatTensor(data[3]))
        #slow_feats = Variable(torch.FloatTensor(data[1]))
        #moderate_feats = Variable(torch.FloatTensor(data[2]))
        #fast_feats = Variable(torch.FloatTensor(data[3]))

        slow_feats = slow_feats.unsqueeze(0) #this is to add a batch size dimension
        moderate_feats = moderate_feats.unsqueeze(0)
        fast_feats = fast_feats.unsqueeze(0)

        # Forward passes

        #print(slow_feats.shape,moderate_feats.shape,fast_feats.shape)
        pad_attn_slow = self._forward(slow_feats, 'slow')
        pad_attn_moderate = self._forward(moderate_feats,'moderate')
        pad_attn_fast = self._forward(fast_feats,'fast')
        
        if self.lstm_output_type == 'same':
            if self.use_second_attention:
                new_tensor = torch.cuda.FloatTensor(1,3,self.lstm_hidden_dim)
                #first_attns = torch.cat(1,(pad_attn_slow, pad_attn_moderate,pad_attn_fast)) #concat to be 1x3xhidden_dim
                new_tensor[:,0,:] = pad_attn_slow
                new_tensor[:,1,:] = pad_attn_moderate
                new_tensor[:,2,:] = pad_attn_fast
                pad_attn = self.final_attn((new_tensor,autograd.Variable(torch.cuda.LongTensor([3]))))#length of 3 always because fast,slow,moderate
            else:    
                # Concatenate slow, moderate and fast
                pad_attn = torch.cat((pad_attn_slow, pad_attn_moderate,pad_attn_fast), 1) #concat to be 1x3*hidden_dim
        elif self.lstm_output_type == 'different':
            if self.use_second_attention:
                #pad all with zeros
                new_tensor = torch.cuda.FloatTensor(1,3,self.hidden_dim_slow)
                padded_moderate = F.pad(pad_attn_moderate,pad=(0,self.hidden_dim_slow-self.hidden_dim_moderate))
                padded_fast = F.pad(pad_attn_fast,pad = (0,self.hidden_dim_slow-self.hidden_dim_fast))
                new_tensor[:,0,:] = pad_attn_slow
                new_tensor[:,1,:] = padded_moderate
                new_tensor[:,2,:] = padded_fast
                pad_attn = self.final_attn((new_tensor,autograd.Variable(torch.cuda.LongTensor([3]))))#length of 3 always because fast,slow,moderate
            else:
                # Concatenate slow, moderate and fast
                pad_attn = torch.cat((pad_attn_slow, pad_attn_moderate,pad_attn_fast), 1) #concat to be 1x3*hidden_dim

        # Pass through FC layer and Softmax
        tag_space = self.hidden2tag(pad_attn)
        tag_score = F.log_softmax(tag_space, dim=1)
        
        # Return predictions
        return tag_score


    def _forward(self, features, flag):
        """
        Runs the forward pass of the model for high frequency features, till the attention mechanism
        """

        lenghts = [features.shape[1]]  # timesteps
        lengths = torch.cuda.LongTensor(lenghts)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first=True)

        # LSTM Layers
        self.hidden = self.init_hidden(self.batch_size, frequency_flag=flag)                                     # Tuple, length 2  Size of [0] and [1]: (2,1,d*10)
        #packed_output, self.hidden = self.lstm_fast(packed, self.hidden)
        #lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]                                    # Size: (1, timesteps, 2*d*10)

        # Attention
        if flag == 'fast':
            packed_output, self.hidden = self.lstm_fast(packed, self.hidden)
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
            pad_attn = self.attn_fast((lstm_out, torch.cuda.LongTensor(lengths)))
        elif flag == 'slow':
            packed_output, self.hidden = self.lstm_slow(packed, self.hidden)
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
            pad_attn = self.attn_slow((lstm_out, torch.cuda.LongTensor(lengths)))
        elif flag == 'moderate':
            packed_output, self.hidden = self.lstm_moderate(packed, self.hidden)
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
            pad_attn = self.attn_moderate((lstm_out, torch.cuda.LongTensor(lengths)))
        else:
            pad_attn = 0 #error
                                 
        return pad_attn


    def init_hidden(self, batch_size, frequency_flag=None):
        if self.lstm_output_type == 'different':
            # num_layes, minibatch size, hidden_dim
            if frequency_flag == 'fast':
                    hidden_dim = self.hidden_dim_fast
            elif frequency_flag == 'slow':
                    hidden_dim = self.hidden_dim_slow
            elif frequency_flag == 'moderate':
                    hidden_dim = self.hidden_dim_moderate
        
     
        elif self.lstm_output_type == 'same':
            hidden_dim = self.lstm_hidden_dim
        
        return (autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            hidden_dim/2).fill_(0)),
                autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            hidden_dim/2).fill_(0)))


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
