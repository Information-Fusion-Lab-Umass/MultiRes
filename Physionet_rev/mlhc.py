import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.layers = params['layers']
        self.dropout = params['dropout']
        self.tagset_size = params['tagset_size']

        #self.num_slow_feats = len(params['slow_features_indexes'])
        #self.num_moderate_feats = len(params['moderate_features_indexes'])
        #self.num_fast_feats = len(params['fast_features_indexes'])


        self.num_slow_feats = 37
        self.num_moderate_feats = 26
        self.num_fast_feats = 4

        self.lstm_hidden_dim = 10
        

        self.lstm_slow = nn.LSTM(self.num_slow_feats, int(self.lstm_hidden_dim / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.lstm_moderate = nn.LSTM(self.num_moderate_feats, int(self.lstm_hidden_dim / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.lstm_fast = nn.LSTM(self.num_fast_feats, int(self.lstm_hidden_dim / 2), num_layers=self.layers, bidirectional=True, batch_first=True, dropout=self.dropout)

        self.hidden_dim_slow = 2*self.num_slow_feats
        self.hidden_dim_moderate = 2*self.num_moderate_feats
        self.hidden_dim_fast = 2*self.num_fast_feats

        self.hidden2tag = nn.Linear(self.lstm_hidden_dim*3, self.tagset_size)

        self.attn_slow = DotAttentionLayer( self.lstm_hidden_dim ).cuda()
        self.attn_moderate = DotAttentionLayer(self.lstm_hidden_dim ).cuda()
        self.attn_fast = DotAttentionLayer( self.lstm_hidden_dim).cuda()

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

        # Concatenate slow, moderate and fast
        pad_attn = torch.cat((pad_attn_slow, pad_attn_moderate,pad_attn_fast), 1)

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
        #lengths = torch.LongTensor(lenghts)
        lengths = autograd.Variable(lengths)

        packed = pack_padded_sequence(features, lengths, batch_first=True)
        batch_size = 1

        # LSTM Layers
        self.hidden = self.init_hidden(batch_size, frequency_flag=flag)                                     # Tuple, length 2  Size of [0] and [1]: (2,1,d*10)
        #packed_output, self.hidden = self.lstm_fast(packed, self.hidden)
        #lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]                                    # Size: (1, timesteps, 2*d*10)

        # Attention
        if flag == 'fast':
            
            packed_output, self.hidden = self.lstm_fast(packed, self.hidden)
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
            pad_attn = self.attn_fast((lstm_out, torch.cuda.LongTensor(lengths)))
            #pad_attn = self.attn_fast((lstm_out, torch.LongTensor(lengths)))
        elif flag == 'slow':
            packed_output, self.hidden = self.lstm_slow(packed, self.hidden)
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
            pad_attn = self.attn_slow((lstm_out, torch.cuda.LongTensor(lengths)))
            #pad_attn = self.attn_slow((lstm_out, torch.LongTensor(lengths)))
        elif flag == 'moderate':
            packed_output, self.hidden = self.lstm_moderate(packed, self.hidden)
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
            pad_attn = self.attn_moderate((lstm_out, torch.cuda.LongTensor(lengths)))
            #pad_attn = self.attn_moderate((lstm_out, torch.LongTensor(lengths)))
        else:
            pad_attn = 0 #error


                                 
        return pad_attn


    def init_hidden(self, batch_size, frequency_flag=None):
        '''
        # num_layes, minibatch size, hidden_dim
        if frequency_flag == 'fast':
            hidden_dim = self.hidden_dim_fast
        elif frequency_flag == 'slow':
            hidden_dim = self.hidden_dim_slow
        elif frequency_flag == 'moderate':
            hidden_dim = self.hidden_dim_moderate
        else:
            hidden_dim = 0
        '''
        hidden_dim = self.lstm_hidden_dim
        
        return (autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            hidden_dim/2).fill_(0)),
                autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            hidden_dim/2).fill_(0)))
        '''
        return (autograd.Variable(torch.FloatTensor(self.layers*2,
                                                        batch_size,
                                                        int(hidden_dim/2)).fill_(0)),
            autograd.Variable(torch.FloatTensor(self.layers*2,
                                                        batch_size,
                                                        int(hidden_dim/2)).fill_(0)))
        '''

                        


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