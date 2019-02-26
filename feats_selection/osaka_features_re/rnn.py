import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNN_osaka(nn.Module):
    def __init__(self, params):
        super(RNN_osaka, self).__init__()
        
        self.hidden_dim = params['hidden_dim']
        self.dropout = params['dropout']
        self.layers = params['layers']
        self.input_dim = params['input_dim']
        self.tagset_size = params['tagset_size']
        self.bilstm_flag = params['bilstm_flag']
        self.attn_category = params['attn_category']
        
        self.LL = nn.Linear(params['num_of_features'],self.input_dim)
        
        if(self.bilstm_flag):
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim/2, num_layers = self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = self.layers, 
                                bidirectional=False, batch_first=True, dropout=self.dropout)
        
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        
        if(self.attn_category == 'dot'):
            print "Dot Attention is being used!"
            self.attn = DotAttentionLayer(self.hidden_dim).cuda()

    
    def init_hidden(self, batch_size):
    # num_layes, minibatch size, hidden_dim
        if(self.bilstm_flag):
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                             batch_size,
                                                             self.hidden_dim/2).fill_(0)),
                   autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            self.hidden_dim/2).fill_(0)))
        else:
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                             batch_size,
                                                             self.hidden_dim).fill_(0)),
                   autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                            batch_size,
                                                            self.hidden_dim).fill_(0)))
    
    def forward(self, features, lengths):
        features = self.LL(features)
        
        packed = pack_padded_sequence(features, lengths, batch_first = True)
        self.hidden = self.init_hidden(features.shape[0])
        
        packed_output, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]
        
        if(self.attn_category=='dot'):
            pad_attn = self.attn((lstm_out, torch.cuda.LongTensor(lengths)))
            tag_space = self.hidden2tag(pad_attn)
        else:
            lstm_out_last = self.get_q(lstm_out, lengths)
            tag_space = self.hidden2tag(lstm_out_last)
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score
    
    def get_q(self, lstm_, len_list):
        q = []
        for ind, each_len in enumerate(len_list):
            t_ = lstm_[ind, each_len-1,:]
            q.append(t_)
        q = torch.stack(q,dim=0)
        return q


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