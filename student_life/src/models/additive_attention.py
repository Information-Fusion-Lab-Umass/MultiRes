"""
Implementation is based on the paper -  NEURAL MACHINE TRANSLATION
BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
Author - Bahdanau et. al.

Deviation from paper - Use of LSTM rather than GRU.
"""
import torch
from torch import nn
from torch import functional


class AdditiveAttentionEncoder(nn.Module):
    """
    This encoder pair is mainly referenced from
    https://github.com/abhinavshaw1993/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb

    The RNN unit used here is LSTM.
    todo(abhinavshaw): Make the RNN configurable.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 is_cuda,
                 dropout_p=0):
        super(AdditiveAttentionEncoder, self).__init__()
        # Input size is same as feature size.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_cuda = is_cuda
        self.dropout_p = dropout_p

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input_seq):
        """

        @param input_seq: The input sequence that needs to be encoded.
        @return: Encoded input sequence and the context vector which
        is formed by the last hidden state of the RNN.
        @attention: The last hidden state (both the forward and the backward hidden states)
                    are put through a fully connected layer with tanh activation(As in paper).
        """

        outputs, (hidden_state, cell_state) = self.rnn(input_seq)
        # outputs(seq_len, num_features)
        # hidden(1, hidden_size)

        """
        Hidden states are indexed as -1 and -2 because, the hidden state contains the forward
        which is the first and backward hidden state which is the second state.
        """
        hidden_forward, hidden_backward = hidden_state[-2, :, :], hidden_state[-1, :, :]
        combined_hidden_state = torch.cat((hidden_forward, hidden_backward), dim=1)
        context_vector = torch.tanh(self.linear(combined_hidden_state))

        return outputs, context_vector

    def get_encoder_dimensions(self):
        encoder_output_dim = self.hidden_size * 2
        context_vector_dim = self.hidden_size

        return encoder_output_dim, context_vector_dim


class AdditiveAttention(nn.Module):
    """
    s_t: Initially, the last hidden state of the Encoder is the initial hidden state of the Decoder.
    This hidden state is the one that is used to calculate the energy between the Encoder outputs. (value a_ij)
    """

    def __init_(self, encoder_output_dim, context_vector_dim, decoder_hidden_dim):
        self.encoder_output_dim = encoder_output_dim
        self.context_vector_dim = context_vector_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.attn = nn.Linear(input_size=self.encoder_output_dim + self.context_vector_dim,
                                hidden_size=self.context_vector_dim)
        # This matrix is multiplied by the energy which is a sequence of vectors of len [context_vector_dim]
        # To get a final vector of seq len, we need v to be [1, context_vector_dim, 1]
        # and the Energy to be [batch_size, seq_len, context_vector_dim]
        self.v = nn.Parameter(torch.rand(self.context_vector_dim, 1))

    def froward(self, encoder_outputs, hidden):
        """

        @param encoder_outputs(batch_size, seq_len, encoder_output_dim): The encoder_outputs that need to be used to calculate the energy
        between the last hidden state of the decoder.

        @param hidden(batch_size, context_vector_dim): Last hidden state of the decoder. It is the last hidden state of the encoder at t=0.
        (initially)
        @return: Return the weights that need to be used to calculate the expected context vector.
        """

        # Preparing the hidden state to match the dimensions of the ended_outputs.
        batch_size, seq_len, encoder_hidden_size = encoder_outputs.shape
        hidden = hidden.unsqueeze(1)
        hidden = hidden.repeat(1, seq_len, 1)

        # Calculating energy between the hidden state and each encoder output.
        # concatenated = [batch_size, seq_len, encoder_output_dim + context_vector_dim]
        concatenated = torch.cat((hidden, encoder_outputs), dim=2)
        # energy = [batch_size, seq_len, context_vector_dim]
        energy = torch.tanh(self.attn(concatenated))

        # v = [context_vector_dim, 1]
        v = self.v.unsqueeze(0).repeat(batch_size, 1, 1)
        # v = [batch_size, context_vector_dim, 1]
        attention = torch.bmm(energy, v).squeeze(2)
        # attention = [batch size, src len]
        attention_weights = functional.softmax(attention, dim=1)

        return attention_weights


# class AdditiveAttentionDecoder(nn.Module):
