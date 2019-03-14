"""
Python module that defines Simple LSTM Based NN module.
"""
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Simple LSTM followe by a dense layer for predicting time series.
    """

    def __init__(self, num_features,
                 num_classes=3,
                 hidden_size=64,
                 dropout=0,
                 bidirectional=False):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        dense_layer_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(dense_layer_hidden_size, num_classes)

    def forward(self, tensor_data):
        # Extracting actual data form the tuple.
        input_sequence = tensor_data[0].unsqueeze(0)

        # print("input shape:", input_sequence.shape)
        assert not (input_sequence != input_sequence).any(), "null exists in input!"

        lstm_out, hidden = self.lstm(input_sequence)
        y_out = lstm_out[:, -1].unsqueeze(0)
        y_out = self.linear(y_out)
        y_out = y_out.squeeze(0).squeeze(0)

        return y_out
