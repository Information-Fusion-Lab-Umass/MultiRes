import warnings
import torch
import torch.nn as nn

from src.definitions import LOW_MODEL_CAPACITY_WARNING


class UserDenseHead(nn.Module):
    def __init__(self, users: list, input_size, hidden_size, num_classes, dropout=0):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(UserDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        # Layer initialization.
        if self.input_size > self.hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        dense_layer = {}
        for user in users:
            # todo(abhinavshaw): Make this configurable to any model of the users choice. can take those layers as a list.
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.num_classes))
            dense_layer[user] = sequential_liner

        self.student_dense_layer = nn.ModuleDict(dense_layer)

    def forward(self, user, input_data):
        return self.student_dense_layer[user](input_data)


class UserLSTM(nn.Module):
    def __init__(self, users: list,
                 input_size,
                 lstm_hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0):
        """
        This model has a LSTM for each user layer for each student.
        This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each LSTM.
        @param lstm_hidden_size: Hidden size of the LSTM.
        """
        super(UserDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        if self.bidirectional:
            self.lstm_hidden_size = self.lstm_hidden_size // 2

        # Layer initialization.
        if self.input_size > self.lstm_hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        lstm_layer = {}
        for user in users:
            # todo(abhinavshaw): Make this configurable to any model of the users choice. can take those layers as a list.
            lstm_layer[user] = nn.LSTM(input_size=input_size,
                                       hidden_size=self.lstm_hidden_size,
                                       batch_first=True,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=dropout)

        self.student_dense_layer = nn.ModuleDict(lstm_layer)

    def forward(self, user, input_data):
        return self.student_dense_layer[user](input_data)
