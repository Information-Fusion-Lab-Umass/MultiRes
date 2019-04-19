import torch
import torch.nn as nn

from src.models import autoencoder
from src.models import user_dense_heads
from src.bin import validations

LOW_MODEL_CAPACITY_WARNING = "Input size greater than hidden size. This may result in a low capacity network"


class MultiTaskAutoEncoderLearner(nn.Module):
    def __init__(self,
                 users: list,
                 autoencoder_input_size,
                 autoencoder_bottleneck_feature_size,
                 autoencoder_num_layers,
                 shared_hidden_layer_size,
                 user_dense_layer_hidden_size,
                 num_classes,
                 num_covariates=0,
                 shared_layer_dropout_prob=0,
                 user_head_dropout_prob=0):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param autoencoder_input_size: Input size of the time series portion on the model.
        @param autoencoder_bottleneck_feature_size: Encoded input size of autoecoder.
        @param autoencoder_num_layers: Num layers in autoencoder LSTM model.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskAutoEncoderLearner, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.autoencoder_input_size = autoencoder_input_size
        self.autoencoder_bottleneck_feature_size = autoencoder_bottleneck_feature_size
        self.autoencoder_num_layers = autoencoder_num_layers
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates
        self.shared_layer_dropout_prob = shared_layer_dropout_prob
        self.user_head_dropout_prob = user_head_dropout_prob

        # Layer initialization.
        self.autoencoder = autoencoder.LSTMAE(self.autoencoder_input_size,
                                              self.autoencoder_bottleneck_feature_size,
                                              self.autoencoder_num_layers,
                                              self.is_cuda_avail)

        self.shared_linear = nn.Linear(self.autoencoder_bottleneck_feature_size + self.num_covariates,
                                       self.shared_hidden_layer_size)

        self.shared_activation = nn.ReLU()
        self.shared_layer_dropout = nn.Dropout(p=self.shared_layer_dropout_prob)
        self.shared_linear_1 = nn.Linear(self.shared_hidden_layer_size, self.shared_hidden_layer_size // 2)
        self.shared_activation_1 = nn.ReLU()

        self.user_heads = user_dense_heads.UserDenseHead(self.users,
                                                         self.shared_hidden_layer_size // 2,
                                                         self.user_dense_layer_hidden_size,
                                                         self.num_classes,
                                                         self.user_head_dropout_prob)

    def forward(self, user, input_seq, covariate_data=None):
        """
        Slightly complex forward pass. The autoencoder part return the decoded output
        which needs to be trained using MAE or MSE. The user head returns a vector of
        class probability distributions which need to be trained using cross entropy.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        autoencoder.
        @param covariate_data: The covariates which will be concatenated with the output
        of the autoencoders before being used for classification.
        @return: output of the autoencoder and the probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        autoencoder_out = self.autoencoder(input_seq)
        bottle_neck = self.autoencoder.get_bottleneck_features(input_seq)
        bottle_neck = bottle_neck[:, -1, :]

        if covariate_data is not None:
            bottle_neck = torch.cat((bottle_neck, covariate_data.unsqueeze(0)), dim=1)

        shared_hidden_state = self.shared_linear(bottle_neck)
        shared_hidden_state = self.shared_activation(shared_hidden_state)
        shared_hidden_state = self.shared_layer_dropout(shared_hidden_state)
        shared_hidden_state_1 = self.shared_linear_1(shared_hidden_state)
        shared_hidden_state_1 = self.shared_activation_1(shared_hidden_state_1)

        y_out = self.user_heads(user, shared_hidden_state)

        return autoencoder_out, y_out


class MultiTaskLSTMLearner(nn.Module):
    def __init__(self,
                 users: list,
                 lstm_input_size,
                 lstm_hidden_size,
                 lstm_num_layers,
                 lstm_bidirectional,
                 shared_hidden_layer_size,
                 user_dense_layer_hidden_size,
                 num_classes,
                 dropout=0,
                 num_covariates=0):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param lstm_input_size: Input size of the time series portion on the model.
        @param lstm_hidden_size: Hidden size of the LSTM.
        @param lstm_num_layers: Num layers in LSTM.
        @param lstm_bidirectional: LSTM is bidirectional if set True.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskLSTMLearner, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_input_size = lstm_input_size

        if self.lstm_bidirectional:
            self.lstm_hidden_size = lstm_hidden_size // 2
        else:
            self.lstm_hidden_size = lstm_hidden_size

        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates

        # Layer initialization.
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True,
                            bidirectional=self.lstm_bidirectional,
                            dropout=self.dropout)

        self.shared_linear = nn.Linear(self.lstm_hidden_size + self.num_covariates,
                                       self.shared_hidden_layer_size)

        self.shared_activation = nn.ReLU()

        self.user_heads = user_dense_heads.UserDenseHead(self.users,
                                                         self.shared_hidden_layer_size,
                                                         self.user_dense_layer_hidden_size,
                                                         self.num_classes)

    def forward(self, user, input_seq, covariate_data=None):
        """
        The input sequence is inputed to the LSTM and the last hidden state of the LSTM
        is passed to the shared layer of the MultiTask Learner.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        autoencoder.
        @param covariate_data: The covariates which will be concatenated with the output
        of the autoencoders before being used for classification.
        @return: output of the autoencoder and the probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        lstm_out = self.lstm(input_seq)
        lstm_last_hidden_state = lstm_out[:, -1, :]

        if covariate_data is not None:
            embedding = torch.cat((lstm_last_hidden_state, covariate_data.unsqueeze(0)), dim=1)

        shared_hidden_state = self.shared_linear(embedding)
        shared_hidden_state = self.shared_activation(shared_hidden_state)

        y_out = self.user_heads(user, shared_hidden_state)

        return y_out


class MultiTaskMultiLSTMLearner(nn.Module):
    def __init__(self,
                 users: list,
                 lstm_input_size,
                 lstm_hidden_size,
                 lstm_num_layers,
                 lstm_bidirectional,
                 shared_hidden_layer_size,
                 user_dense_layer_hidden_size,
                 num_classes,
                 num_covariates=0,
                 lstm_dropout_prob=0,
                 shared_layer_dropout_prob=0,
                 user_head_dropout_prob=0):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param lstm_input_size: Input size of the time series portion on the model.
        @param lstm_hidden_size: Hidden size of the LSTM.
        @param lstm_num_layers: Num layers in LSTM.
        @param lstm_bidirectional: LSTM is bidirectional if set True.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskMultiLSTMLearner, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_input_size = lstm_input_size
        self.lstm_dropout_prob = lstm_dropout_prob
        self.shared_layer_dropout_prob = shared_layer_dropout_prob
        self.user_head_dropout_prob = user_head_dropout_prob
        self.lstm_num_layers = lstm_num_layers
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates

        # Layer initialization.
        self.user_lstm = user_dense_heads.UserLSTM(input_size=self.lstm_input_size,
                                                   lstm_hidden_size=self.lstm_hidden_size,
                                                   num_layers=self.lstm_num_layers,
                                                   bidirectional=self.lstm_bidirectional,
                                                   dropout=self.lstm_dropout_prob)

        self.shared_linear = nn.Linear(self.lstm_hidden_size + self.num_covariates,
                                       self.shared_hidden_layer_size)

        self.shared_activation = nn.ReLU()
        self.shared_layer_dropout = nn.Dropout(p=self.shared_layer_dropout_prob)

        self.user_heads = user_dense_heads.UserDenseHead(self.users,
                                                         self.shared_hidden_layer_size,
                                                         self.user_dense_layer_hidden_size,
                                                         self.num_classes)

    def forward(self, user, input_seq, covariate_data=None):
        """
        The input sequence is inputed to the LSTM user head and the last hidden state of the LSTM
        is passed to the shared layer of the MultiTask Learner.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        autoencoder.
        @param covariate_data: The covariates which will be concatenated with the output
        of the LSTM before being used for classification.
        @return: output of the LSTM and the probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        lstm_out = self.user_lstm(user, input_seq)
        lstm_last_hidden_state = lstm_out[:, -1, :]

        if covariate_data is not None:
            embedding = torch.cat((lstm_last_hidden_state, covariate_data.unsqueeze(0)), dim=1)

        shared_hidden_state = self.shared_linear(embedding)
        shared_hidden_state = self.shared_activation(shared_hidden_state)
        shared_hidden_state = self.shared_layer_dropout(shared_hidden_state)

        y_out = self.user_heads(user, shared_hidden_state)

        return y_out
