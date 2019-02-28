"""
Script for training StudentLife on GRU-D
"""
import torch

import src.experiments.grud.helper as helper
import src.bin.validations as validations
import src.bin.trainer as trainer
import src.bin.tensorify as tensorify

from src.models.grud import GRUD
from src.utils import read_utils as reader
from src import definitions

GRU_D_CONFIG = reader.read_yaml(definitions.MODEL_CONFIG_FILE_PATH)['gru_d']


def initialize_gru(num_features, hidden_size, output_size,
                   x_mean, num_layers, learning_rate, dropout_type='mloss'):
    ######################## Initialization ########################
    # Note : GRUD accepts data with rows as features and columns as time steps!
    model = GRUD(input_size=num_features,
                 hidden_size=hidden_size,
                 output_size=output_size,
                 dropout=0,
                 dropout_type=dropout_type,
                 x_mean=x_mean,
                 num_layers=num_layers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_gru():
    # Data
    student_list = GRU_D_CONFIG[definitions.STUDENT_LIST_CONFIG_KEY]
    data = helper.get_data_for_gru_d(*student_list)

    # Parameter Setup
    output_size = GRU_D_CONFIG['classes']
    first_key = next(iter(data['data'].keys()))
    num_features = len(data['data'][first_key][0])
    hidden_size = num_features
    num_layers = GRU_D_CONFIG['num_layers']
    x_mean = GRU_D_CONFIG['x_mean']
    learning_rate = GRU_D_CONFIG['learning_rate']
    learning_rate_decay = GRU_D_CONFIG['learning_rate_decay']
    n_epochs = GRU_D_CONFIG['epochs']

    # Cuda Enabled.
    if torch.cuda.device_count() >0:
        cuda_enabled = True
    else:
        cuda_enabled = False

    # Data to tensors
    data = tensorify.tensorify_data_gru_d(data)
    model, criterion, optimizer = initialize_gru(num_features,
                                                 hidden_size,
                                                 output_size,
                                                 x_mean,
                                                 num_layers,
                                                 learning_rate)

train_gru()
