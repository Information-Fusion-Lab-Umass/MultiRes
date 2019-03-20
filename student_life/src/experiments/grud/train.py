"""
Script for training StudentLife on GRU-D
"""
import torch

from src.bin import plotting
from src.bin import scoring
from src.experiments.grud import helper
from src.bin import trainer
from src.bin import tensorify
from src import definitions
from src.utils import read_utils as reader
from src.models.grud import GRUD

GRU_D_CONFIG = reader.read_yaml(definitions.MODEL_CONFIG_FILE_PATH)['gru_d']
CLUSTER_MODE = reader.read_yaml(definitions.FEATURE_CONFIG_FILE_PATH)['cluster_mode']

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
    n_epochs = GRU_D_CONFIG['epochs']

    # CUDA Enabled.
    if torch.cuda.device_count() > 0:
        cuda_enabled = True
    else:
        cuda_enabled = False

    print("CUDA Status: ", cuda_enabled, end="\n\n")

    # Data to tensors
    data = tensorify.tensorify_data_gru_d(data, cuda_enabled)
    model, criterion, optimizer = initialize_gru(num_features,
                                                 hidden_size,
                                                 output_size,
                                                 x_mean,
                                                 num_layers,
                                                 learning_rate)

    loss_over_epochs, scores_over_epochs = plotting.get_empty_stat_over_n_epoch_dictionaries()

    for epoch in range(1, n_epochs + 1):
        print("xxxxxxxxxxxxxx epoch: {} xxxxxxxxxxxxxx".format(epoch))
        train_loss, train_labels, train_preds = trainer.evaluate_set(data, 'train_ids', model, criterion, optimizer)
        val_loss, val_labels, val_preds = trainer.evaluate_set(data, 'val_ids', model, criterion)
        test_loss, test_labels, test_preds = trainer.evaluate_set(data, 'train_ids', model, criterion)

        loss_over_epochs['train_loss'].append(train_loss)
        loss_over_epochs['val_loss'].append(val_loss)
        loss_over_epochs['test_loss'].append(test_loss)

        train_scores, val_scores, test_scores = scoring.get_precission_recall_f_scores(train_labels=train_labels,
                                                                                       train_preds=train_preds,
                                                                                       val_labels=val_labels,
                                                                                       val_preds=val_preds,
                                                                                       test_labels=test_labels,
                                                                                       test_preds=test_preds)

        scores_over_epochs['train_scores'].append(train_scores)
        scores_over_epochs['val_scores'].append(val_scores)
        scores_over_epochs['test_scores'].append(test_scores)

        # Plot every 10 Epochs.
        if epoch % 10 == 0 and not CLUSTER_MODE:
            plotting.plot_score_over_n_epochs(scores_over_epochs, score_type='f1', file_path="./gru_d/f1_at_epoch_{}".format(epoch))
            plotting.plot_loss_over_n_epochs(loss_over_epochs, file_path="./gru_d/loss_at_epoch_{}".format(epoch))

        print("Train Loss: {} Val Loss: {} Test Loss: {}".format(loss_over_epochs['train_loss'],
                                                                 loss_over_epochs['val_loss'],
                                                                 loss_over_epochs['test_loss']))
        val_precision, val_recall, val_f1, _ = val_scores
        print("Precision: {} Recall: {} F1 Score: {}".format(val_precision, val_recall, val_f1))


train_gru()
