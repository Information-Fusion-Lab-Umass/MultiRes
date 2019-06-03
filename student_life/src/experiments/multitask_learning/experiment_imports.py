"""
Imports required to run experiments.
This script keeps the experiments script brief.
"""

import os

import torch
import tqdm

from statistics import mean as list_mean
from sklearn import metrics

from torch import nn
from copy import deepcopy
from src import definitions
from src.bin import tensorify
from src.bin import statistics
from src.bin import trainer
from src.data_manager import cross_val
from src.data_manager import student_life_var_binned_data_manager as data_manager

# Models
from src.models import autoencoder_classifier
from src.models.multitask_learning import multitask_autoencoder
from src.models.multitask_learning import multitask_lstm
from src.models import model_config_loader

# Utils.
from src.utils import write_utils
from src.utils.read_utils import read_pickle
from src.utils import data_conversion_utils as conversions
from src.utils import print_utils
from src.bin import checkpointing