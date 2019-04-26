import itertools

from src import definitions
from src.utils import read_utils


def get_hyper_parameter_list_for_grid_search(experiment="multitask_learner_auto_encoder"):
    experiment_config = read_utils.read_yaml(definitions.GRID_SEARCH_CONFIG_FILE_PATH)[experiment]
    hyper_parameter_list = []
    params = experiment_config.keys()

    for param in params:
        hyper_parameter_list.append(experiment_config[param])

    hyper_parameters_list = list(itertools.product(*hyper_parameter_list))
    final_hyper_prameters_list = []

    for hyper_parameters in hyper_parameters_list:
        hyper_prameters_dict = {}
        for idx, param in enumerate(params):
            hyper_prameters_dict[param] = hyper_parameters[idx]

        final_hyper_prameters_list.append(hyper_prameters_dict)

    return final_hyper_prameters_list
