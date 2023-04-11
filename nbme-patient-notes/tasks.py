# third party
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

# first party
from data import NBMEDataset, load_training_data
from model import NBMEModel
from utils import (
    Configuration,
    create_labels_for_scoring,
    get_character_probabilities,
    get_predictions,
    get_thresholded_sequences,
    get_score,
    training_function,
    validation_function,
)


def run_model_pipeline():
    """
    Function that runs the NBME pipeline
    """
    pass
