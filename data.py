# third party
import pandas as pd
from datasets import load_dataset


def load_squad_data():  # noqa
    """
    Function to load the squad v2 dataset from huggingface 🤗
    """
    dataset = load_dataset("squad_v2")

    # Create the training dataset
    training_data = pd.DataFrame(dataset["train"])

    # Create the validation dataset
    validation_data = pd.DataFrame(dataset["validation"])

    return training_data, validation_data
