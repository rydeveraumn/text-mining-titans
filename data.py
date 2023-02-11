# third party
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

# first party
from utils import prepare_train_features, prepare_validation_features


def load_squad_data(tokenizer):  # noqa
    """
    Function to load the squad v2 dataset from huggingface 🤗
    """
    dataset = load_dataset("squad_v2")
    raw_validation_data = dataset["validation"].select(range(100))

    # Inputs
    max_length = 384
    doc_stride = 128

    # Set up partial functions - training features
    training_features = partial(
        prepare_train_features,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
    )

    # Set up partial function - validation features
    validation_features = partial(
        prepare_validation_features,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
    )

    # Create the training dataset
    training_data = (
        dataset["train"]
        .select(range(100))
        .map(
            training_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
    )

    # Create the validation training st
    validation_data = raw_validation_data.map(
        validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    # Combine the validation data for training
    # and evaluation
    validation_datasets = {
        "raw_validation_data": raw_validation_data,
        "validation_data": validation_data,
    }

    return training_data, validation_datasets


class SquadDataset(Dataset):
    """
    Class that builds the encodings for the tokens
    from the Squad V2 dataset
    """

    def __init__(self, data, mode="training"):  # noqa
        self.data = data
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.mode = mode

        # If mode is training then we need to extract
        # the start and stop positions
        if self.mode == "training":
            available_features = data.features.keys()
            if "start_positions" not in available_features:
                raise ValueError(
                    "Training data does not have the correct format!"
                )  # noqa

            self.start_positions = data["start_positions"]
            self.end_positions = data["end_positions"]

    def __len__(self):  # noqa
        return len(self.data)

    def __getitem__(self, idx):  # noqa
        # We need to get the outputs for all of the texts
        # Contexts
        # Set up input ids to be torch tensors
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(
            self.attention_mask[idx], dtype=torch.long
        )  # noqa

        # Gather outputs
        outputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.mode == "training":
            start_positions = torch.tensor(
                self.start_positions[idx], dtype=torch.long
            )  # noqa
            end_positions = torch.tensor(
                self.end_positions[idx], dtype=torch.long
            )  # noqa

            outputs["start_positions"] = start_positions
            outputs["end_positions"] = end_positions

        return outputs
