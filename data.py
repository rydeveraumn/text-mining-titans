# third party
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor

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


def load_examples(
    data_dir, data_file, tokenizer, evaluate=False, output_examples=False
):
    processor = SquadV2Processor()
    if evaluate:
        examples = processor.get_dev_examples(data_dir, filename=data_file)
    else:
        examples = processor.get_train_examples(data_dir, filename=data_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=not evaluate,
        return_dataset="pt",
        threads=1,
    )

    if output_examples:
        return dataset, examples, features

    return dataset


class SquadDataset(Dataset):
    """
    Class that builds the encodings for the tokens
    from the Squad V2 dataset
    """

    def __init__(self, data, mode="training"):  # noqa
        self.data = data
        self.input_ids = data[0].tolist()
        self.attention_mask = data[1].tolist()
        self.mode = mode

        # If mode is training then we need to extract
        # the start and stop positions
        if self.mode == "training":
            # Add the start and end positions
            self.start_positions = data[3].tolist()
            self.end_positions = data[4].tolist()

    def __len__(self):  # noqa
        return len(self.input_ids)

    def __getitem__(self, idx):  # noqa
        # We need to get the outputs for all of the texts
        # Contexts
        # Set up input ids to be torch tensors
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)

        # Gather outputs
        outputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.mode == "training":
            start_positions = torch.tensor(self.start_positions[idx], dtype=torch.long)
            end_positions = torch.tensor(self.end_positions[idx], dtype=torch.long)

            outputs["start_positions"] = start_positions
            outputs["end_positions"] = end_positions

        return outputs
