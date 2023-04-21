# first party
import ast

# third party
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset


def load_training_data(config):
    """
    Load in the train dataframe
    The training dataframe is where all of the labels are located
    in total there are 14300 annotations for 1000 patient notes
    """
    train_df = pd.read_csv("./nbme_data/train.csv")
    train_df["location"] = train_df["location"].apply(lambda x: x.replace(";", "', '"))

    # Turn the string-list annotations into a list
    train_df["annotation"] = train_df["annotation"].apply(ast.literal_eval)

    # Turn the string-location into a list
    train_df["location"] = train_df["location"].apply(ast.literal_eval)

    # Load in the patient notes which is the main text
    # that we will be modeling
    patient_notes_df = pd.read_csv("./nbme_data/patient_notes.csv")

    # Load in the features text
    features_df = pd.read_csv("./nbme_data/features.csv")

    # Merge the features
    train_df = pd.merge(
        train_df,
        features_df,
        on=["case_num", "feature_num"],
    )

    # Merge the patient notes
    train_df = pd.merge(
        train_df,
        patient_notes_df,
        on=["case_num", "pn_num"],
    )

    # Get the lenghts of the annotations list
    train_df["annotation_length"] = train_df["annotation"].apply(len)

    # For this project let's just follow some of the training
    # notebooks from Kaggle and add the validation fold number
    # in the training data directly
    group_k_fold = GroupKFold(n_splits=config.n_splits)
    groups = train_df["pn_num"].values

    # Get the folds
    fold_numbers = np.zeros(len(train_df))
    for index, (train_index, test_index) in enumerate(
        group_k_fold.split(train_df, groups=groups)
    ):
        fold_numbers[test_index] = index

    # Add the fold number
    train_df["fold_number"] = fold_numbers.astype(np.int8)

    return train_df


def build_pseudo_data():
    """
    Function to build the pseudo-data for inputs and building
    the soft targets
    """
    patient_notes_df = pd.read_csv("./nbme_data/patient_notes.csv")
    features_df = pd.read_csv("./nbme_data/features.csv")
    train_df = pd.read_csv("./nbme_data/train.csv")

    # Filter the patient notes to patients that are not in
    # the training data
    train_patients = train_df["pn_num"].unique()

    patient_mask = ~patient_notes_df["pn_num"].isin(train_patients)
    patient_notes_df = patient_notes_df.loc[patient_mask].reset_index(drop=True)

    # Subsample the patients further
    psuedo_label_pn_num = patient_notes_df["pn_num"].unique()
    psuedo_label_pn_num = np.random.choice(psuedo_label_pn_num, 10000, replace=False)

    # reduce patient mask
    reduced_patient_mask = patient_notes_df["pn_num"].isin(psuedo_label_pn_num)
    patient_notes_df = patient_notes_df.loc[reduced_patient_mask].reset_index(drop=True)

    # Psuedo-training data
    pseudo_train_df = pd.merge(
        patient_notes_df, features_df, on=["case_num"], how="inner"
    )
    # Set up fake information here for now
    pseudo_train_df["annotation_length"] = 0
    pseudo_train_df["location"] = 0

    return pseudo_train_df


def build_nbme_input(config, patient_notes_text, feature_text):
    """
    Function that builds the input for the NBME data.
    The output will be the input_ids, token_type_ids
    and attention_mask

    We will use the config which will be a class that
    has the tokenizer and max sequence length as an
    attribute

    We will use both the patient note (text) and the feature
    text
    """
    inputs = config.tokenizer(
        patient_notes_text,
        feature_text,
        add_special_tokens=True,
        max_length=config.max_length,
        padding="max_length",
        return_offsets_mapping=False,
    )

    # Turn the inputs into torch tensors
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)

    return inputs


def build_nbme_labels(config, patient_notes_text, annotation_length, location_list):
    """
    Function that builds the label for NBME. For the labels
    each row could have the same patient note but a different
    annotation

    annotation_length: is the number of annoations in the
    row

    location_list: highlights where the annotation exists
    in the text at the character level
    """
    annotation_encoding = config.tokenizer(
        patient_notes_text,
        add_special_tokens=True,
        max_length=config.max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )

    # Get the offset mapping
    offset_mapping = annotation_encoding["offset_mapping"]
    ignore_indexes = np.array(annotation_encoding.sequence_ids()) != 0
    labels = np.zeros(len(offset_mapping))
    labels[ignore_indexes] = -1

    # If the annotation_length is not equal to zero
    if annotation_length != 0:
        for i in location_list:
            start_index, end_index = i.split()
            start_index, end_index = (int(start_index) - 1, int(end_index))

            # Get the labels
            for index, (start, end) in enumerate(offset_mapping):
                if (start >= start_index) and (end <= end_index):
                    labels[index] = 1

    return torch.tensor(labels, dtype=torch.float)


class NBMEDataset(Dataset):  # noqa
    def __init__(self, data, config, build_label=True):  # noqa
        # Setup data and configuration
        self.data = data
        self.config = config
        self.build_label = build_label

        # Main inputs for the features for the model
        self.patient_notes_text = data["pn_history"].values
        self.feature_text = data["feature_text"].values
        self.annotation_length = data["annotation_length"].values
        self.location = data["location"].values

    def __len__(self):  # noqa
        return len(self.data)

    def __getitem__(self, idx):  # noqa
        # Get the data for the item
        patient_notes_text = self.patient_notes_text[idx]
        feature_text = self.feature_text[idx]
        annotation_length = self.annotation_length[idx]
        location = self.location[idx]

        # Build the inputs
        inputs = build_nbme_input(self.config, patient_notes_text, feature_text)

        # Build the labels
        if self.build_label:
            labels = build_nbme_labels(
                self.config, patient_notes_text, annotation_length, location
            )

            return inputs, labels

        return inputs
