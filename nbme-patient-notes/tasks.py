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
    # Load in the data
    print("Loading configuration and data")
    config = Configuration()
    data = load_training_data(config=config)
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Create train, validation & test
    print("Setting up data for training")
    train_df = data.loc[data["fold_number"] != 4].reset_index(drop=True)
    test_df = data.loc[data["fold_number"] == 4].reset_index(drop=True)
    test_patient_notes_texts = test_df["pn_history"].values
    test_labels = test_df["location"].apply(create_labels_for_scoring)

    # Now get the training and test data
    train_df = train_df.loc[train_df["fold_number"] != 3].reset_index(drop=True)
    valid_df = train_df.loc[train_df["fold_number"] == 3].reset_index(drop=True)
    valid_patient_notes_texts = valid_df["pn_history"].values
    valid_labels = valid_df["location"].apply(create_labels_for_scoring)

    # Create the datasets and data loaders
    training_dataset = NBMEDataset(train_df, config)
    valid_dataset = NBMEDataset(valid_df, config)
    test_dataset = NBMEDataset(test_df, config)

    # Training, valid and test loaders
    train_loader = DataLoader(
        training_dataset, batch_size=8, shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=False
    )

    # Get the loss and optimizers and model
    model = NBMEModel(config=config)
    model = model.to(device=device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Now set up the model to run
    print("Run and evaluate model")
    for epoch in range(1):
        training_function(
            config=config,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
        )

        # Get the probability outputs - for validation
        valid_predictions, valid_labels = validation_function(
            config, valid_loader, model, device
        )

        # Reshape the predictions and labels
        samples = len(valid_df)
        valid_predictions = valid_predictions.reshape((samples, config.max_length))
        valid_labels = valid_labels.reshape((samples, config.max_length))

        # Get character probabilities
        valid_character_probabilities = get_character_probabilities(
            valid_patient_notes_texts, valid_predictions, config
        )

        # Get results
        valid_results = get_thresholded_sequences(valid_character_probabilities)
        valid_preds = get_predictions(valid_results)
        valid_score = get_score(valid_labels, valid_preds)
        print("Scores on validation data:")
        print(valid_score)

        # Get the probability outputs - for test
        test_predictions, test_labels = validation_function(
            config, test_loader, model, device
        )

        # Reshape the predictions and labels
        samples = len(test_df)
        test_predictions = test_predictions.reshape((samples, config.max_length))
        test_labels = test_labels.reshape((samples, config.max_length))

        # Get character probabilities
        test_character_probabilities = get_character_probabilities(
            test_patient_notes_texts, test_predictions, config
        )

        # Get results
        test_results = get_thresholded_sequences(test_character_probabilities)
        test_preds = get_predictions(test_results)
        test_score = get_score(test_labels, test_preds)
        print("Scores on test data:")
        print(test_score)
