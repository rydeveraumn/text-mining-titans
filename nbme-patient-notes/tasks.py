# third party
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

# first party
from data import NBMEDataset, build_pseudo_data, load_training_data
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
    # Load in the data
    config = Configuration()
    data = load_training_data(config=config)
    pseudo_data = build_pseudo_data().sample(frac=0.01).reset_index(drop=True)
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Get training and validation data
    print("Load and build data loaders for training and evaluation")
    train_df = data.loc[data["fold_number"] != 4].reset_index(drop=True)
    valid_df = data.loc[data["fold_number"] == 4].reset_index(drop=True)
    valid_patient_notes_texts = valid_df["pn_history"].values
    valid_labels = valid_df["location"].apply(create_labels_for_scoring)

    # Pseduo data
    pseudo_patient_notes_texts = pseudo_data["pn_history"].values

    # Create the datasets and data loaders
    training_dataset = NBMEDataset(train_df, config)
    valid_dataset = NBMEDataset(valid_df, config)
    pseudo_train_dataset = NBMEDataset(pseudo_data, config, build_label=False)

    # Training loaders
    train_loader = DataLoader(
        training_dataset, batch_size=8, shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=False
    )
    pseudo_loader = DataLoader(
        pseudo_train_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Get the loss and optimizers and model
    model = NBMEModel(config=config)
    model = model.to(device=device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # TODO: still need to make training function and validation
    print("Train model")
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

        # Get the probability outputs
        predictions, labels = validation_function(config, valid_loader, model, device)

        # Reshape the predictions and labels
        samples = len(valid_df)
        predictions = predictions.reshape((samples, config.max_length))
        labels = labels.reshape((samples, config.max_length))

        # Get character probabilities
        character_probabilities = get_character_probabilities(
            valid_patient_notes_texts, predictions, config
        )

        # Get results
        results = get_thresholded_sequences(character_probabilities)
        preds = get_predictions(results)
        score = get_score(valid_labels, preds)
        print(score)

        # Save the model after each epoch
        PATH = f"./models/deberta_v3_base_cpt_epoch_{epoch}.pt"
        torch.save(model.state_dict(), PATH)

    print("Build Pseudo Labels and save output")
    pseudo_data = build_pseudo_predictions(
        config, model, pseudo_data, pseudo_loader, device
    )
    pseudo_data["fold_number"] = 0
    pseudo_data["id"] = (
        pseudo_data["pn_num"].astype(str) + "_" + pseudo_data["feature_num"].astype(str)
    )
    pseudo_data.to_csv("./nbme_data/pseudo_train.csv", index=False)
