# third party
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForQuestionAnswering

# first party
from data import SquadDataset, load_squad_dataset
from utils import trainer

# TODO: set up logger


def train_model():
    """
    Function to train the QA model
    """
    # model_name
    model_name = "roberta-base"
    batch_size = 16

    # Set up the device
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")  # noqa
    )  # noqa

    # Setup the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)
    model = model.to(device)

    # Load the training and the validation data
    training_data, validation_data = load_squad_dataset(tokenizer)

    # Setup the training and validation dataset for loading into torch
    train_dataset = SquadDataset(training_data, mode="training")
    validation_dataset = SquadDataset(validation_data, mode="validation")

    # Set up the training and validation data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # noqa
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    # Setup the optimizer that we will use to fine-tune the model
    # For most cases Adam or AdamW work fine
    lr = 2e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Number of epochs for training
    epochs = 3

    # Run the model training
    # We will certainly need to run this on a GPU we
    # can use MSI
    losses = trainer(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        epochs=epochs,  # noqa
        device=device,
    )
