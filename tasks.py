# third party
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForQuestionAnswering

# first party
from data import SquadDataset, load_squad_data
from utils import trainer, question_and_answer_evaluation

# TODO: set up logger


def train_model():
    """
    Function to train the QA model
    """
    # model_name
    model_name = "roberta-base"
    batch_size = 16
    path = "./model_weights/text-mining-titans-roberta-qa.pt"

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
    training_data, validation_datasets = load_squad_data(tokenizer)
    validation_data = validation_datasets["validation_data"]

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
    # TODO: Save losses
    for epoch in epochs:
        losses = trainer(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # Evaluate the model
        evaluation_metrics = question_and_answer_evaluation(
            model=model,
            data_loader=validation_loader,
            validation_datasets=validation_datasets,
            device=device,
        )
        print(evaluation_metrics)

    # After training save the model
    torch.save(model.state_dict(), path)
