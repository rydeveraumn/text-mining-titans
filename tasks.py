# stdlib
import logging

# third party
import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    TrainingArguments,
)

# first party
from data import SquadDataset, load_examples
from utils import CustomTrainer, build_incorrect_samples, evaluate

# Create logger
logger = logging.getLogger(__name__)


def train_model(create_training_data=False):
    """
    Function to train the QA model
    """
    # configuration
    model_name = "roberta-base"
    batch_size = 4
    data_dir = "./squad_data"
    train_data_file = "train-v2.0.json"
    validation_data_file = "dev-v2.0.json"

    # Set up the device
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")  # noqa
    )  # noqa

    # Setup the model and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=True, use_fast=False
    )
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # Create features for SQUAD-V2 data
    if create_training_data:
        train_dataset = load_examples(
            data_dir=data_dir,
            data_file=train_data_file,
            tokenizer=tokenizer,
            evaluate=False,
            output_examples=False,
        )
        torch.save(train_dataset, "train_dataset.pt")
    else:
        train_dataset = torch.load("train_dataset.pt")

    # Create the validation set
    validation_dataset, validation_examples, validation_features = load_examples(
        data_dir=data_dir,
        data_file=validation_data_file,
        tokenizer=tokenizer,
        evaluate=True,
        output_examples=True,
    )
    validation_datasets = (validation_dataset, validation_examples, validation_features)

    # Create the train test split
    training_dataset = train_dataset[: (len(train_dataset) - 10000)]
    test_dataset = train_dataset[(len(train_dataset) - 10000) :]

    # Convert the training and test data into a Dataset
    training_dataset = SquadDataset(training_dataset, mode="training")
    test_dataset = SquadDataset(test_dataset, mode="training")

    # Setup training aruguments and trainer
    data_collator = DefaultDataCollator()

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="model_weights",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    # Setup trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Get the evaluation for the final checkpoint
    results, examples, predictions = evaluate(
        output_dir="results",
        model=model,
        tokenizer=tokenizer,
        device=device,
        datasets=validation_datasets,
        prefix="final_evaluation",
    )

    # log the results
    logger.info(f"Results : {results}")

    # Create the loss plot
    losses = pd.read_csv("./results/train-step-losses.csv")

    # Create the figure and axes and build plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    losses.rolling(20).mean().plot(ax=ax)
    ax.set_title("Training Loss Per Step")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.grid()

    # Save figure
    fig.tight_layout()
    fig.savefig("./results/training_loss.png")

    # For the final output build the incorrect samples
    build_incorrect_samples(examples, predictions)


if __name__ == "__main__":
    train_model()
