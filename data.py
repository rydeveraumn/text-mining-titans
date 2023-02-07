# third party
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def extract_answer_start(x):  # noqa
    """
    Function that extracts the answer start key from the squad
    v2 dataset
    """
    answer_start = x.get("answer_start")

    # If the length is zero then return np.nan
    if len(answer_start) == 0:
        return -1

    return answer_start[0]


def load_squad_data():  # noqa
    """
    Function to load the squad v2 dataset from huggingface 🤗
    """
    dataset = load_dataset("squad_v2")

    # Create the training dataset
    training_data = pd.DataFrame(dataset["train"])
    training_data["answers_text"] = training_data["answers"].apply(
        lambda x: "".join(x.get("text"))
    )
    training_data["answers_start"] = training_data["answers"].apply(
        extract_answer_start
    )

    # Create the validation dataset
    validation_data = pd.DataFrame(dataset["validation"])
    validation_data["answers_text"] = validation_data["answers"].apply(
        lambda x: "".join(x.get("text"))
    )
    validation_data["answers_start"] = validation_data["answers"].apply(
        extract_answer_start
    )

    return training_data, validation_data


def prepare_inputs(tokenizer, text):
    """
    Function that will prepare the input ids and attention masks
    for the task
    """
    data = tokenizer(
        text,
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        pad_to_max_length=True,
    )

    # Input ids
    input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
    attention_masks = torch.tensor(data["attention_mask"], dtype=torch.long)

    return input_ids, attention_masks


class SquadDataset(Dataset):
    """
    Class that builds the encodings for the tokens
    from the Squad V2 dataset
    """

    def __init__(self, df, tokenizer):  # noqa
        self.df = df
        self.context = df["context"].values
        self.questions = df["question"].values
        self.answers_text = df["answers_text"].values
        self.answers_start = df["answers_start"].values
        self.tokenizer = tokenizer

    def __len__(self):  # noqa
        return len(self.df)

    def __getitem__(self, idx):  # noqa
        # We need to get the outputs for all of the texts
        # Contexts
        context_inputs, context_attention_masks = prepare_inputs(
            self.tokenizer, self.context[idx]
        )

        # Questions
        question_inputs, question_attention_masks = prepare_inputs(
            self.tokenizer, self.questions[idx]
        )

        # Answers / labels
        answers_inputs, answers_attention_masks = prepare_inputs(
            self.tokenizer, self.answers_text[idx]
        )

        # Answers start for model training / used as labels
        # There were a lot of start values missing - we will need to figure out what
        # we want to do with that data
        answers_start = torch.tensor(self.answers_start[idx], dtype=torch.long)

        # Gather outputs
        outputs = {
            "context_inputs": context_inputs,
            "context_attention_masks": context_attention_masks,
            "question_inputs": question_inputs,
            "question_attention_masks": question_attention_masks,
            "answers_inputs": answers_inputs,
            "answers_attention_masks": answers_attention_masks,
            "answers_start": answers_start,
        }

        return outputs
