# first party
import glob
import os

# third party
import numpy as np
import pandas as pd
from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,
    Laplace,
    StupidBackoff,
    WittenBellInterpolated,
)
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.util import bigrams, trigrams
from sklearn.model_selection import train_test_split

fpath = "ngram_authorship_train"


def load_data(split="paragraph"):
    """
    Function to load the data from the author text files. The way
    we will load in the data is to treat each paragraph as a full
    text
    """
    # Get the different file names
    data_dir = "./data"
    path = os.path.join(data_dir, fpath, "*")

    data_list = []
    authorlist = []
    for filename in glob.glob(path):
        # For now we will use the utf-8 version
        if "utf8" in filename:
            # Get the author name
            author_name = filename.split("/")[-1].split("_")[0]
            authorlist.append(author_name)

            # Read in the data
            data = open(filename).read().splitlines()

            # Combine the paragrahs
            if split == "paragraph":
                data = [line if line != "" else "\n" for line in data]
                data = " ".join(data).split("\n")

            elif split == "sentence":
                data = [line for line in data if line != ""]
                data = " ".join(data)
                data = sent_tokenize(data)

            data = [(line.lower().strip(), author_name) for line in data]
            data = [(line, author) for line, author in data if line != ""]
            data_list.append(data)

    # Combine all of the data
    data_list = [item for sublist in data_list for item in sublist]

    # Split the data into training and validation sets
    train_data, test_data = train_test_split(
        data_list, test_size=0.10, random_state=2023
    )

    # Put the data in a data frame
    columns = ["text", "author"]
    train_data = pd.DataFrame(train_data, columns=columns)
    test_data = pd.DataFrame(test_data, columns=columns)

    return train_data, test_data, authorlist


def train(train_data, authorlist, model_type="MLE", n=3, **kwargs):
    """
    Function that will create a model of text by an author. We
    can put in the training data and the author names. We can also specify
    the model of interest.
    """
    # Set up the tokenizer we will clean punctuation with the tokenizer
    tokenizer = RegexpTokenizer(r"\w+")

    # model dictionary
    models = {}

    # Extract the data and build the model
    for author_name in authorlist:
        # Initalize model - needs to be reinitialized for every author
        # or else we have the same model
        if model_type == "MLE":
            model = MLE(order=n)

        elif model_type == "Laplace":
            model = Laplace(order=n)

        elif model_type == "AbsoluteDiscountingInterpolated":
            model = AbsoluteDiscountingInterpolated(order=n, **kwargs)

        elif model_type == "KneserNeyInterpolated":
            model = KneserNeyInterpolated(order=n)

        elif model_type == "WittenBellInterpolated":
            model = WittenBellInterpolated(order=n)

        elif model_type == "StupidBackoff":
            model = StupidBackoff(order=n)

        else:
            raise ValueError("model does not exists!")

        # Get the data
        author_mask = train_data["author"] == author_name
        author_data = train_data.loc[author_mask].reset_index(drop=True)

        # Extract the text data & clean + tokenize
        text_data = author_data["text"].tolist()
        text_data = [tokenizer.tokenize(line) for line in text_data]

        # Create the training data and vocab
        train, vocab = padded_everygram_pipeline(n, text_data)

        # Fit the model
        model.fit(train, vocab)

        # Add the model to dictionary with authorship
        models[author_name] = model

    # Add model attributes
    models["attributes"] = {"tokenizer": tokenizer, "n": n}

    return models


def predict(test_data, authorlist, models):
    """
    Function that will predict the authorship based on the models
    from training.
    """
    # Extract the attributes from training
    tokenizer = models["attributes"]["tokenizer"]
    n = models["attributes"]["n"]

    # Remove attributes
    models = {k: v for k, v in models.items() if k != "attributes"}
    print(models)

    # Create the text & labels
    test_text = test_data["text"].tolist()
    test_text = [tokenizer.tokenize(line) for line in test_text]

    # Map test labels to author index
    labels = test_data["author"].tolist()
    mapping = dict(zip(authorlist, range(len(authorlist))))
    labels = list(map(lambda x: mapping[x], labels))

    # Use the n_gram creator
    if n == 2:
        method = bigrams

    elif n == 3:
        method = trigrams

    else:
        raise ValueError("This n is not an option!")

    # Iterate over the test data and the models
    predictions = []
    for text in test_text:
        # Set up the bigrams / trigrams. Note that this n in
        # pad_both_ends is not the same n used to define the
        # bigrams / trigrams
        text = list(method(pad_both_ends(text, n=2)))

        # Go through the models
        author_perplexity = []
        for author_name in authorlist:
            model = models[author_name]
            score = model.perplexity(text)

            # Append the score
            author_perplexity.append(score)

        # Get the argmin for prediction
        author_index = np.argmin(author_perplexity)
        predictions.append(author_index)

    return predictions, labels


if __name__ == "__main__":
    pass
