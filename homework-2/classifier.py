# first party
import os
import sys

# third party
import numpy as np
import pandas as pd
from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,
    Laplace,
    StupidBackoff,
)
from nltk.lm.api import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm.smoothing import WittenBell
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.util import bigrams, trigrams
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set up filepath
fpath = "ngram_authorship_train"


class InterpolatedLanguageModel(LanguageModel):
    """Logic common to all interpolated language models.
    The idea to abstract this comes from Chen & Goodman 1995.
    Do not instantiate this class directly!
    """

    def __init__(self, smoothing_cls, order, **kwargs):  # noqa
        params = kwargs.pop("params", {})
        super().__init__(order, **kwargs)
        self.estimator = smoothing_cls(self.vocab, self.counts, **params)

    def unmasked_score(self, word, context=None):  # noqa
        near_zero = 1e-10
        if not context:
            # The base recursion case: no context, we only have a unigram.
            return self.estimator.unigram_score(word)
        if not self.counts[context]:
            # It can also happen that we have no data for this context.
            # In that case we defer to the lower-order ngram.
            # This is the same as setting alpha to 0 and gamma to 1.
            alpha, gamma = near_zero, 1
        else:
            alpha, gamma = self.estimator.alpha_gamma(word, context)
            alpha = alpha + near_zero

        return alpha + gamma * self.unmasked_score(word, context[1:])

    def entropy(self, text_ngrams):
        """Calculate cross-entropy of model for given evaluation text.
        :param Iterable(tuple(str)) text_ngrams: A sequence of ngram tuples.
        :rtype: float
        """
        return -1 * np.mean(
            [self.logscore(ngram[-1], ngram[:-1]) for ngram in text_ngrams],
        )


class WittenBellInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Witten-Bell smoothing."""

    def __init__(self, order, **kwargs):  # noqa
        super().__init__(WittenBell, order, **kwargs)


def prepare_data(filename, author_name, split):
    """
    Function that prepares the data.
    """
    # Read in the data
    data = open(filename).read().splitlines()

    # Combine the paragrahs
    # Note: After some thought I don't think this is the right interpretation
    # afterall. However, I will keep the code for now
    if split == "paragraph":
        data = [line if line != "" else "\n" for line in data]
        data = " ".join(data).split("\n")

    elif split == "sentence":
        data = [line for line in data if line != ""]
        data = " ".join(data)
        data = sent_tokenize(data)

    data = [(line.lower().strip(), author_name) for line in data]
    data = [(line, author) for line, author in data if line != ""]

    return data


def load_data(filenames, test_filenames=None, split="sentence"):
    """
    Function to load the data from the author text files. The way
    we will load in the data is to treat each paragraph as a full
    text
    """
    # Logging
    print("Setting up the data")

    data_list = []
    authorlist = []
    for filename in filenames:
        # For now we will use the utf-8 version
        if "utf8" in filename:
            # Get the author name
            author_name = filename.split("/")[-1].split("_")[0]
            authorlist.append(author_name)

            # Read in the data
            data = prepare_data(filename, author_name=author_name, split=split)
            data_list.append(data)

    # Combine all of the data
    data_list = [item for sublist in data_list for item in sublist]

    # Split the data into training and validation sets if there is not a test
    # set present
    columns = ["text", "author"]
    if test_filenames is None:
        print("Splitting into training and development data sets ...")
        train_data, test_data = train_test_split(
            data_list, test_size=0.10, random_state=2023
        )

        # Put the data in a data frame
        train_data = pd.DataFrame(train_data, columns=columns)
        test_data = pd.DataFrame(test_data, columns=columns)

    else:
        print("Training on full data and using outside testing data ...")
        train_data = pd.DataFrame(data_list, columns=columns)

        # If there is a test file then we need to prepare this data as well
        print("Preparing the testing data ...")
        test_data_list = []
        for test_filename in test_filenames:
            # Get the author name
            author_name = test_filename.split("/")[-1].split("_")[0]

            # Prepare the data
            test_data = prepare_data(
                test_filename, author_name=author_name, split=split
            )
            test_data_list.append(test_data)

        # Flatten data
        test_data_list = [item for sublist in test_data_list for item in sublist]
        test_data = pd.DataFrame(test_data_list, columns=columns)

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
    print("training LMs ...")
    for author_name in authorlist:
        print(f"training LM for {author_name}")
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
            model = StupidBackoff(order=n, **kwargs)

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

    # Create the text & labels
    print("Extracting dev set data ...")
    test_text = test_data["text"].tolist()
    test_text = [tokenizer.tokenize(line) for line in test_text]

    # Map test labels to author index
    labels = test_data["author"].tolist()
    mapping = dict(zip(authorlist, range(len(authorlist))))
    labels = list(map(lambda x: mapping[x], labels))

    # Iterate over the test data and the models
    predictions = []
    print("Predicting ...")
    for text in test_text:
        # Set up the bigrams / trigrams. Note that this n in
        # pad_both_ends is not the same n used to define the
        # bigrams / trigrams
        bigram_text = list(bigrams(pad_both_ends(text, n=2)))
        trigram_text = list(trigrams(pad_both_ends(text, n=4)))

        # Go through the models
        author_perplexity = []
        for author_name in authorlist:
            model = models[author_name]

            # Combine bigrams + trigrams perplexity
            score_bigrams = model.perplexity(bigram_text)
            score_trigrams = model.perplexity(trigram_text)
            score = (score_bigrams + score_trigrams) / 2.0

            # Append the score
            author_perplexity.append(score)

        # Get the argmin for prediction
        author_index = np.argmin(author_perplexity)
        predictions.append(author_index)

    # Turn labels and predictions into numpy array
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    # Calculate the overall accuracy and the author
    # specific accuracy
    print("Results on dev / test set")
    for index, author_name in enumerate(authorlist):
        # Get the author specific labels
        mask = labels == index

        # Get the data
        author_predictions = predictions[mask]
        author_labels = labels[mask]

        # Get the numerator and demoninator
        num = (author_labels == author_predictions).sum()
        den = len(author_labels)
        score = num / den

        print(f"Accuracy for {author_name} = {score * 100}%")

    # Get overall accuracy
    overall_score = accuracy_score(labels, predictions)
    print(f"Overall accuracy = {overall_score * 100}%")

    return predictions, labels


def main_task(filenames, test_filenames=None):
    """
    Task that performs the classification and text generation
    """
    MODEL_TYPE = "WittenBellInterpolated"
    n = 3

    train_data, test_data, authorlist = load_data(
        filenames=filenames, test_filenames=test_filenames
    )

    # Gathering the models
    models = train(train_data, authorlist, model_type=MODEL_TYPE, n=n)

    # Get the predictions on the dev set
    predictions, labels = predict(test_data, authorlist, models)


if __name__ == "__main__":
    authorlist_filename_index = sys.argv.index("authorlist.txt")
    authorlist_filename = sys.argv[authorlist_filename_index]
    filenames = open(authorlist_filename).read().splitlines()

    if "-test" not in sys.argv:
        test_filenames = None

    else:
        test_filename_index = sys.argv.index("testfile.txt")
        test_filename = sys.argv[test_filename_index]
        test_filenames = open(test_filename).read().splitlines()

    # Start main task
    main_task(filenames=filenames, test_filenames=test_filenames)
