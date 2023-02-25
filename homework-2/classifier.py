# first party
import glob
import io
import os
import sys

# third party
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.lm import (
    MLE,
    AbsoluteDiscountingInterpolated,
    KneserNeyInterpolated,
    Laplace,
    StupidBackoff,
    WittenBellInterpolated,
)
from nltk.lm.preprocessing import (
    pad_both_ends,
    padded_everygram_pipeline,
    padded_everygrams,
)
from nltk.util import bigrams

fpath = "ngram_authorship_train"


def load_data():
    """
    Function to load the data from the author text files. The way
    we will load in the data is to treat each paragraph as a full
    text
    """
    # Get the different file names
    data_dir = "./data"
    path = os.path.join(data_dir, fpath, "*")

    data_list = []
    for filename in glob.glob(path):
        # For now we will use the utf-8 version
        if "utf8" in filename:
            # Read in the data
            data = open(filename).read().splitlines()

            # Combine the paragrahs
            data = [line if line != "" else "\n" for line in data]
            data = "".join(data).split("\n")
            data_list.append(data)

    return pd.concat(data_list)


def train(authorlist):
    with open(authorlist) as file:
        file_names = [line.strip() for line in file]

    for fname in file_names:
        with open(os.path.join(fpath, fname)) as file:
            # get all text
            text = file.read()
            print(fname)

            # tokenize text
            tokenized_text = []
            for sent in sent_tokenize(text):
                tokenized_text.append(list(map(str.lower, word_tokenize(sent))))

            # train-test split
            n = len(tokenized_text)
            split = (int)(0.9 * n)
            train_text = tokenized_text[:split]
            test_text = tokenized_text[split:]

            # generate ngrams
            train, vocab = padded_everygram_pipeline(3, train_text)

            # build model
            model = MLE(3)
            model.fit(train, vocab)

            # calc perplexity
            perplexity = 0
            for sent in test_text:
                sent = list(bigrams(pad_both_ends(sent, n=2)))
                perp = model.perplexity(sent)
                perplexity += perp
                print(perp)

            perplexity /= len(test_text)
            print(perplexity)


def classify(authorlist, testfile):
    with open(authorlist) as file:
        file_names = [line.strip() for line in file]

    models = []
    for fname in file_names:
        with open(os.path.join(fpath, fname)) as file:
            # get all text
            text = file.read()

            # tokenize text
            tokenized_text = []
            for sent in sent_tokenize(text):
                tokenized_text.append(list(map(str.lower, word_tokenize(sent))))

            # generate ngrams
            train, vocab = padded_everygram_pipeline(3, tokenized_text)

            # build model
            model = KneserNeyInterpolated(3)
            model.fit(train, vocab)
            models.append(model)

    # classify testfile
    with open(testfile) as file:
        for line in file:
            line = line.strip()
            test = list(bigrams(pad_both_ends(line, n=2)))

            # calc perplexity for each model
            min_perplexity = -1
            min_idx = 0
            for i, model in enumerate(models):
                perplexity = model.perplexity(test)
                if perplexity == -1 or perplexity < min_perplexity:
                    min_perplexity = perplexity
                    min_idx = i
            print(file_names[i].split("_")[0].split(".")[0])


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        print("Invalid arguments")
    if len(args) == 2:
        train(args[1])
    else:
        classify(args[1], args[3])
