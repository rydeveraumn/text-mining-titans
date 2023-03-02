# CSCI Homework 2 README 📚


## Running Code 💻

### `authorlist.txt` and `testfile.txt`

We held a lot of conversations in the group slack and just for good measure we wanted to place the assumptions that we have for the contents of the input files. We assume that `authorlist.txt` is a text file with a list of paths to each author's text file on the local comptuter it will be run on. For example, here was the content of our `authorlist.txt` and `testfile.txt`

```
authorlist.txt

./data/ngram_authorship_train/austen_utf8.txt
./data/ngram_authorship_train/dickens_utf8.txt
./data/ngram_authorship_train/tolstoy_utf8.txt
./data/ngram_authorship_train/wilde_utf8.txt
./data/ngram_authorship_train/austen.txt
./data/ngram_authorship_train/dickens.txt
./data/ngram_authorship_train/tolstoy.txt
./data/ngram_authorship_train/wilde.txt
```

```
testfile.txt

./data/ngram_authorship_test/austen_test_sents_utf8.txt
./data/ngram_authorship_test/dickens_test_sents_utf8.txt
./data/ngram_authorship_test/tolstoy_test_sents_utf8.txt
./data/ngram_authorship_test/wilde_test_sents_utf8.txt
```

### Command line 🐍

To run our code we ran it just as was advertised in the homework PDF. We used python version `version=3.6+`

Without `-test` flag

```
python classifier.py authorlist.txt
```

With `-test` flag

```
python classifier.py authorlist.txt -test testfile.txt
```