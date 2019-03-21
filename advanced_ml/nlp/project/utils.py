import nltk
import pickle
import re
import numpy as np

# from gensim.models import KeyedVectors
from gensim.models import KeyedVectors

nltk.download("stopwords")
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    "INTENT_RECOGNIZER": "intent_recognizer.pkl",
    "TAG_CLASSIFIER": "tag_classifier.pkl",
    "TFIDF_VECTORIZER": "tfidf_vectorizer.pkl",
    "THREAD_EMBEDDINGS_FOLDER": "thread_embeddings_by_tags",
    #    "WORD_EMBEDDINGS": "word_embeddings.tsv",
    "WORD_EMBEDDINGS": "starspace_model.tsv",
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile("[/(){}\[\]\|@,;]")
    bad_symbols_re = re.compile("[^0-9a-z #+_]")
    stopwords_set = set(stopwords.words("english"))

    text = text.lower()
    text = replace_by_space_re.sub(" ", text)
    text = bad_symbols_re.sub("", text)
    text = " ".join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    embeddings = {}
    for line in open(embeddings_path, encoding="utf-8"):
        row = line.strip().split("\t")
        embeddings[row[0]] = np.array(row[1:], dtype=np.float32)
    embeddings_dim = embeddings[list(embeddings)[0]].shape[0]

    return embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    result = np.zeros(dim)
    cnt = 0
    words = question.split()
    for word in words:
        if word in embeddings:
            result += np.array(embeddings[word])
            cnt += 1
    if cnt != 0:
        result /= cnt
    return result


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, "rb") as f:
        return pickle.load(f)
