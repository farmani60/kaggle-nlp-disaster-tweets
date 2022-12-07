
import io

import numpy as np
from nltk.tokenize import word_tokenize


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def sentence_to_vec(sentence, embedding_dict, stop_words, tokenizer):
    words = str(sentence).lower()
    words = tokenizer(words)
    words = [w for w in words if w not in stop_words]

    embeddings = []
    for w in words:
        if w in embedding_dict:
            embeddings.append(embedding_dict[w])

    if len(embeddings) == 0:
        return np.zeros(300)

    embeddings = np.array(embeddings)
    embedding_vector = embeddings.sum(axis=0)
    return embedding_vector / np.sqrt((embedding_vector ** 2).sum())


def vectorize(df):
    df = df.sample(frac=1).reset_index(drop=True)
    print("Loading embeddings...")
    embeddings = load_vectors(fname="/home/reza/Documents/kaggle/kaggle-nlp-disaster-tweets/wiki-news-300d-1M.vec")

    vectors = []
    for text_ in df.text.values:
        vectors.append(sentence_to_vec(sentence=text_, embedding_dict=embeddings, stop_words=[], tokenizer=word_tokenize))

    vectors = np.array(vectors)
    return vectors
