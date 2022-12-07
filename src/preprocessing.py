
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def text_preprocessing(text):
    # Convert text into lowercase
    text = text.lower()

    # remove link from the text
    text = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        " ", text)

    # Tokenize text into list
    tokenize_text = nltk.word_tokenize(text)

    # remove Stopwords
    text_without_stopwords = [i for i in tokenize_text if i not in stopwords.words('english')]

    # Remove Punctuation
    text_without_punc = [i for i in text_without_stopwords if i not in string.punctuation]

    ps = PorterStemmer()
    # fetch only alphanumeric values and apply stemming on that word
    transformed_text = [ps.stem(i) for i in text_without_punc if i.isalnum() == True]

    return " ".join(transformed_text)


def preprocess_data(df):

    # drop redundant columns
    df.drop(columns=["id", "location"], inplace=True)

    # drop null values
    df.dropna(inplace=True)

    # drop duplicated values
    df.drop_duplicates(keep="first", inplace=True)

    # Merge Keyword and Text Column and create single Content Column.
    df["content"] = df["keyword"] + " " + df["text"]

    # Let's Apply This Transformation Function on Our Content Column
    df['text'] = df['content'].apply(text_preprocessing)

    # Drop title author and old content column
    final_df = df.drop(['keyword', 'content'], axis=1)

    return final_df
