
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from models import logistic_regression, naive_bayes
from src.create_folds import run
from src.data_loader import load_data
from src.preprocessing import preprocess_data


def train():
    _nb_fold = 5
    _data_abs_path = "/home/reza/Documents/kaggle/kaggle-nlp-disaster-tweets/input/train.csv"

    # load data
    df = load_data(_data_abs_path)

    # peprocessing
    df = preprocess_data(df)

    # create fold
    df = run(df, _nb_fold)

    for fold_ in range(_nb_fold):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        # count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

        x_train = tfidf_vec.fit_transform(train_df.text)
        x_test = tfidf_vec.transform(test_df.text)

        model = logistic_regression(max_iter=500)

        model.fit(x_train, train_df.target)
        preds = model.predict(x_test)

        accuracy = metrics.accuracy_score(test_df.target, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")


if __name__ == "__main__":
    train()
