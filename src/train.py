
from nltk.tokenize import word_tokenize
from sklearn import linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer

from create_folds import run


def train():
    _nb_fold = 5

    df = run(_nb_fold)

    for fold_ in range(_nb_fold):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

        x_train = count_vec.fit_transform(train_df.text)
        x_test = count_vec.transform(test_df.text)

        model = linear_model.LogisticRegression(max_iter=500)

        model.fit(x_train, train_df.target)
        preds = model.predict(x_test)

        accuracy = metrics.accuracy_score(test_df.target, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")


if __name__ == "__main__":
    train()
