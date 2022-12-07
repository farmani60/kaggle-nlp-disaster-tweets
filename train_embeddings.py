from sklearn import metrics, model_selection

from models import logistic_regression
from src.data_loader import load_data
from src.fasttext import vectorize


def train():
    _nb_fold = 5
    _data_abs_path = "/home/reza/Documents/kaggle/kaggle-nlp-disaster-tweets/input/train.csv"

    # load data
    df = load_data(_data_abs_path)

    embeddings = vectorize(df)
    y = df.target.values

    k_fold = model_selection.StratifiedKFold(n_splits=_nb_fold)

    for fold_, (t_, v_) in enumerate(k_fold.split(X=embeddings, y=y)):
        print(f"Training fold: {fold_}")
        x_train = embeddings[t_, :]
        y_train = embeddings[t_]

        x_test = embeddings[v_, :]
        y_test = embeddings[v_, :]

        model = logistic_regression(max_iter=500)

        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, preds)
        print(f"Accuracy = {accuracy}")
        print("")


if __name__ == "__main__":
    train()
