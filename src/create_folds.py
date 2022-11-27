import os

import pandas as pd
from sklearn import linear_model, metrics, model_selection


def run(nb_folds=5):
    _data_abs_path = "/home/reza/Documents/kaggle/kaggle-nlp-disaster-tweets/input"

    df = pd.read_csv(os.path.join(_data_abs_path, "train.csv"))
    df = df.drop(columns=["id", "keyword", "location"], axis=1)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target

    k_fold = model_selection.StratifiedKFold(n_splits=nb_folds)

    for f, (t_, v_) in enumerate(k_fold.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    return df
