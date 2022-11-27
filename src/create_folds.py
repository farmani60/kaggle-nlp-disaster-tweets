
from sklearn import model_selection


def run(df, nb_folds=5):

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.target

    k_fold = model_selection.StratifiedKFold(n_splits=nb_folds)

    for f, (t_, v_) in enumerate(k_fold.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    return df
