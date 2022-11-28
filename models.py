
from sklearn import linear_model, naive_bayes


def logistic_regression(max_iter=500):
    return linear_model.LogisticRegression(max_iter=max_iter)


def naive_bayes():
    return naive_bayes.MultinomialNB()
