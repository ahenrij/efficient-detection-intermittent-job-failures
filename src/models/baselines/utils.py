"""TF-IDF utility functions."""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from time import perf_counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from src.models import evaluator


def vectorize(X_text, y):
    vectorizer = TfidfVectorizer(max_features=1000)
    selectkbest = SelectKBest(chi2, k=300)

    start_time = perf_counter()
    X_text_features = vectorizer.fit_transform(X_text).toarray()
    X_selected_text = selectkbest.fit_transform(X_text_features, y)
    duration = perf_counter() - start_time
    return X_selected_text, vectorizer, selectkbest, duration


def fit_model(sets):
    """Fit model."""
    # print("Training model with data:", sets["train"]["X"].shape)
    dtrain = xgb.DMatrix(sets["train"]["X"], label=sets["train"]["y"])
    dvalid = xgb.DMatrix(sets["valid"]["X"], label=sets["valid"]["y"])
    dtest = xgb.DMatrix(sets["test"]["X"])

    params = {"max_depth": 100, "eta": 1, "objective": "binary:logistic"}
    evallist = [(dtrain, "train"), (dvalid, "valid")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,
        evals=evallist,
        maximize=True,
        early_stopping_rounds=3,
        verbose_eval=0,
    )
    explainer = shap.TreeExplainer(model)

    shap_values = {}
    shap_values["train"] = explainer.shap_values(sets["train"]["X"])
    shap_values["valid"] = explainer.shap_values(sets["valid"]["X"])
    shap_values["test"] = explainer.shap_values(sets["test"]["X"])

    pred = model.predict(dtest)
    pred_prob = pred
    pred = [int(round(e)) for e in pred]

    return model, explainer, shap_values, pred, pred_prob


def evaluate(model1, model2, vectorizer, selectkbest, explainer1, alpha, beta, select_col, X_test, y_test):
    """Evaluate the two-step model on unseen (manually) labeled data."""
    X_text = X_test["log"]
    X_features = X_test[["n_past_reruns", "n_commit_since_brown"]]

    X_text = vectorizer.transform(X_text).toarray()
    X_text = selectkbest.transform(X_text)
    X_text = pd.DataFrame(X_text)

    pred1 = model1.predict(xgb.DMatrix(X_text))
    shap_values = explainer1.shap_values(X_text)

    X_model2 = np.concatenate((np.array(shap_values[:, select_col]), X_features.to_numpy()), axis=1)
    pred2 = model2.predict(xgb.DMatrix(X_model2))

    pred_prob_ranged = [(a * (100. - beta) + b * beta) / 100.0 for a, b in zip(pred1, pred2)]
    pred_ranged = [int(e >= float(alpha / 100.0)) for e in pred_prob_ranged]

    return evaluator.compute_metrics(y_test, pred_ranged)