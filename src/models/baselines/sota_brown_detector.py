"""Replicating SOTA (Olewicki et al.'s) brown job detection approach."""

import numpy as np
import argparse
from time import perf_counter
from typing import Tuple
import pandas as pd


from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from src.models import evaluator
from src.models.baselines.utils import evaluate, fit_model, vectorize
from src.preprocessing import log
from src.utils import utils


# constants
label_column = "brown"
text_column = "log"
feature_columns = [
    "n_past_reruns",
    "n_past_successes",
    "n_past_fails",
    "n_commit_since_brown",
    "time_since_brown",
    "recent_brownness_ratio",
]


def load_dataframe(input_file: str) -> pd.DataFrame:
    """Read input file and apply log processing."""
    df = pd.read_csv(input_file)
    df["log"] = df["log"].astype(str)
    df["log"] = df["log"].apply(log.clean)
    df.sort_values(by="created_at", ascending=True, inplace=True)
    return df


def split(
    df: pd.DataFrame, random_seed: int = 42, test_size: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into (train, test). Defaults to split into halves."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=df["flaky"],
    )


def create_sets(features, vectorized_text, y, train_index, test_index, seed):
    sets = {}
    sets["train"] = {}
    sets["train"]["X"] = vectorized_text.iloc[train_index, :]
    sets["train"]["features"] = features.iloc[train_index, :]
    sets["train"]["y"] = y.iloc[train_index]

    sets["valid"], sets["test"] = {}, {}
    sets["valid"]["X"], sets["test"]["X"] = train_test_split(
        vectorized_text.iloc[test_index, :], random_state=seed, test_size=0.5
    )
    sets["valid"]["features"], sets["test"]["features"] = train_test_split(
        features.iloc[test_index, :], random_state=seed, test_size=0.5
    )
    sets["valid"]["y"], sets["test"]["y"] = train_test_split(
        y.iloc[test_index], random_state=seed, test_size=0.5
    )
    return sets


def create_second_model_sets(shap_values, sets):
    """Create sets of shap values and metric features."""
    list_add = ["n_past_reruns", "n_commit_since_brown"]

    select_col = np.std(shap_values["train"], axis=0) != 0
    select_col = [i for i, e in enumerate(select_col) if e]
    mat2 = np.concatenate((np.array(shap_values["train"][:, select_col]), sets["train"]["features"][list_add].to_numpy()), axis=1)
    valid_X2 = np.concatenate((np.array(shap_values["valid"][:, select_col]), sets["valid"]["features"][list_add].to_numpy()), axis=1)
    test_X2 = np.concatenate((np.array(shap_values["test"][:, select_col]), sets["test"]["features"][list_add].to_numpy()), axis=1)

    second_sets = {}
    second_sets["train"] = {"X": mat2, "y": sets["train"]["y"]}
    second_sets["valid"] = {"X": valid_X2, "y": sets["valid"]["y"]}
    second_sets["test"] = {"X": test_X2, "y": sets["test"]["y"]}
    return select_col, second_sets


def run(input_file: str, sample_file: str, output_file: str, seed: int = 42):
    """Run a training and evaluation loop."""
    # Read and process data
    df = pd.read_pickle(input_file)
    df[text_column] = df[text_column].astype(str)

    sample = pd.read_csv(sample_file)

    df = df[~df["id"].isin(sample["id"].tolist())]

    X = df[text_column]
    y = df[label_column]

    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    # Create TF-IDF features
    X_text, vectorizer, selectkbest, vectorization_time = vectorize(X, y)
    X_text = pd.DataFrame(X_text)
    print("vectorized:", X_text.shape)

    results = []

    # train / valid / test split
    for _, (train_index, test_index) in tqdm(enumerate(skf.split(X_text, y))):
        sets = create_sets(
            features=df[feature_columns],
            vectorized_text=X_text,
            y=y,
            train_index=train_index,
            test_index=test_index,
            seed=seed,
        )
        ############################################
        #   first vocabulary-based model training
        ############################################
        start_time = perf_counter()
        model1, explainer1, shap_values, pred, pred_prob = fit_model(sets)
        training_time = vectorization_time + perf_counter() - start_time

        ############################################
        #   second features-based model training
        ############################################
        select_col, second_sets = create_second_model_sets(shap_values=shap_values, sets=sets)

        start_time = perf_counter()
        model2, explainer2, shap_val2, pred_2, pred_prob_2 = fit_model(second_sets)
        training_time += perf_counter() - start_time

        split_results = []
        for alpha in range(0, 110, 10):
            for beta in range(10, 100, 10):
                pred_prob_ranged = [(a * (100. - beta) + b * beta) /
                                    100.0 for a, b in zip(pred_prob, pred_prob_2)]
                pred_ranged = [int(e >= float(alpha / 100.0))
                            for e in pred_prob_ranged]

                # id = '%.1fvar_%dtresh' % (float(beta), alpha)
                result_ranged = evaluator.compute_metrics(sets["test"]["y"], pred_ranged)

                split_results.append({
                    **result_ranged,
                    "training_time": training_time,
                    "alpha": alpha,
                    "beta": beta
                })

        best_split_result = max(split_results, key=lambda x: x["1_f1_score"])
        results.append(best_split_result)

    best_result = max(results, key=lambda x: x["1_f1_score"])
    # print(best_result)

    _, test_set = split(sample, seed, test_size=0.5)
    generalization_result = evaluate(model1=model1, 
                                     model2=model2, 
                                     vectorizer=vectorizer, 
                                     selectkbest=selectkbest,
                                     explainer1=explainer1,
                                     alpha=best_result["alpha"],
                                     beta=best_result["beta"],
                                     select_col=select_col,
                                     X_test=test_set[["log", "n_past_reruns", "n_commit_since_brown"]],
                                     y_test=test_set["flaky"])
    
    utils.to_csv([generalization_result], output_file)
    print("F1-Score: ", generalization_result["1_f1_score"])


def main(project: str, random_seed: int = 0):
    print(f"Running for random_seed: {random_seed}")
    run(
        input_file=f"data/prepared/{project}.pickle",
        sample_file=f"data/labeled/{project}.csv",
        output_file=f"data/results/baselines/{project}.csv",
        seed=random_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project")
    parser.add_argument("-s", "--seed")
    args = parser.parse_args()
    main(args.project, int(args.seed))
