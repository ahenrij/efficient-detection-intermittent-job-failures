import gc
import torch
import argparse
import pandas as pd
from functools import partial
from time import perf_counter
from typing import Any, Dict, Tuple
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, Trainer, sample_dataset

from src.preprocessing import log
from src.utils import utils, constants
from src.models import hp_search, evaluator


# constants
label_column = "flaky"
text_column = "log"


def load_dataframe(input_file: str) -> pd.DataFrame:
    """Read input file and apply log processing."""
    df = pd.read_csv(input_file)
    df["log"] = df["log"].astype(str)
    df["log"] = df["log"].apply(log.clean)
    df.sort_values(by="created_at", ascending=True, inplace=True)
    return df


def model_init(params: Dict[str, Any]) -> SetFitModel:
    # PRETRAINED_ST_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
    PRETRAINED_ST_MODEL = "BAAI/bge-small-en-v1.5"
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "trust_remote_code": True,
        # "local_files_only": True,
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        },
    }
    # memory management
    gc.collect()
    torch.cuda.empty_cache()
    return SetFitModel.from_pretrained(PRETRAINED_ST_MODEL, **params)


def split(
    df: pd.DataFrame, random_seed: int = 42, test_size: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into (train, test). Defaults to split into halves."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=df[label_column],
    )


def cross_project_eval(model, random_seed: int = 42) -> Dict[str, float]:
    """Cross-project evaluation of the trained model."""
    results = {}

    for project in constants.PROJECTS:
        # select test dataset
        df = load_dataframe(f"data/labeled/{project}.csv")
        _, test = split(df, random_seed, test_size=0.5)
        test_dataset = Dataset.from_pandas(test)
        X_test, y_test = test_dataset[text_column], test_dataset[label_column]

        # predict and evaluate
        y_pred = model.predict(X_test)
        results[project] = evaluator.f1_score(y_pred, y_test)

        # free memory
        del df, test, test_dataset, X_test, y_test

    return results


def run(input_file: str, output_file: str, seed: int = 42, num_shots: int = 20):
    """Run a training and evaluation loop."""
    # read and process data
    df = load_dataframe(input_file)

    # split data into 25% train, 25% valid, 50% test
    train, test = split(df, seed, test_size=0.5)
    train, valid = split(train, seed, test_size=0.5)

    # create dataset dict
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(train)
    dataset["valid"] = Dataset.from_pandas(valid)
    dataset["test"] = Dataset.from_pandas(test)

    # free memory
    del df, train, valid, test

    # sample n_shots from training dataset
    train_dataset = sample_dataset(
        dataset=dataset["train"],
        label_column=label_column,
        num_samples=num_shots,
        seed=seed,
    )

    # create trainer
    trainer = Trainer(
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=dataset["valid"],
        metric="f1",
        metric_kwargs={"average": "binary"},
        column_mapping={
            text_column: "text",
            label_column: "label",
        },
    )

    # hyperparameters tuning
    trainer.run_hp_search_optuna = hp_search.run_hp_search_optuna_updated
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=partial(hp_search.hp_space, seed=seed),
        n_trials=5,
    )

    # training with best hyperparameters
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    
    start_time = perf_counter()
    trainer.train()
    training_time = perf_counter() - start_time

    X_test, y_test = dataset["test"][text_column], dataset["test"][label_column]

    # evaluation on in-project test data
    y_pred = trainer.model.predict(X_test)

    result = evaluator.compute_metrics(y_pred, y_test)
    result["random_seed"] = seed
    result["num_shots"] = num_shots
    result["training_time"] = training_time

    # cross-project evaluation
    cross_result = cross_project_eval(model=trainer.model, random_seed=seed)

    # save result to csv
    merged_result = result | cross_result
    utils.to_csv([merged_result], output_file)

    # memory management
    del dataset, train_dataset
    del trainer, best_run, X_test, y_test, y_pred, result
    gc.collect()
    torch.cuda.empty_cache()


def main(project: str, num_shots: int = 1, random_seed: int = 42):
    print(f"Running for num_shots: {num_shots} and random_seed: {random_seed}")
    run(
        input_file=f"data/labeled/{project}.csv",
        output_file=f"data/results/runs/{project}.csv",
        seed=random_seed,
        num_shots=num_shots,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project")
    parser.add_argument("-n", "--shots")
    parser.add_argument("-s", "--seed")
    args = parser.parse_args()
    main(args.project, int(args.shots), int(args.seed))
