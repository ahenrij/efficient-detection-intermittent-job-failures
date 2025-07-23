"""Compress data to reduce memory usage."""

import pandas as pd
from tqdm import tqdm


def reduce_memory_usage(project_name: str):
    """Convert data types and save to pickle"""

    df = pd.read_csv(f"data/{project_name}.csv")

    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns

    df["log"] = df["log"].astype("Sparse[str]")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, format="mixed")
    df["status"] = df["status"].astype("category")
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    df.to_pickle(f"data/{project_name}.pickle")

    # clear memory
    del df


if __name__ == "__main__":
    projects = [
        "veloren"
    ]
    for p in tqdm(projects):
        reduce_memory_usage(p)
