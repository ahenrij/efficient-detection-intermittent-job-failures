"""Utility to quickly search similar brown jobs based on log regex."""

import pandas as pd


if __name__ == "__main__":
    project = "A"
    search_regex = r"error during connect.*no such host"

    df: pd.DataFrame = pd.read_pickle(f"data/prepared/{project}.pickle")
    df = df[~df["log"].isna()]
    df["log"] = df["log"].astype(str)
    df = df[df["log"].str.len() > 0]

    df = df[df["brown"] == 1]

    similar_flaky_job_ids = df[
        df["log"].str.lower().str.contains(search_regex, regex=True)
    ]["id"].tolist()[:5]

    project_id = df.iloc[0]["project"]

    print([f"{project_id}_{job_id}" for job_id in similar_flaky_job_ids])
