#!/usr/bin/env pypy
"""Label jobs with flakiness and keep only failures."""

import os
import logging
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from src.utils import utils, constants


tqdm.pandas()
logging.basicConfig(level=logging.INFO)


statistics = []
logger = logging.getLogger("bbdllm")


def prepare_data(project_name: str):
    """Label dataset, filter out irrelevant jobs, and perform feature engineering."""
    logger.info("Preparing jobs for project %s...", project_name)

    df: pd.DataFrame = pd.read_pickle(f"data/{project_name}.pickle")
    df = df[~df["commit"].isna()]
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")

    # label brown jobs
    logger.info("Labeling brown jobs...")
    reruns = utils.list_rerun_suites(df)
    flaky_reruns = utils.list_flaky_rerun_suites(reruns)
    flaky_job_ids = list(itertools.chain(*flaky_reruns["id"].to_list()))
    df["brown"] = df.progress_apply(
        lambda job: (
            1 if ((job["id"] in flaky_job_ids) and (job["status"] == "failed")) else 0
        ),
        axis=1,
    ).astype("category")
    df.sort_values(by="created_at", ascending=True, inplace=True)

    # keep only failed jobs and sort by creation datetimes
    logger.info("Creating job failures dataset...")
    df = df[df["status"] == "failed"]
    df.sort_values(by="created_at", ascending=True, inplace=True)

    # compute additional features
    logger.info("Computing additional features...")

    def rerun_counts(job_id: int):
        """Returns the total, success, and failed number of job reruns for a job."""
        # print("job", job_id)
        rerun_sequence = (
            reruns[reruns["id"].apply(lambda l: job_id in l)].iloc[0].to_dict()
        )
        idx = rerun_sequence["id"].index(job_id)
        statuses = rerun_sequence["status"][:idx]
        total_reruns = len(statuses)
        success_reruns = len(list(filter(lambda x: x == "success", statuses)))
        failed_reruns = len(list(filter(lambda x: x == "failed", statuses)))
        return total_reruns, success_reruns, failed_reruns

    def recent_brownness_ratio(job_creation_dt):
        """Compute brown failure ratio in the last 5 job failures."""

        last_jobs = df[df["created_at"] < job_creation_dt].tail(5)
        return last_jobs["brown"].astype(int).mean()

    def features_since_brown(job_creation_dt):
        """Compute temporal features of a job since last flaky."""
        brown_event_dates = pd.Series(df[df["brown"] == 1]["created_at"].to_list())
        last_brown_dt = brown_event_dates[brown_event_dates < job_creation_dt].max()

        mask = (df["created_at"] > last_brown_dt) & (
            df["created_at"] <= job_creation_dt
        )
        jobs_since_brown = df[mask]
        n_commit_since_brown = jobs_since_brown["commit"].nunique()
        time_since_brown = (job_creation_dt - last_brown_dt).total_seconds()
        return (
            n_commit_since_brown,
            time_since_brown,
            recent_brownness_ratio(job_creation_dt),
        )

    def compute_features(job):
        id = job["id"]
        creation_date = job["created_at"]

        a, b, c = rerun_counts(id)
        d, e, f = features_since_brown(creation_date)
        return a, b, c, d, e, f

    (
        df["n_past_reruns"],
        df["n_past_successes"],
        df["n_past_fails"],
        df["n_commit_since_brown"],
        df["time_since_brown"],
        df["recent_brownness_ratio"],
    ) = zip(*df.progress_apply(compute_features, axis=1))

    # keep only useful columns
    df = df[
        [
            "id",
            "name",
            "commit",
            "project",
            "created_at",
            "n_past_reruns",
            "n_past_successes",
            "n_past_fails",
            "n_commit_since_brown",
            "time_since_brown",
            "recent_brownness_ratio",
            "log",
            "brown",
        ]
    ]

    # save prepared dataset
    logger.info("Saving dataset...")
    df.to_pickle(f"data/prepared/{project_name}.pickle")

    # free memory
    del df, reruns, flaky_reruns, flaky_job_ids


def compute_statistics(project_name: str):
    """Compute project statistics"""
    df: pd.DataFrame = pd.read_pickle(f"data/{project_name}.pickle")
    df = df[~df["commit"].isna()]
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="mixed")

    project_info = {}
    project_info["name"] = project_name
    project_info["months"] = int(
        (df["created_at"].max() - df["created_at"].min())
        / (np.timedelta64(1, "D") * constants.AVG_N_DAYS_PER_MONTH)
    )
    project_info["commits"] = df["commit"].nunique()
    project_info["jobs"] = df.shape[0]
    project_info["success"] = df[df["status"] == "success"].shape[0]
    project_info["failed"] = df[df["status"] == "failed"].shape[0]
    # project_info["logs"] = df[(df["status"] == "failed") & ~df["log"].isna()]
    del df

    df: pd.DataFrame = pd.read_pickle(f"data/prepared/{project_name}.pickle")
    n_brown = df[df["brown"] == 1].shape[0]
    project_info["BFR"] = round(n_brown * 100 / df.shape[0], 2)

    statistics.append(project_info)
    del df


if __name__ == "__main__":
    for p in tqdm(constants.PROJECTS[:5]):
        if not os.path.isfile(f"data/prepared/{p}.pickle"):
            prepare_data(p)
        compute_statistics(p)

    stats = pd.DataFrame.from_records(data=statistics)
    stats.to_csv("data/results/statistics.csv", index=False)
