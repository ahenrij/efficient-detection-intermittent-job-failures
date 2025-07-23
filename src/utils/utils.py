"""Utilities."""

import os
import csv
import pandas as pd
from typing import Dict, List
from collections.abc import MutableMapping


def list_flaky_rerun_suites(reruns: pd.DataFrame):
    """Returns the list of rerun suites that contains at least one success and one failed jobs.

    Each result row is a rerun suite with a column id containing the ordered of rerun suite's jobs.
    """
    flaky_reruns = reruns[
        reruns["status"].map(lambda x: set(["success", "failed"]).issubset(x))
    ].reset_index(drop=True)
    return flaky_reruns


def list_rerun_suites(jobs: pd.DataFrame):
    """Each result row is a rerun suite with a column id containing the ordered of rerun suite's jobs.

    Jobs with no rerun are include in a singleton list.
    """
    reruns = (
        jobs[jobs["status"].isin(["success", "failed"])]
        .sort_values(by=["created_at"], ascending=True)
        .groupby(["project", "commit", "name"])
        .aggregate(
            {
                "id": list,
                "status": list,
                "created_at": list,
                "finished_at": list,
            }
        )
    ).reset_index()
    return reruns


def to_csv(data: List[Dict], output_file: str, mode: str = None):
    """Save list of dictionnaries to csv file.
    No effect if data is an empty array.
    """
    if len(data) == 0:
        return
    columns = data[0].keys()
    if mode is None:
        mode = "a" if os.path.isfile(output_file) else "w"

    with open(output_file, mode, newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, columns)
        if mode == "w":
            dict_writer.writeheader()
        dict_writer.writerows(data)


def flatten(dictionary, parent_key="", separator="_"):
    """Flatten a nested dictionary."""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
