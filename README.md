# Artifact for Efficient Detection of Intermittent Job Failures Using Few-Shot Learning

This replication package includes source code, experimental results and notebooks used for the paper.

## Requirements

* [Poetry](https://python-poetry.org/docs/)
* [Python >= 3.10](https://www.python.org/downloads/)
* [Poetry Shell plugin](https://github.com/python-poetry/poetry-plugin-shell)

```sh
poetry self add poetry-plugin-shell
```

## Setup

### Install dependencies

```sh
poetry install
```

### Activate virtual environment

```sh
poetry shell
```

### Unzip datasets

```sh
unzip data/prepared.zip -d data/
```

```sh
unzip data/sampled.zip -d data/
```

```sh
unzip data/logs/raw.zip -d data/logs/
```

```sh
unzip data/labeled.zip -d data/
```

## Running experiments

Example of 12-shot on the project `veloren`. The `seed` arguments should be changed for another reproducible repeat.

```sh
python src/models/run.py --project veloren --shots 12 --seed 1
```

During our experiments we use the following values for each argument:

* `project`: A, B, C, D, E, veloren
* `shots`: 1 to 15
* `seed`: 1 to 100

Run the SOTA brown job detector on the project `veloren` for comparison.

```sh
python src/models/baselines/sota_brown_detector.py --project veloren --seed 1
```
