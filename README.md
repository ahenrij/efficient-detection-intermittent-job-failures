# Artifact for Efficient Detection of Intermittent Job Failures Using Few-Shot Learning

Replication package of the paper [Efficient Detection of Intermittent Job Failures Using Few-Shot Learning](https://arxiv.org/abs/2507.04173) accepted at the 41st International Conference on Software Maintenance and Evolution ICSME 2025, Industry Track.

This replication package includes:

* [Source Code](src/models/) for creating FSL models for detecting intermittent job failures and running the experiments.
* [Experimental Results](data/results/) including raw results from running the experiment on the Veloren project.
* [Jupyter Notebooks](notebooks/) used for conducting the study.

To conduct the study, we collected build job data from GitLab projects using the python-gitlab library. For confidentiality reasons, the data collected from TELUS projects are not included. However, we included the build job [dataset](data/labeled.zip) collected and manually labeled from the open-source (OS) project Veloren to facilitate reproducibility and reuse.

## Content of the Replication Package

1.) `notebooks/` includes the **Jupyter Notebooks** used to prepare data and answer our RQs. These notebooks are not exercisable, but for read-only purpose.

* [Data Preparation for baseline replication](notebooks/data_preparation.ipynb)

* [RQ1. Labeling Error Rates](notebooks/RQ1_labeling_error.ipynb)
* [RQ2. FSL Performance Analysis](notebooks/RQ2_fsl_evaluation.ipynb)
* [RQ3. Cross-Predictions Analysis](notebooks/RQ3_fsl_cross_project.ipynb)

2.) `data/` includes the **datasets** of the studied open-source project Veloren.

* [Prepared Dataset](data/prepared.zip) `prepared.zip` with automated labels and features for baseline replication
* [Sample Dataset](data/sampled.zip) `sampled.zip` for performing manual labeling
* [Labeled Sample Dataset](data/labeled.zip) `labeled.zip` including the manual and automated labels. This dataset is the input of the FSL model for the OS project.
* [Raw Sampled Logs](data/logs/raw.zip) `logs/raw.zip` of each job in the sampled dataset. Each log file in the directory is named as follows:

`{projectId}_{jobId}_{automatedLabel}_{manualLabel}_{failureCategoryId}.log`

where the `failureCategoryId` maps on the categories in the [failure_reasons.csv](data/results/failure_reasons.csv) file.

2.) `src/` contains the source code for:

* [Creating and evaluating an FSL model](src/models/run.py) `models/run.py`
* [Creating and evaluation a baseline model](src/models/baselines/sota_brown_detector.py) `models/baselines/sota_brown_detector.py`
* [FSL hyperparameter search module](src/models/hp_search.py) `models/hp_search.py`
* [FSL model evaluator module](src/models/evaluator.py) `models/evaluator.py`
* [Log pre-processing utilities](src/preprocessing/) `preprocessing/log.py`

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

### Unzip all datasets

```sh
unzip data/prepared.zip -d .
```

```sh
unzip data/sampled.zip -d .
```

```sh
unzip data/logs/raw.zip -d .
```

```sh
unzip data/labeled.zip -d .
```

## Running experiments

Example of one-shot on the OS project (Veloren). The `seed` arguments can be changed for another reproducible repeat.

```sh
python src/models/run.py --project veloren --shots 1 --seed 1
```

Results are appended to the `data/results/veloren.csv` file. Actual results obtained for the Veloren project during our experiments are recorded in `data/results/veloren_saved.csv`.

During our experiments we used the following values for each argument:

* `project`: A, B, C, D, E, veloren
* `shots`: 1 to 15
* `seed`: 1 to 100

Run the SOTA brown job detector on the project `veloren` for comparison.

```sh
python src/models/baselines/sota_brown_detector.py --project veloren --seed 1
```
