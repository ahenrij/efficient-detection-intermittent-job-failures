# Artifact for Efficient Detection of Intermittent Job Failures Using Few-Shot Learning

Research Artifact of the paper [Efficient Detection of Intermittent Job Failures Using Few-Shot Learning](https://arxiv.org/abs/2507.04173) accepted at the 41st International Conference on Software Maintenance and Evolution ICSME 2025, Industry Track.

This artifact has been awarded the "__Open Research Object__" and "__Research Object Reviewed__" badges at ICSME 2025, Artifact Evaluation Track. It includes:

* [SLID - Source Code](src/models/) for creating and evaluating few-shot fine-tuned Small Language models for Intermittent job failures Detection.
* [Experimental Results](data/results/) including raw results from running the experiment on the Veloren project.
* [Jupyter Notebooks](notebooks/) used for conducting the study.

For the purpose of the original study, we collected CI job data from GitLab projects using the [glbuild](https://pypi.org/project/glbuild/) Python library. For confidentiality reasons, the data collected from TELUS projects are not included. However, we included the build job [dataset](data/labeled.zip) collected and manually labeled from the open-source software (OSS) project Veloren to facilitate reproducibility and reuse.

## Description of Contents

1.) `notebooks/` includes the __Jupyter Notebooks__ used to prepare data and answer our RQs. These notebooks are not exercisable, but for __read-only__ purpose.

* [Data Preparation for baseline replication](notebooks/data_preparation.ipynb)

* [RQ1. Labeling Error Rates](notebooks/RQ1_labeling_error.ipynb)
* [RQ2. FSL Performance Analysis](notebooks/RQ2_fsl_evaluation.ipynb)
* [RQ3. Cross-Predictions Analysis](notebooks/RQ3_fsl_cross_project.ipynb)

2.) `data/` includes the __datasets__ of the studied OSS project Veloren.

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

## Setup (development)

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
unzip data/prepared.zip -d .
```

Optionally, also unzip `data/sampled.zip`, `data/labeled.zip`, and `data/logs/raw.zip`

## Train and evaluate models

Here is an example of one-shot fine-tuning using the OSS project's CI job data included in this package. The `seed` arguments can be changed for another reproducible repeat.

NOTE: We recommend 16GB or more of GPU and a Linux-based operating system for fast training (~5min for one-shot).

```sh
python src/models/run.py --project veloren --shots 1 --seed 1
```

FSL results are appended to the `data/results/runs/veloren.csv` file. FSL results obtained on the Veloren project during our experiments are recorded in `data/results/runs/veloren_saved.csv`.

Expected results content is described in the following table:

|0_precision       |0_recall           |1_precision       |1_recall|1_f1_score        |random_seed       |num_shots         |training_time|
|------------------|-------------------|------------------|--------|------------------|------------------|------------------|-------------|
|0.78 |0.96 |0.91|0.57|0.70|1                 |1                 |0.41|
|0.95|0.36|0.48|0.97|0.64|4                 |1                 |0.74|
|0.75              |0.87 |0.72|0.52|0.61|2                 |1                 |0.50|
|0.79|0.98 |0.95|0.6     |0.73|3                 |1                 |0.48|
|0.80|0.95 |0.9               |0.63|0.74 |5                 |1                 |0.39|

During our experiments we used the following values for each argument:

* `project`: A, B, C, D, E, veloren
* `shots`: 1 to 15
* `seed`: 1 to 100

Run the SOTA brown job detector on the project `veloren` for comparison.

```sh
python src/models/baselines/sota_brown_detector.py --project veloren --seed 1
```

Baseline results are appended to the `data/results/baselines/veloren.csv` file. Baseline results obtained on the Veloren project during our experiments are recorded in `data/results/baselines/veloren_saved.csv`.
