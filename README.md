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

Example of one-shot experiment on the OS project (Veloren). The `seed` arguments can be changed for another reproducible repeat.

IMPORTANT: We ran our experiments on a 16GB GPU (VRAM) Linux-based OS.

```sh
python src/models/run.py --project veloren --shots 1 --seed 1
```

FSL Results are appended to the `data/results/runs/veloren.csv` file. FSL results obtained on the Veloren project during our experiments are recorded in `data/results/runs/veloren_saved.csv`.

Expected results content is described in the following table:

|0_precision       |0_recall           |1_precision       |1_recall|1_f1_score        |random_seed       |num_shots         |training_time|
|------------------|-------------------|------------------|--------|------------------|------------------|------------------|-------------|
|0.782608695652174 |0.9642857142857143 |0.9111111111111111|0.5774647887323944|0.7068965517241379|1                 |1                 |0.4177382302004844|
|0.9534883720930233|0.36283185840707965|0.4857142857142857|0.9714285714285714|0.6476190476190476|4                 |1                 |0.7450696988962591|
|0.75              |0.8761061946902655 |0.7254901960784313|0.5285714285714286|0.6115702479338843|2                 |1                 |0.5001221669372171|
|0.7985611510791367|0.9823008849557522 |0.9545454545454546|0.6     |0.7368421052631579|3                 |1                 |0.4844527270179242|
|0.8045112781954887|0.9553571428571429 |0.9               |0.6338028169014085|0.743801652892562 |5                 |1                 |0.39875594596378505|

During our experiments we used the following values for each argument:

* `project`: A, B, C, D, E, veloren
* `shots`: 1 to 15
* `seed`: 1 to 100

Run the SOTA brown job detector on the project `veloren` for comparison.

```sh
python src/models/baselines/sota_brown_detector.py --project veloren --seed 1
```

Baseline Results are appended to the `data/results/baselines/veloren.csv` file. Baseline results obtained on the Veloren project during our experiments are recorded in `data/results/baselines/veloren_saved.csv`.
