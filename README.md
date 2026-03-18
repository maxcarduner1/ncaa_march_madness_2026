# NCAA March Madness 2026 – Bracket Prediction

Machine learning pipeline to predict NCAA March Madness brackets using **Databricks AutoML**, MLflow experiment tracking, and a Databricks App to showcase predictions.

## Overview

- **Data:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) (Kaggle). Raw CSVs are stored in a Unity Catalog volume and converted to Delta tables.
- **Catalog / schema:** `serverless_stable_82l7qq_catalog`.`march_madness_2026`
- **Workspace:** Databricks sandbox (deployed via Asset Bundles).

## Repo structure

| Path | Description |
|------|-------------|
| `ncaa-2026/` | Databricks Asset Bundle (notebooks, app, jobs) |
| `ncaa-2026/notebooks/` | Pipeline: ingestion → features → AutoML → evaluation → predictions |
| `ncaa-2026/app/` | Databricks App (FastAPI + React) to view bracket and predictions |
| `ncaa-2026/resources/` | Bundle resources (jobs, app config) |

## Pipeline (notebooks)

1. **01_data_ingestion.py** – Unzip data from UC volume, create schema/volume, load CSVs into Delta tables.
2. **02_feature_engineering.py** – Build features for training and submission format.
3. **03_automl_training.py** – Train models with Databricks AutoML; track runs in MLflow.
4. **04_model_evaluation.py** – Evaluate on a holdout set and select best model.
5. **05_generate_predictions.py** – Generate Stage 1 / Stage 2 submission-ready predictions.

## Submission format

- **Stage 1:** All possible matchups from recent seasons (2022–2025) for development.
- **Stage 2:** All possible matchups for the current season; these are the predictions for the live bracket.

See `SampleSubmissionStage1.csv` and `SampleSubmissionStage2.csv` in the Kaggle data for the exact format.

## Deploy and run

From the repo root:

```bash
cd ncaa-2026
databricks bundle deploy -t dev
```

Configure the bundle target and workspace in `databricks.yml` and `resources/*.yml` as needed.

## App

The Databricks App in `ncaa-2026/app/` serves the bracket UI and model predictions. Deploy it with the bundle or via the app resource defined in `resources/app.yml`.
