# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - AutoML Training
# MAGIC
# MAGIC Runs Databricks AutoML on the matchup features to find the best model
# MAGIC for predicting tournament game outcomes. Tracks everything with MLflow.

# COMMAND ----------

import mlflow
from databricks import automl

CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA = "march_madness_2026"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/march_madness_2026_automl"

# COMMAND ----------

# DBTITLE 1,Load Training Data (2003-2024 for Train/Val, 2025 Held Out)
automl_data = spark.table(f"{CATALOG}.{SCHEMA}.automl_training_data")

# Use seasons 2003-2024 for training; hold out 2025 for evaluation
train_val_data = automl_data.filter("Season >= 2003 AND Season <= 2024")
holdout_2025 = automl_data.filter("Season = 2025")

print(f"Train/Val data: {train_val_data.count():,} rows (seasons 2003-2024)")
print(f"Holdout 2025:   {holdout_2025.count():,} rows")

# Drop Season, TeamA, TeamB from training -- model should only see features
drop_cols = ["Season", "TeamA", "TeamB"]
train_val_clean = train_val_data.drop(*drop_cols)

print(f"Feature columns: {len(train_val_clean.columns) - 1}")  # minus label
print(f"Label: team1_won")

# COMMAND ----------

# DBTITLE 1,Run AutoML Classification
mlflow.set_experiment(EXPERIMENT_NAME)

summary = automl.classify(
    dataset=train_val_clean,
    target_col="team1_won",
    primary_metric="log_loss",
    timeout_minutes=30,
    experiment_name=EXPERIMENT_NAME,
)

# COMMAND ----------

# DBTITLE 1,AutoML Results Summary
print("=" * 60)
print("AutoML Training Complete")
print("=" * 60)
print(f"Best trial notebook: {summary.best_trial.notebook_path}")
print(f"Best model:          {summary.best_trial.model_description}")
print()

# Print all metrics
for metric, value in sorted(summary.best_trial.metrics.items()):
    print(f"  {metric}: {value:.6f}")

print()
print(f"MLflow run ID: {summary.best_trial.mlflow_run_id}")
print(f"Experiment:    {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Log Best Model Artifact Info
best_run_id = summary.best_trial.mlflow_run_id

# Load the best model to verify it works
best_model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")

# Quick sanity check: predict on a small sample
sample = train_val_clean.limit(5).toPandas()
features = sample.drop(columns=["team1_won"])
preds = best_model.predict(features)
print("Sample predictions (sanity check):")
for i, (pred, actual) in enumerate(zip(preds, sample["team1_won"])):
    print(f"  Row {i}: predicted={pred:.4f}, actual={actual}")

# COMMAND ----------

# DBTITLE 1,Save AutoML Summary for Downstream Notebooks
# Store the run ID so notebooks 04 and 05 can pick it up
spark.createDataFrame(
    [
        {
            "run_id": best_run_id,
            "model_description": summary.best_trial.model_description,
            "val_log_loss": float(summary.best_trial.metrics.get("val_log_loss", 0)),
            "val_f1_score": float(summary.best_trial.metrics.get("val_f1_score", 0)),
            "experiment_name": EXPERIMENT_NAME,
        }
    ]
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.automl_best_run"
)

print(f"\nBest run info saved to {CATALOG}.{SCHEMA}.automl_best_run")

# COMMAND ----------

# DBTITLE 1,Explore Feature Importance
import matplotlib.pyplot as plt

try:
    # Try to get feature importance from the underlying sklearn model
    inner_model = best_model._model_impl.python_model
    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "model") and hasattr(inner_model.model, "feature_importances_"):
        importances = inner_model.model.feature_importances_
    else:
        # Try to access via the sklearn pipeline
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(best_run_id)
        importances = None

    if importances is not None:
        feature_names = features.columns.tolist()
        fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        print("Top 20 Features:")
        for name, imp in fi[:20]:
            print(f"  {name}: {imp:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = fi[:20]
        ax.barh([n for n, _ in reversed(top_n)], [i for _, i in reversed(top_n)])
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Feature Importances")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        display(fig)

except Exception as e:
    print(f"Could not extract feature importance: {e}")
    print("This is normal for some model types (e.g., stacked ensembles)")
