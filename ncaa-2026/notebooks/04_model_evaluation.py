# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Model Evaluation
# MAGIC
# MAGIC Evaluates the best AutoML model on the 2025 tournament holdout set.
# MAGIC Calculates log_loss, accuracy, Brier score, and compares vs baselines.
# MAGIC Registers the best model in Unity Catalog Model Registry.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA = "march_madness_2026"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.ncaa_bracket_model"

# COMMAND ----------

# DBTITLE 1,Load Best Run Info and Model
best_run_info = spark.table(f"{CATALOG}.{SCHEMA}.automl_best_run").first()
best_run_id = best_run_info["run_id"]
model_description = best_run_info["model_description"]
val_log_loss = best_run_info["val_log_loss"]

print(f"Best run ID:        {best_run_id}")
print(f"Model:              {model_description}")
print(f"Validation LogLoss: {val_log_loss:.6f}")

# Load the model
model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")

# COMMAND ----------

# DBTITLE 1,Prepare 2025 Holdout Data
holdout = spark.table(f"{CATALOG}.{SCHEMA}.automl_training_data").filter("Season = 2025")

holdout_pd = holdout.toPandas()
print(f"2025 tournament games: {len(holdout_pd)}")

# Separate features and labels
drop_cols = ["Season", "TeamA", "TeamB", "team1_won"]
X_holdout = holdout_pd.drop(columns=drop_cols)
y_holdout = holdout_pd["team1_won"].values

print(f"Features: {X_holdout.shape[1]}")
print(f"Class balance: {y_holdout.mean():.2f} (fraction team1 wins)")

# COMMAND ----------

# DBTITLE 1,Generate Predictions
y_pred_proba = model.predict(X_holdout)

# Ensure probabilities are clipped to avoid log(0)
y_pred_proba = np.clip(y_pred_proba, 0.001, 0.999)
y_pred_class = (y_pred_proba >= 0.5).astype(int)

print(f"Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
print(f"Mean prediction:  {y_pred_proba.mean():.4f}")

# COMMAND ----------

# DBTITLE 1,Calculate Metrics
# Model metrics
model_logloss = log_loss(y_holdout, y_pred_proba)
model_accuracy = accuracy_score(y_holdout, y_pred_class)
model_brier = brier_score_loss(y_holdout, y_pred_proba)
model_auc = roc_auc_score(y_holdout, y_pred_proba)

# Baseline: predict 0.5 for everything
baseline_proba = np.full_like(y_pred_proba, 0.5)
baseline_logloss = log_loss(y_holdout, baseline_proba)
baseline_brier = brier_score_loss(y_holdout, baseline_proba)
baseline_accuracy = 0.5  # random guess

# Seed-based baseline: use seed difference as predictor
# Higher seed (lower number) = better team
seeds_2025 = spark.table(f"{CATALOG}.{SCHEMA}.mncaatourney_seeds").filter("Season = 2025").toPandas()
seed_map = {}
for _, row in seeds_2025.iterrows():
    import re
    seed_num = int(re.search(r"\d+", row["Seed"]).group())
    seed_map[row["TeamID"]] = seed_num

seed_preds = []
for _, row in holdout_pd.iterrows():
    seed_a = seed_map.get(row["TeamA"], 8)
    seed_b = seed_map.get(row["TeamB"], 8)
    # Lower seed number = better team = higher probability
    diff = seed_b - seed_a
    # Logistic transform of seed difference
    p = 1.0 / (1.0 + np.exp(-0.15 * diff))
    seed_preds.append(p)

seed_preds = np.array(seed_preds)
seed_logloss = log_loss(y_holdout, np.clip(seed_preds, 0.001, 0.999))
seed_accuracy = accuracy_score(y_holdout, (seed_preds >= 0.5).astype(int))
seed_brier = brier_score_loss(y_holdout, seed_preds)

# COMMAND ----------

# DBTITLE 1,Print Evaluation Results
print("=" * 60)
print("2025 Tournament Holdout Evaluation")
print("=" * 60)
print()
print(f"{'Metric':<20} {'Model':>12} {'Seed-Based':>12} {'Random':>12}")
print("-" * 56)
print(f"{'Log Loss':<20} {model_logloss:>12.4f} {seed_logloss:>12.4f} {baseline_logloss:>12.4f}")
print(f"{'Accuracy':<20} {model_accuracy:>12.1%} {seed_accuracy:>12.1%} {baseline_accuracy:>12.1%}")
print(f"{'Brier Score':<20} {model_brier:>12.4f} {seed_brier:>12.4f} {baseline_brier:>12.4f}")
print(f"{'AUC-ROC':<20} {model_auc:>12.4f} {'N/A':>12} {'0.5000':>12}")
print()

improvement_over_random = (baseline_logloss - model_logloss) / baseline_logloss * 100
improvement_over_seed = (seed_logloss - model_logloss) / seed_logloss * 100
print(f"LogLoss improvement over random:     {improvement_over_random:+.1f}%")
print(f"LogLoss improvement over seed-based: {improvement_over_seed:+.1f}%")

# COMMAND ----------

# DBTITLE 1,Visualize Predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Prediction distribution
axes[0].hist(y_pred_proba, bins=30, alpha=0.7, edgecolor="black")
axes[0].set_xlabel("Predicted P(Team A wins)")
axes[0].set_ylabel("Count")
axes[0].set_title("Prediction Distribution")
axes[0].axvline(x=0.5, color="red", linestyle="--")

# 2. Calibration plot
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_holdout, y_pred_proba, n_bins=10)
axes[1].plot(prob_pred, prob_true, marker="o", label="Model")
axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Fraction of Positives")
axes[1].set_title("Calibration Curve")
axes[1].legend()

# 3. Comparison bar chart
metrics_data = {
    "Log Loss": [model_logloss, seed_logloss, baseline_logloss],
    "Brier Score": [model_brier, seed_brier, baseline_brier],
}
x = np.arange(len(metrics_data))
width = 0.25
labels = ["ML Model", "Seed-Based", "Random"]
for i, label in enumerate(labels):
    vals = [metrics_data[m][i] for m in metrics_data]
    axes[2].bar(x + i * width, vals, width, label=label)
axes[2].set_xticks(x + width)
axes[2].set_xticklabels(metrics_data.keys())
axes[2].set_ylabel("Score (lower is better)")
axes[2].set_title("Model Comparison")
axes[2].legend()

plt.tight_layout()
display(fig)

# COMMAND ----------

# DBTITLE 1,Save Evaluation Metrics
metrics_df = spark.createDataFrame([
    {
        "metric": "log_loss",
        "model_value": float(model_logloss),
        "seed_baseline": float(seed_logloss),
        "random_baseline": float(baseline_logloss),
    },
    {
        "metric": "accuracy",
        "model_value": float(model_accuracy),
        "seed_baseline": float(seed_accuracy),
        "random_baseline": float(baseline_accuracy),
    },
    {
        "metric": "brier_score",
        "model_value": float(model_brier),
        "seed_baseline": float(seed_brier),
        "random_baseline": float(baseline_brier),
    },
    {
        "metric": "auc_roc",
        "model_value": float(model_auc),
        "seed_baseline": 0.0,
        "random_baseline": 0.5,
    },
])

metrics_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_evaluation_metrics"
)
print(f"Metrics saved to {CATALOG}.{SCHEMA}.model_evaluation_metrics")

# COMMAND ----------

# DBTITLE 1,Register Model in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Register the model
result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME,
)

print(f"Model registered: {MODEL_NAME}")
print(f"Version: {result.version}")

# Add description
client = mlflow.tracking.MlflowClient()
client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description=(
        f"NCAA March Madness bracket prediction model. "
        f"Trained on 2003-2024 tournament data. "
        f"2025 holdout log_loss={model_logloss:.4f}, accuracy={model_accuracy:.1%}. "
        f"Model type: {model_description}"
    ),
)

# Set alias for production
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=result.version,
)

print(f"Alias 'champion' set to version {result.version}")

# COMMAND ----------

# DBTITLE 1,Save Holdout Predictions for App
holdout_with_preds = holdout_pd[["Season", "TeamA", "TeamB", "team1_won"]].copy()
holdout_with_preds["predicted_prob"] = y_pred_proba

spark.createDataFrame(holdout_with_preds).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{CATALOG}.{SCHEMA}.holdout_2025_predictions")

print(f"Holdout predictions saved to {CATALOG}.{SCHEMA}.holdout_2025_predictions")
print("\nModel evaluation complete!")
