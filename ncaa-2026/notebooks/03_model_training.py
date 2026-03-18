# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Model Training
# MAGIC
# MAGIC Trains three model families (XGBoost, Random Forest, Logistic Regression) across
# MAGIC a defined hyperparameter grid. Every trial is tracked in MLflow. The best model
# MAGIC by validation log-loss is saved to `automl_best_run` (best run table) for notebooks 04 and 05.
# MAGIC
# MAGIC **Train:** seasons 2003–2022
# MAGIC **Validation:** seasons 2023–2024
# MAGIC **Holdout (untouched):** 2025 tournament

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn matplotlib --quiet

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import xgboost as xgb

CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA  = "march_madness_2026"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

EXPERIMENT_NAME = (
    f"/Users/{spark.sql('SELECT current_user()').first()[0]}"
    f"/march_madness_2026_training_v2"
)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Load & Split Data
df = spark.table(f"{CATALOG}.{SCHEMA}.training_features").toPandas()

print(f"Total rows: {len(df):,}  |  Seasons: {df['Season'].min()}–{df['Season'].max()}")
print(f"Class balance (team1_won=1): {df['team1_won'].mean():.3f}")

DROP_COLS = ["Season", "TeamA", "TeamB", "team1_won"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

train_mask = df["Season"] <= 2022
val_mask   = df["Season"].isin([2023, 2024])

X_train = df.loc[train_mask, FEATURE_COLS]
y_train = df.loc[train_mask, "team1_won"].values
X_val   = df.loc[val_mask,   FEATURE_COLS]
y_val   = df.loc[val_mask,   "team1_won"].values

print(f"\nTrain: {len(X_train):,} rows (2003–2022)")
print(f"Val:   {len(X_val):,}   rows (2023–2024)")
print(f"Features: {len(FEATURE_COLS)}")
print(f"Feature list: {FEATURE_COLS}")

# COMMAND ----------

# DBTITLE 1,Probability Wrapper (pyfunc)
# Wraps any sklearn/xgb classifier so model.predict() returns P(team1 wins),
# matching the interface expected by notebooks 04 and 05.

class ProbabilityModel(mlflow.pyfunc.PythonModel):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, context, model_input):
        return self.clf.predict_proba(model_input)[:, 1]


def eval_metrics(clf, X, y):
    proba = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return {
        "log_loss":    log_loss(y, proba),
        "accuracy":    accuracy_score(y, (proba >= 0.5).astype(int)),
        "brier_score": brier_score_loss(y, proba),
    }

# COMMAND ----------

# DBTITLE 1,Hyperparameter Grids

XGB_GRID = [
    {"n_estimators": 100, "max_depth": 3,  "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1},
    {"n_estimators": 200, "max_depth": 4,  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
    {"n_estimators": 300, "max_depth": 5,  "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5},
    {"n_estimators": 400, "max_depth": 4,  "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 5},
    {"n_estimators": 200, "max_depth": 6,  "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.6, "min_child_weight": 3},
]

RF_GRID = [
    {"n_estimators": 200, "max_depth": 8,    "min_samples_leaf": 5,  "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 12,   "min_samples_leaf": 3,  "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 6,    "min_samples_leaf": 5,  "max_features": 0.5},
]

LR_GRID = [
    {"C": 0.001, "penalty": "l2", "solver": "lbfgs",     "max_iter": 2000},
    {"C": 0.01,  "penalty": "l2", "solver": "lbfgs",     "max_iter": 2000},
    {"C": 0.1,   "penalty": "l2", "solver": "lbfgs",     "max_iter": 2000},
    {"C": 1.0,   "penalty": "l2", "solver": "lbfgs",     "max_iter": 2000},
    {"C": 0.1,   "penalty": "l1", "solver": "liblinear", "max_iter": 2000},
    {"C": 1.0,   "penalty": "l1", "solver": "liblinear", "max_iter": 2000},
]

print(f"Total trials: {len(XGB_GRID)} XGB + {len(RF_GRID)} RF + {len(LR_GRID)} LR = {len(XGB_GRID)+len(RF_GRID)+len(LR_GRID)}")

# COMMAND ----------

# DBTITLE 1,Training Loop

results = []

def run_trial(model_type: str, clf, params: dict) -> dict:
    """Train clf, log everything to MLflow, return result dict."""
    clf.fit(X_train, y_train)

    train_m = eval_metrics(clf, X_train, y_train)
    val_m   = eval_metrics(clf, X_val,   y_val)

    # Infer signature from a small validation sample: DataFrame in → float array out
    sample_input = X_val.iloc[:10]
    sample_output = clf.predict_proba(sample_input)[:, 1]
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.log_params({f"param_{k}": v for k, v in params.items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_metrics({f"val_{k}":   v for k, v in val_m.items()})
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ProbabilityModel(clf),
            signature=signature,
            input_example=sample_input,
        )
        run_id = run.info.run_id

    return {
        "run_id":         run_id,
        "model_type":     model_type,
        "params":         str(params),
        "val_log_loss":   val_m["log_loss"],
        "val_accuracy":   val_m["accuracy"],
        "val_brier":      val_m["brier_score"],
        "train_log_loss": train_m["log_loss"],
    }


# --- XGBoost ---
print("Training XGBoost...")
for i, p in enumerate(XGB_GRID):
    clf = xgb.XGBClassifier(
        **p,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    r = run_trial("XGBoost", clf, p)
    results.append(r)
    print(f"  [{i+1}/{len(XGB_GRID)}] val_log_loss={r['val_log_loss']:.4f}  {p}")

# --- Random Forest ---
print("\nTraining Random Forest...")
for i, p in enumerate(RF_GRID):
    clf = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
    r = run_trial("RandomForest", clf, p)
    results.append(r)
    print(f"  [{i+1}/{len(RF_GRID)}] val_log_loss={r['val_log_loss']:.4f}  {p}")

# --- Logistic Regression ---
print("\nTraining Logistic Regression...")
for i, p in enumerate(LR_GRID):
    lr  = LogisticRegression(**p, random_state=42)
    clf = Pipeline([("scaler", StandardScaler()), ("lr", lr)])
    r = run_trial("LogisticRegression", clf, p)
    results.append(r)
    print(f"  [{i+1}/{len(LR_GRID)}] val_log_loss={r['val_log_loss']:.4f}  {p}")

print(f"\nAll {len(results)} trials complete.")

# COMMAND ----------

# DBTITLE 1,Results Summary

results_df = pd.DataFrame(results).sort_values("val_log_loss").reset_index(drop=True)

print("=" * 80)
print(f"{'Rank':<5} {'Model':<22} {'Val LogLoss':>12} {'Val Acc':>10} {'Val Brier':>10} {'Train LogLoss':>14}")
print("-" * 80)
for rank, row in results_df.iterrows():
    print(
        f"  {rank+1:<4} {row['model_type']:<22} "
        f"{row['val_log_loss']:>12.4f} {row['val_accuracy']:>10.1%} "
        f"{row['val_brier']:>10.4f} {row['train_log_loss']:>14.4f}"
    )

best = results_df.iloc[0]
print(f"\nBest: {best['model_type']}  val_log_loss={best['val_log_loss']:.4f}")
print(f"Params: {best['params']}")
print(f"Run ID: {best['run_id']}")

# COMMAND ----------

# DBTITLE 1,Comparison Chart

colors = {"XGBoost": "#e07b39", "RandomForest": "#4c9be8", "LogisticRegression": "#56b356"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax_idx, metric in enumerate(["val_log_loss", "val_accuracy", "val_brier"]):
    ax = axes[ax_idx]
    for mtype, grp in results_df.groupby("model_type"):
        ax.scatter(
            grp.index, grp[metric],
            label=mtype, color=colors.get(mtype, "gray"),
            s=80, zorder=3,
        )
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Trial (sorted by val log-loss)")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f"All {len(results_df)} Trials — NCAA 2026 Model Selection", fontsize=13)
plt.tight_layout()

with mlflow.start_run(run_name="model_comparison_summary"):
    mlflow.log_figure(fig, "model_comparison.png")

display(fig)

# COMMAND ----------

# DBTITLE 1,Feature Importance (Best Tree Model)

best_tree_row = results_df[results_df["model_type"].isin(["XGBoost", "RandomForest"])].iloc[0]
print(f"Feature importance for best tree: {best_tree_row['model_type']}")

best_params = eval(best_tree_row["params"])
if best_tree_row["model_type"] == "XGBoost":
    best_tree_clf = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
else:
    best_tree_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

best_tree_clf.fit(X_train, y_train)
fi = sorted(zip(FEATURE_COLS, best_tree_clf.feature_importances_), key=lambda x: x[1], reverse=True)

print("\nTop 20 Features:")
for name, imp in fi[:20]:
    print(f"  {name:<40} {imp:.4f}  {'█' * int(imp * 150)}")

fig2, ax = plt.subplots(figsize=(10, 8))
top_n = fi[:20]
ax.barh([n for n, _ in reversed(top_n)], [i for _, i in reversed(top_n)], color="#4c9be8")
ax.set_xlabel("Importance")
ax.set_title(f"Top 20 Feature Importances — {best_tree_row['model_type']}")
plt.tight_layout()
display(fig2)

# COMMAND ----------

# DBTITLE 1,Save Best Run for Notebooks 04 & 05

# Same schema as before — notebook 04 reads this without modification
spark.createDataFrame([{
    "run_id":            best["run_id"],
    "model_description": f"{best['model_type']} (val_log_loss={best['val_log_loss']:.4f})",
    "val_log_loss":      float(best["val_log_loss"]),
    "val_f1_score":      float(best["val_accuracy"]),
    "experiment_name":   EXPERIMENT_NAME,
}]).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.automl_best_run"
)

# Full results for the app's model metrics panel
spark.createDataFrame(results_df).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{CATALOG}.{SCHEMA}.model_trial_results")

print(f"Best run  → {CATALOG}.{SCHEMA}.automl_best_run")
print(f"All trials → {CATALOG}.{SCHEMA}.model_trial_results")
print(f"\nNext: run notebook 04 to evaluate on the 2025 holdout and register the champion.")
