# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Generate Predictions
# MAGIC
# MAGIC Builds 2026 team features from the current regular-season data (if not already
# MAGIC computed), then generates win-probability predictions for every possible 2026
# MAGIC tournament matchup. Outputs both a Kaggle submission CSV and a predictions
# MAGIC table for the Databricks App.

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn pandas numpy xgboost "typing_extensions>=4.6.0" 

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import re
from pyspark.sql import functions as F, Window
from pyspark.sql.types import IntegerType

CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA  = "march_madness_2026"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

MODEL_NAME     = f"{CATALOG}.{SCHEMA}.ncaa_bracket_model"
CURRENT_SEASON = 2026

# COMMAND ----------

# DBTITLE 1,Load Champion Model
import types as _types
import cloudpickle.cloudpickle as _cpc
import cloudpickle

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{MODEL_NAME}@champion"

# Retrieve model signature without loading the pickle
model_info = mlflow.models.get_model_info(model_uri)
sig_inputs = model_info.signature.inputs
MODEL_FEATURE_COLS = [inp.name for inp in sig_inputs]

# The model was saved with Python 3.10 but this env runs Python 3.12.
# Python 3.12 changed the code() constructor (added qualname, exceptiontable),
# which breaks cloudpickle deserialization. Patch to map the old 16-arg format
# to the new 18-arg format.
_orig_builtin_type = _cpc._builtin_type

def _compat_code(*args):
    if len(args) == 16:
        (argcount, posonlyargcount, kwonlyargcount, nlocals, stacksize, flags,
         codestring, constants, names, varnames, filename, name,
         firstlineno, lnotab, freevars, cellvars) = args
        return _types.CodeType(
            argcount, posonlyargcount, kwonlyargcount, nlocals, stacksize, flags,
            codestring, constants, names, varnames, filename, name,
            name, firstlineno, lnotab, b"", freevars, cellvars)
    return _types.CodeType(*args)

_cpc._builtin_type = lambda n: _compat_code if n == "CodeType" else _orig_builtin_type(n)

local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
with open(f"{local_path}/python_model.pkl", "rb") as f:
    _pyfunc_model = cloudpickle.load(f)

_cpc._builtin_type = _orig_builtin_type  # restore

# Wrap the underlying classifier so .predict() returns probabilities,
# matching the pyfunc interface expected by downstream cells.
class _ModelWrapper:
    def __init__(self, clf, info):
        self._clf = clf
        self.metadata = info
    def predict(self, X):
        return self._clf.predict_proba(X)[:, 1]

model = _ModelWrapper(_pyfunc_model.clf, model_info)
print(f"Loaded model: {MODEL_NAME}@champion")
print(f"Model expects {len(MODEL_FEATURE_COLS)} features: {MODEL_FEATURE_COLS[:5]} ...")

# COMMAND ----------

# DBTITLE 1,Check Whether 2026 Features Already Exist
# Load pre-built scoring features from notebook 02
scoring_features = spark.table(f"{CATALOG}.{SCHEMA}.scoring_features").toPandas()
print(f"Scoring features loaded: {scoring_features.shape[0]:,} matchups × {scoring_features.shape[1]} cols")

# COMMAND ----------

# DBTITLE 1,Load Sample Submission (Stage 2 - 2026 Matchups)
# Build submission DataFrame from scoring_features
submission_2026 = scoring_features[["Season", "TeamA", "TeamB"]].copy()
submission_2026["ID"] = submission_2026.apply(
    lambda r: f"{int(r['Season'])}_{int(r['TeamA'])}_{int(r['TeamB'])}", axis=1
)
submission_2026["Pred"] = 0.5  # placeholder

print(f"Stage 2 matchups to predict: {len(submission_2026):,}")
teams_in_bracket = set(submission_2026["TeamA"].tolist() + submission_2026["TeamB"].tolist())
print(f"Unique teams in bracket: {len(teams_in_bracket)}")

# COMMAND ----------

# DBTITLE 1,Load 2026 Team Feature Vectors
# Align scoring_features to exact model signature column order
features_df = scoring_features.reindex(columns=MODEL_FEATURE_COLS, fill_value=0.0)

# Enforce dtypes to match model signature
for inp in sig_inputs:
    col = inp.name
    if col not in features_df.columns:
        continue
    type_str = str(inp.type)
    if "long" in type_str:
        features_df[col] = features_df[col].fillna(0).round().astype("int64")
    elif "integer" in type_str:
        features_df[col] = features_df[col].fillna(0).round().astype("int32")
    else:
        features_df[col] = features_df[col].fillna(0.0).astype("float64")

print(f"Feature matrix: {features_df.shape} (rows × cols) — ready for prediction")
print(f"Sample columns: {list(features_df.columns[:6])} ...")

# COMMAND ----------

# DBTITLE 1,Generate Win Probabilities
predictions = model.predict(features_df)
predictions = np.clip(predictions, 0.001, 0.999)
submission_2026["Pred"] = predictions

print(f"Prediction statistics:")
print(f"  Mean:  {predictions.mean():.4f}")
print(f"  Std:   {predictions.std():.4f}")
print(f"  Min:   {predictions.min():.4f}")
print(f"  Max:   {predictions.max():.4f}")

# COMMAND ----------

# DBTITLE 1,Enrich with Team Names and Seeds
teams_pd = spark.table(f"{CATALOG}.{SCHEMA}.mteams").toPandas()
team_name_map = dict(zip(teams_pd["TeamID"], teams_pd["TeamName"]))

seeds_pd = spark.table(f"{CATALOG}.{SCHEMA}.mncaatourney_seeds").filter(f"Season = {CURRENT_SEASON}").toPandas()
seed_map = dict(zip(seeds_pd["TeamID"], seeds_pd["Seed"]))

submission_2026["TeamA_Name"] = submission_2026["TeamA"].map(team_name_map)
submission_2026["TeamB_Name"] = submission_2026["TeamB"].map(team_name_map)
submission_2026["TeamA_Seed"] = submission_2026["TeamA"].map(seed_map)
submission_2026["TeamB_Seed"] = submission_2026["TeamB"].map(seed_map)

# COMMAND ----------

# DBTITLE 1,Preview Predicted First-Round Results
def parse_seed_num(s):
    if s and isinstance(s, str):
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 99
    return 99

seeded = submission_2026[submission_2026["TeamA_Seed"].notna() & submission_2026["TeamB_Seed"].notna()].copy()
seeded["TeamA_Region"]  = seeded["TeamA_Seed"].apply(lambda s: s[0] if s else None)
seeded["TeamA_SeedNum"] = seeded["TeamA_Seed"].apply(parse_seed_num)
seeded["TeamB_Region"]  = seeded["TeamB_Seed"].apply(lambda s: s[0] if s else None)
seeded["TeamB_SeedNum"] = seeded["TeamB_Seed"].apply(parse_seed_num)

first_round = seeded[
    (seeded["TeamA_Region"] == seeded["TeamB_Region"]) &
    (seeded["TeamA_SeedNum"] + seeded["TeamB_SeedNum"] == 17)
].copy().sort_values(["TeamA_Region", "TeamA_SeedNum"])

regions = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
print("PREDICTED FIRST ROUND")
print("=" * 72)
for code, name in regions.items():
    games = first_round[first_round["TeamA_Region"] == code]
    if len(games):
        print(f"\n{name} Region:")
        for _, g in games.iterrows():
            winner = g["TeamA_Name"] if g["Pred"] > 0.5 else g["TeamB_Name"]
            prob   = max(g["Pred"], 1 - g["Pred"])
            print(f"  ({g['TeamA_SeedNum']:>2}) {g['TeamA_Name']:<25} vs "
                  f"({g['TeamB_SeedNum']:>2}) {g['TeamB_Name']:<25} → {winner} ({prob:.1%})")

# COMMAND ----------

# DBTITLE 1,Save Kaggle Submission CSV
volume_path   = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"
kaggle_path   = f"{volume_path}/submission_stage2_2026.csv"
submission_2026[["ID", "Pred"]].to_csv(kaggle_path, index=False)
print(f"Kaggle submission saved: {kaggle_path}  ({len(submission_2026):,} rows)")

# COMMAND ----------

# DBTITLE 1,Save Predictions Table for App
spark.createDataFrame(
    submission_2026[["ID","Season","TeamA","TeamB","TeamA_Name","TeamB_Name","TeamA_Seed","TeamB_Seed","Pred"]]
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.predictions_2026"
)
print(f"App predictions → {CATALOG}.{SCHEMA}.predictions_2026")

# COMMAND ----------

# DBTITLE 1,Save Bracket Data for App
bracket_data = first_round[[
    "TeamA","TeamB","TeamA_Name","TeamB_Name","TeamA_Seed","TeamB_Seed","Pred"
]].copy()
bracket_data["region"]  = bracket_data["TeamA_Seed"].apply(lambda s: regions.get(s[0], "Unknown") if s else "Unknown")
bracket_data["round"]   = 1
bracket_data["seed_a"]  = bracket_data["TeamA_Seed"].apply(parse_seed_num)
bracket_data["seed_b"]  = bracket_data["TeamB_Seed"].apply(parse_seed_num)

spark.createDataFrame(bracket_data).write.mode("overwrite").option("overwriteSchema","true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.bracket_first_round_2026"
)
print(f"Bracket data → {CATALOG}.{SCHEMA}.bracket_first_round_2026")
print("\nPrediction pipeline complete!")