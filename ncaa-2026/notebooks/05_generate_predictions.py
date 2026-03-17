# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Generate Predictions
# MAGIC
# MAGIC Loads the champion model from Unity Catalog and generates predictions
# MAGIC for all possible 2026 tournament matchups. Outputs a Kaggle submission
# MAGIC CSV and a predictions table for the Databricks App.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA = "march_madness_2026"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.ncaa_bracket_model"
CURRENT_SEASON = 2026

# COMMAND ----------

# DBTITLE 1,Load Champion Model
mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print(f"Loaded model: {MODEL_NAME}@champion")

# COMMAND ----------

# DBTITLE 1,Load Sample Submission (Stage 2 - 2026 Matchups)
submission = spark.table(f"{CATALOG}.{SCHEMA}.sample_submission_stage2").toPandas()
print(f"Stage 2 matchups: {len(submission):,}")
submission.head()

# COMMAND ----------

# DBTITLE 1,Parse Matchup IDs
# ID format: YYYY_XXXX_YYYY (Season_TeamA_TeamB where TeamA < TeamB)
submission["Season"] = submission["ID"].apply(lambda x: int(x.split("_")[0]))
submission["TeamA"] = submission["ID"].apply(lambda x: int(x.split("_")[1]))
submission["TeamB"] = submission["ID"].apply(lambda x: int(x.split("_")[2]))

# Filter to 2026 only
submission_2026 = submission[submission["Season"] == CURRENT_SEASON].copy()
print(f"2026 matchups to predict: {len(submission_2026):,}")

# Get unique teams
teams_in_bracket = set(submission_2026["TeamA"].tolist() + submission_2026["TeamB"].tolist())
print(f"Teams in bracket: {len(teams_in_bracket)}")

# COMMAND ----------

# DBTITLE 1,Load 2026 Team Features
team_features = spark.table(f"{CATALOG}.{SCHEMA}.team_season_features")

# Get feature columns (same as training)
all_cols = team_features.columns
feature_cols = [
    c for c in all_cols
    if c not in ("Season", "TeamID", "Seed", "seed_region")
]

# Load 2026 features; if 2026 not available yet, use latest season
latest_season = team_features.agg(F.max("Season")).first()[0]
use_season = CURRENT_SEASON if latest_season >= CURRENT_SEASON else latest_season
print(f"Using season {use_season} features (latest available: {latest_season})")

team_feat_pd = (
    team_features
    .filter(F.col("Season") == use_season)
    .toPandas()
    .set_index("TeamID")
)

print(f"Teams with features: {len(team_feat_pd)}")

# COMMAND ----------

# DBTITLE 1,Build Feature Vectors for All 2026 Matchups
feature_rows = []
missing_teams = set()

for _, row in submission_2026.iterrows():
    ta = row["TeamA"]
    tb = row["TeamB"]

    if ta not in team_feat_pd.index:
        missing_teams.add(ta)
    if tb not in team_feat_pd.index:
        missing_teams.add(tb)

    fa = {}
    fb = {}
    for c in feature_cols:
        fa[c] = team_feat_pd.loc[ta, c] if ta in team_feat_pd.index else 0.0
        fb[c] = team_feat_pd.loc[tb, c] if tb in team_feat_pd.index else 0.0

    diff_row = {}
    for c in feature_cols:
        diff_row[f"diff_{c}"] = float(fa[c]) - float(fb[c])

    feature_rows.append(diff_row)

if missing_teams:
    print(f"WARNING: {len(missing_teams)} teams missing features, using 0: {sorted(missing_teams)[:10]}...")

features_df = pd.DataFrame(feature_rows)
print(f"Feature matrix: {features_df.shape}")

# COMMAND ----------

# DBTITLE 1,Generate Predictions
predictions = model.predict(features_df)
predictions = np.clip(predictions, 0.001, 0.999)

submission_2026["Pred"] = predictions

print(f"Prediction statistics:")
print(f"  Mean:   {predictions.mean():.4f}")
print(f"  Std:    {predictions.std():.4f}")
print(f"  Min:    {predictions.min():.4f}")
print(f"  Max:    {predictions.max():.4f}")

# COMMAND ----------

# DBTITLE 1,Enrich Predictions with Team Names and Seeds
teams_pd = spark.table(f"{CATALOG}.{SCHEMA}.mteams").toPandas()
team_name_map = dict(zip(teams_pd["TeamID"], teams_pd["TeamName"]))

seeds_pd = (
    spark.table(f"{CATALOG}.{SCHEMA}.mncaatourney_seeds")
    .filter(f"Season = {use_season}")
    .toPandas()
)
seed_map = dict(zip(seeds_pd["TeamID"], seeds_pd["Seed"]))

submission_2026["TeamA_Name"] = submission_2026["TeamA"].map(team_name_map)
submission_2026["TeamB_Name"] = submission_2026["TeamB"].map(team_name_map)
submission_2026["TeamA_Seed"] = submission_2026["TeamA"].map(seed_map)
submission_2026["TeamB_Seed"] = submission_2026["TeamB"].map(seed_map)

# Show some interesting matchups (high probability upsets, close games)
close_games = submission_2026.nlargest(20, "Pred").sort_values("Pred", ascending=False)
print("\nHighest win probabilities for Team A:")
for _, r in close_games.head(10).iterrows():
    print(f"  {r['TeamA_Name']} ({r['TeamA_Seed']}) vs {r['TeamB_Name']} ({r['TeamB_Seed']}): {r['Pred']:.4f}")

# COMMAND ----------

# DBTITLE 1,Save Kaggle Submission CSV
volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"

# Format for Kaggle: ID, Pred
kaggle_submission = submission_2026[["ID", "Pred"]].copy()
kaggle_path = f"{volume_path}/submission_stage2_2026.csv"
kaggle_submission.to_csv(kaggle_path, index=False)

print(f"Kaggle submission saved: {kaggle_path}")
print(f"  Rows: {len(kaggle_submission):,}")

# COMMAND ----------

# DBTITLE 1,Save Full Predictions Table for App
predictions_df = spark.createDataFrame(
    submission_2026[
        [
            "ID", "Season", "TeamA", "TeamB",
            "TeamA_Name", "TeamB_Name",
            "TeamA_Seed", "TeamB_Seed",
            "Pred",
        ]
    ]
)

predictions_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.predictions_2026"
)
print(f"App predictions saved to {CATALOG}.{SCHEMA}.predictions_2026")

# COMMAND ----------

# DBTITLE 1,Generate Bracket (First Round Matchups by Region)
import re

def parse_seed(seed_str):
    """Parse seed string like 'W01' -> ('W', 1)"""
    if seed_str and isinstance(seed_str, str):
        region = seed_str[0]
        num = int(re.search(r"\d+", seed_str).group())
        return region, num
    return None, None

# Build first round matchups: 1v16, 2v15, 3v14, ... 8v9
bracket_matchups = []
regions = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}

# Get seeded teams
seeded = submission_2026[submission_2026["TeamA_Seed"].notna() & submission_2026["TeamB_Seed"].notna()].copy()
seeded["TeamA_Region"] = seeded["TeamA_Seed"].apply(lambda s: s[0] if s else None)
seeded["TeamA_SeedNum"] = seeded["TeamA_Seed"].apply(lambda s: int(re.search(r"\d+", s).group()) if s else 99)
seeded["TeamB_Region"] = seeded["TeamB_Seed"].apply(lambda s: s[0] if s else None)
seeded["TeamB_SeedNum"] = seeded["TeamB_Seed"].apply(lambda s: int(re.search(r"\d+", s).group()) if s else 99)

# Find first-round matchups (seeds sum to 17)
first_round = seeded[
    (seeded["TeamA_Region"] == seeded["TeamB_Region"])
    & (seeded["TeamA_SeedNum"] + seeded["TeamB_SeedNum"] == 17)
].copy()

first_round = first_round.sort_values(["TeamA_Region", "TeamA_SeedNum"])

print("PREDICTED FIRST ROUND:")
print("=" * 70)
for region_code, region_name in regions.items():
    region_games = first_round[first_round["TeamA_Region"] == region_code]
    if len(region_games) > 0:
        print(f"\n{region_name} Region:")
        for _, g in region_games.iterrows():
            winner = g["TeamA_Name"] if g["Pred"] > 0.5 else g["TeamB_Name"]
            prob = g["Pred"] if g["Pred"] > 0.5 else 1 - g["Pred"]
            print(
                f"  ({g['TeamA_SeedNum']:>2}) {g['TeamA_Name']:<25} vs "
                f"({g['TeamB_SeedNum']:>2}) {g['TeamB_Name']:<25} "
                f"-> {winner} ({prob:.1%})"
            )

# COMMAND ----------

# DBTITLE 1,Save Bracket Data for App
bracket_data = first_round[
    [
        "TeamA", "TeamB", "TeamA_Name", "TeamB_Name",
        "TeamA_Seed", "TeamB_Seed", "Pred",
    ]
].copy()
bracket_data["region"] = bracket_data["TeamA_Seed"].apply(lambda s: regions.get(s[0], "Unknown") if s else "Unknown")
bracket_data["round"] = 1
bracket_data["seed_a"] = bracket_data["TeamA_Seed"].apply(lambda s: int(re.search(r"\d+", s).group()) if s else 99)
bracket_data["seed_b"] = bracket_data["TeamB_Seed"].apply(lambda s: int(re.search(r"\d+", s).group()) if s else 99)

spark.createDataFrame(bracket_data).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{CATALOG}.{SCHEMA}.bracket_first_round_2026")

print(f"\nBracket data saved to {CATALOG}.{SCHEMA}.bracket_first_round_2026")
print("\nPrediction pipeline complete!")
