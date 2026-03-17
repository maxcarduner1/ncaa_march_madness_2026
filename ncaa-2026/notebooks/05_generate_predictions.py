# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Generate Predictions
# MAGIC
# MAGIC Builds 2026 team features from the current regular-season data (if not already
# MAGIC computed), then generates win-probability predictions for every possible 2026
# MAGIC tournament matchup. Outputs both a Kaggle submission CSV and a predictions
# MAGIC table for the Databricks App.

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn pandas numpy xgboost --quiet

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
mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print(f"Loaded model: {MODEL_NAME}@champion")

# Recover the feature column names from the model signature
sig_inputs = model.metadata.signature.inputs
MODEL_FEATURE_COLS = [inp.name for inp in sig_inputs]
print(f"Model expects {len(MODEL_FEATURE_COLS)} features: {MODEL_FEATURE_COLS[:5]} ...")

# COMMAND ----------

# DBTITLE 1,Check Whether 2026 Features Already Exist
team_features_tbl = spark.table(f"{CATALOG}.{SCHEMA}.team_season_features")
latest_season = team_features_tbl.agg(F.max("Season")).first()[0]
print(f"Latest season in team_season_features: {latest_season}")

NEED_2026_FEATURES = latest_season < CURRENT_SEASON
if NEED_2026_FEATURES:
    print(f"2026 features not found — will build from 2026 regular-season data")
else:
    print(f"2026 features already present — skipping rebuild")

# COMMAND ----------

# DBTITLE 1,Build 2026 Team Features (if needed)
if NEED_2026_FEATURES:

    compact_2026  = spark.table(f"{CATALOG}.{SCHEMA}.mregularseason_compact").filter("Season = 2026")
    detailed_2026 = spark.table(f"{CATALOG}.{SCHEMA}.mregularseason_detailed").filter("Season = 2026")
    seeds_2026    = spark.table(f"{CATALOG}.{SCHEMA}.mncaatourney_seeds").filter("Season = 2026")

    n_games = compact_2026.count()
    print(f"2026 regular season games: {n_games:,}")

    # ── Win / loss records ────────────────────────────────────────────────────
    wins = (
        compact_2026
        .groupBy("Season", "WTeamID")
        .agg(
            F.count("*").alias("wins"),
            F.avg("WScore").alias("avg_pts_scored_w"),
            F.avg("LScore").alias("avg_pts_allowed_w"),
            F.avg(F.col("WScore") - F.col("LScore")).alias("avg_margin_w"),
        )
        .withColumnRenamed("WTeamID", "TeamID")
    )
    losses = (
        compact_2026
        .groupBy("Season", "LTeamID")
        .agg(
            F.count("*").alias("losses"),
            F.avg("LScore").alias("avg_pts_scored_l"),
            F.avg("WScore").alias("avg_pts_allowed_l"),
            F.avg(F.col("LScore") - F.col("WScore")).alias("avg_margin_l"),
        )
        .withColumnRenamed("LTeamID", "TeamID")
    )
    season_records = (
        wins.join(losses, ["Season", "TeamID"], "outer").fillna(0)
        .withColumn("games", F.col("wins") + F.col("losses"))
        .withColumn("win_pct", F.col("wins") / F.col("games"))
        .withColumn("avg_pts_scored",
            (F.col("avg_pts_scored_w") * F.col("wins") + F.col("avg_pts_scored_l") * F.col("losses")) / F.col("games"))
        .withColumn("avg_pts_allowed",
            (F.col("avg_pts_allowed_w") * F.col("wins") + F.col("avg_pts_allowed_l") * F.col("losses")) / F.col("games"))
        .withColumn("avg_margin",
            (F.col("avg_margin_w") * F.col("wins") + F.col("avg_margin_l") * F.col("losses")) / F.col("games"))
        .select("Season", "TeamID", "wins", "losses", "games", "win_pct",
                "avg_pts_scored", "avg_pts_allowed", "avg_margin")
    )
    print(f"Season records: {season_records.count():,} teams")

    # ── Away win % ────────────────────────────────────────────────────────────
    away_wins = (
        compact_2026.filter(F.col("WLoc") == "A")
        .groupBy("Season", "WTeamID").agg(F.count("*").alias("away_wins"))
        .withColumnRenamed("WTeamID", "TeamID")
    )
    away_losses = (
        compact_2026.filter(F.col("WLoc") == "H")
        .groupBy("Season", "LTeamID").agg(F.count("*").alias("away_losses"))
        .withColumnRenamed("LTeamID", "TeamID")
    )
    away_record = (
        away_wins.join(away_losses, ["Season", "TeamID"], "outer").fillna(0)
        .withColumn("away_games", F.col("away_wins") + F.col("away_losses"))
        .withColumn("away_win_pct",
            F.when(F.col("away_games") > 0, F.col("away_wins") / F.col("away_games")).otherwise(0.5))
        .select("Season", "TeamID", "away_win_pct")
    )

    # ── Detailed shooting / efficiency stats ──────────────────────────────────
    det_wins = (
        detailed_2026.groupBy("Season", "WTeamID")
        .agg(
            F.avg("WFGM").alias("w_fgm"), F.avg("WFGA").alias("w_fga"),
            F.avg("WFGM3").alias("w_fgm3"), F.avg("WFGA3").alias("w_fga3"),
            F.avg("WFTM").alias("w_ftm"), F.avg("WFTA").alias("w_fta"),
            F.avg("WOR").alias("w_or"), F.avg("WDR").alias("w_dr"),
            F.avg("WAst").alias("w_ast"), F.avg("WTO").alias("w_to"),
            F.avg("WStl").alias("w_stl"), F.avg("WBlk").alias("w_blk"),
            F.avg("WPF").alias("w_pf"), F.count("*").alias("w_detail_games"),
        ).withColumnRenamed("WTeamID", "TeamID")
    )
    det_losses = (
        detailed_2026.groupBy("Season", "LTeamID")
        .agg(
            F.avg("LFGM").alias("l_fgm"), F.avg("LFGA").alias("l_fga"),
            F.avg("LFGM3").alias("l_fgm3"), F.avg("LFGA3").alias("l_fga3"),
            F.avg("LFTM").alias("l_ftm"), F.avg("LFTA").alias("l_fta"),
            F.avg("LOR").alias("l_or"), F.avg("LDR").alias("l_dr"),
            F.avg("LAst").alias("l_ast"), F.avg("LTO").alias("l_to"),
            F.avg("LStl").alias("l_stl"), F.avg("LBlk").alias("l_blk"),
            F.avg("LPF").alias("l_pf"), F.count("*").alias("l_detail_games"),
        ).withColumnRenamed("LTeamID", "TeamID")
    )
    detail_stats = (
        det_wins.join(det_losses, ["Season", "TeamID"], "outer").fillna(0)
        .withColumn("detail_games", F.col("w_detail_games") + F.col("l_detail_games"))
    )
    for stat in ["fgm","fga","fgm3","fga3","ftm","fta","or","dr","ast","to","stl","blk","pf"]:
        w_col, l_col = f"w_{stat}", f"l_{stat}"
        detail_stats = detail_stats.withColumn(f"avg_{stat}",
            F.when(F.col("detail_games") > 0,
                (F.col(w_col)*F.col("w_detail_games") + F.col(l_col)*F.col("l_detail_games")) / F.col("detail_games")
            ).otherwise(0))
    detail_stats = (
        detail_stats
        .withColumn("fg_pct",       F.when(F.col("avg_fga")  > 0, F.col("avg_fgm")  / F.col("avg_fga")).otherwise(0))
        .withColumn("fg3_pct",      F.when(F.col("avg_fga3") > 0, F.col("avg_fgm3") / F.col("avg_fga3")).otherwise(0))
        .withColumn("ft_pct",       F.when(F.col("avg_fta")  > 0, F.col("avg_ftm")  / F.col("avg_fta")).otherwise(0))
        .withColumn("ast_to_ratio", F.when(F.col("avg_to")   > 0, F.col("avg_ast")  / F.col("avg_to")).otherwise(1))
        .withColumn("total_rebounds", F.col("avg_or") + F.col("avg_dr"))
        .select("Season", "TeamID", "fg_pct","fg3_pct","ft_pct",
                "avg_or","avg_dr","total_rebounds","avg_ast","avg_to",
                "ast_to_ratio","avg_stl","avg_blk","avg_pf")
    )

    # ── Elo: carry forward 2025 end-of-season ratings, update with 2026 games ─
    elo_2025 = (
        spark.table(f"{CATALOG}.{SCHEMA}.team_season_features")
        .filter("Season = 2025")
        .select("TeamID", "elo_rating")
        .toPandas()
        .set_index("TeamID")["elo_rating"]
        .to_dict()
    )
    INIT_ELO = float(np.mean(list(elo_2025.values()))) if elo_2025 else 1500.0
    print(f"Carrying forward {len(elo_2025)} Elo ratings from 2025 (mean={INIT_ELO:.0f})")

    compact_2026_pd = compact_2026.orderBy("DayNum").toPandas()
    K, HOME_ADV = 32, 100
    current_elo = {int(k): float(v) for k, v in elo_2025.items()}

    for _, row in compact_2026_pd.iterrows():
        w, l, loc = int(row["WTeamID"]), int(row["LTeamID"]), row.get("WLoc", "N")
        for t in [w, l]:
            if t not in current_elo:
                current_elo[t] = INIT_ELO
        elo_w_adj = current_elo[w] + (HOME_ADV if loc == "H" else 0)
        elo_l_adj = current_elo[l] + (HOME_ADV if loc == "A" else 0)
        exp_w = 1.0 / (1.0 + 10.0 ** ((elo_l_adj - elo_w_adj) / 400.0))
        current_elo[w] += K * (1 - exp_w)
        current_elo[l] += K * (0 - (1 - exp_w))

    elo_df_2026 = spark.createDataFrame(
        pd.DataFrame([{"Season": CURRENT_SEASON, "TeamID": t, "elo_rating": e}
                      for t, e in current_elo.items()])
    )
    print(f"2026 Elo ratings computed for {len(current_elo)} teams")

    # ── Last-10 form ──────────────────────────────────────────────────────────
    w_spec = Window.partitionBy("Season", "TeamID").orderBy("DayNum").rowsBetween(-9, 0)
    games_as_team = (
        compact_2026
        .select(F.col("Season"), F.col("DayNum"), F.col("WTeamID").alias("TeamID"), F.lit(1).alias("won"))
        .union(compact_2026.select(F.col("Season"), F.col("DayNum"), F.col("LTeamID").alias("TeamID"), F.lit(0).alias("won")))
    )
    last_10 = (
        games_as_team
        .withColumn("recent_wins",  F.sum("won").over(w_spec))
        .withColumn("recent_games", F.count("won").over(w_spec))
        .withColumn("recent_form",  F.col("recent_wins") / F.col("recent_games"))
        .groupBy("Season", "TeamID").agg(F.last("recent_form").alias("last10_win_pct"))
    )

    # ── Seeds ─────────────────────────────────────────────────────────────────
    seed_parsed = (
        seeds_2026
        .withColumn("seed_region", F.substring("Seed", 1, 1))
        .withColumn("seed_num", F.regexp_extract("Seed", r"[WXYZ](\d+)", 1).cast(IntegerType()))
        .select("Season", "TeamID", "Seed", "seed_region", "seed_num")
    )

    # ── Massey ordinals ───────────────────────────────────────────────────────
    try:
        massey_agg_2026 = (
            spark.table(f"{CATALOG}.{SCHEMA}.mmassey_ordinals")
            .filter("Season = 2026")
            .groupBy("Season", "TeamID")
            .agg(
                F.avg("OrdinalRank").alias("avg_massey_rank"),
                F.min("OrdinalRank").alias("best_massey_rank"),
                F.count(F.lit(1)).alias("massey_count"),
            )
        )
        has_massey = massey_agg_2026.count() > 0
    except Exception:
        has_massey = False
    print(f"Massey ordinals for 2026: {'available' if has_massey else 'not available'}")

    # ── Assemble 2026 team features ───────────────────────────────────────────
    team_features_2026 = (
        season_records
        .join(away_record,  ["Season", "TeamID"], "left")
        .join(detail_stats, ["Season", "TeamID"], "left")
        .join(elo_df_2026,  ["Season", "TeamID"], "left")
        .join(last_10,      ["Season", "TeamID"], "left")
        .join(seed_parsed,  ["Season", "TeamID"], "left")
    )
    if has_massey:
        team_features_2026 = team_features_2026.join(massey_agg_2026, ["Season", "TeamID"], "left")

    # Align columns to existing table schema (add any missing cols as 0)
    existing_cols = team_features_tbl.columns
    for col_name in existing_cols:
        if col_name not in team_features_2026.columns:
            team_features_2026 = team_features_2026.withColumn(col_name, F.lit(0.0))
    team_features_2026 = team_features_2026.select(existing_cols).fillna(0)

    row_count = team_features_2026.count()
    print(f"2026 team features built: {row_count} teams")

    # Append to existing table
    team_features_2026.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.team_season_features")
    print(f"Appended 2026 features → {CATALOG}.{SCHEMA}.team_season_features")

# COMMAND ----------

# DBTITLE 1,Load Sample Submission (Stage 2 - 2026 Matchups)
submission = spark.table(f"{CATALOG}.{SCHEMA}.sample_submission_stage2").toPandas()
submission["Season"] = submission["ID"].apply(lambda x: int(x.split("_")[0]))
submission["TeamA"]  = submission["ID"].apply(lambda x: int(x.split("_")[1]))
submission["TeamB"]  = submission["ID"].apply(lambda x: int(x.split("_")[2]))
submission_2026 = submission[submission["Season"] == CURRENT_SEASON].copy()
print(f"Stage 2 matchups to predict: {len(submission_2026):,}")

teams_in_bracket = set(submission_2026["TeamA"].tolist() + submission_2026["TeamB"].tolist())
print(f"Unique teams in bracket: {len(teams_in_bracket)}")

# COMMAND ----------

# DBTITLE 1,Load 2026 Team Feature Vectors
team_feat_pd = (
    spark.table(f"{CATALOG}.{SCHEMA}.team_season_features")
    .filter(f"Season = {CURRENT_SEASON}")
    .toPandas()
    .set_index("TeamID")
)
print(f"Teams with 2026 features: {len(team_feat_pd)}")

# Derive the feature columns exactly as notebook 02 does
feat_cols_from_table = [
    c for c in team_feat_pd.columns
    if c not in ("Season", "Seed", "seed_region")
]

# Cross-reference against what the model actually needs (a_*, b_*, diff_* columns)
# The model uses a_{c}, b_{c}, diff_{c} for each c in feat_cols_from_table
all_expected = [f"{p}_{c}" for c in feat_cols_from_table for p in ("a", "b", "diff")]
missing_in_model = [col for col in all_expected if col not in MODEL_FEATURE_COLS]
extra_in_model   = [col for col in MODEL_FEATURE_COLS if col not in all_expected]
if missing_in_model:
    print(f"WARNING: {len(missing_in_model)} diff cols not in model signature — will be ignored")
if extra_in_model:
    print(f"WARNING: {len(extra_in_model)} model features not in team table — will be 0.0")

# COMMAND ----------

# DBTITLE 1,Build Feature Matrix for All 2026 Matchups
feature_rows = []
missing_teams = set()

for _, row in submission_2026.iterrows():
    ta, tb = row["TeamA"], row["TeamB"]
    if ta not in team_feat_pd.index: missing_teams.add(ta)
    if tb not in team_feat_pd.index: missing_teams.add(tb)

    fa = team_feat_pd.loc[ta] if ta in team_feat_pd.index else pd.Series(0.0, index=team_feat_pd.columns)
    fb = team_feat_pd.loc[tb] if tb in team_feat_pd.index else pd.Series(0.0, index=team_feat_pd.columns)

    feat_row = {}
    for c in feat_cols_from_table:
        a_val = float(fa.get(c, 0.0))
        b_val = float(fb.get(c, 0.0))
        feat_row[f"a_{c}"] = a_val
        feat_row[f"b_{c}"] = b_val
        feat_row[f"diff_{c}"] = a_val - b_val
    feature_rows.append(feat_row)

if missing_teams:
    print(f"WARNING: {len(missing_teams)} teams missing 2026 features (zeroed out): {sorted(missing_teams)[:10]}")

# Align to exact model signature column order, filling any gaps with 0
features_df = pd.DataFrame(feature_rows).reindex(columns=MODEL_FEATURE_COLS, fill_value=0.0)
print(f"Feature matrix: {features_df.shape} — ready for prediction")

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
