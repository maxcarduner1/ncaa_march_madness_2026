# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Feature Engineering
# MAGIC
# MAGIC Builds per-season team-level features and pairwise matchup features for
# MAGIC every tournament game in history (2003-2025). Output is a training table
# MAGIC with one row per ordered (TeamA < TeamB) matchup plus a binary label.
# MAGIC
# MAGIC Skips all computation if `training_features` already exists.
# MAGIC Set `FORCE_RERUN = True` to rebuild from scratch.

# COMMAND ----------

# MAGIC %pip install pandas numpy --quiet

# COMMAND ----------

# DBTITLE 1,Config
from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType, IntegerType

CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA  = "march_madness_2026"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Set True to recompute all features even if the output tables already exist
FORCE_RERUN = False

# COMMAND ----------

# DBTITLE 1,Check Existing Assets
existing_tables = {
    row.tableName
    for row in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}

SKIP_COMPUTE = "training_features" in existing_tables and not FORCE_RERUN
SKIP_TEAM_FEATURES = "team_season_features" in existing_tables and not FORCE_RERUN

if SKIP_COMPUTE:
    print("✓ training_features already exists — skipping feature engineering")
    print("  Set FORCE_RERUN = True in the Config cell to rebuild")
else:
    missing = [t for t in ["training_features", "team_season_features"] if t not in existing_tables]
    print(f"Will compute: {missing or 'all feature tables'}")

# COMMAND ----------

# DBTITLE 1,Load Base Tables
compact       = spark.table("mregularseason_compact")
detailed      = spark.table("mregularseason_detailed")
tourney       = spark.table("mncaatourney_compact")
seeds         = spark.table("mncaatourney_seeds")
teams         = spark.table("mteams")

if not SKIP_COMPUTE:
    seasons = compact.select("Season").distinct().orderBy("Season").collect()
    print(f"Seasons in regular season data: {seasons[0].Season} - {seasons[-1].Season}")
    tourney_seasons = tourney.select("Season").distinct().orderBy("Season").collect()
    print(f"Seasons in tourney data: {tourney_seasons[0].Season} - {tourney_seasons[-1].Season}")

# COMMAND ----------

# DBTITLE 1,Build Win/Loss Records Per Season
if not SKIP_COMPUTE:
    wins = (
        compact
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
        compact
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
        wins.join(losses, ["Season", "TeamID"], "outer")
        .fillna(0)
        .withColumn("games", F.col("wins") + F.col("losses"))
        .withColumn("win_pct", F.col("wins") / F.col("games"))
        .withColumn(
            "avg_pts_scored",
            (F.col("avg_pts_scored_w") * F.col("wins") + F.col("avg_pts_scored_l") * F.col("losses"))
            / F.col("games"),
        )
        .withColumn(
            "avg_pts_allowed",
            (F.col("avg_pts_allowed_w") * F.col("wins") + F.col("avg_pts_allowed_l") * F.col("losses"))
            / F.col("games"),
        )
        .withColumn(
            "avg_margin",
            (F.col("avg_margin_w") * F.col("wins") + F.col("avg_margin_l") * F.col("losses"))
            / F.col("games"),
        )
        .select("Season", "TeamID", "wins", "losses", "games", "win_pct",
                "avg_pts_scored", "avg_pts_allowed", "avg_margin")
    )
    print(f"Season records: {season_records.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Away Game Win Percentage
if not SKIP_COMPUTE:
    away_wins = (
        compact.filter(F.col("WLoc") == "A")
        .groupBy("Season", "WTeamID")
        .agg(F.count("*").alias("away_wins"))
        .withColumnRenamed("WTeamID", "TeamID")
    )

    away_losses = (
        compact.filter(F.col("WLoc") == "H")
        .groupBy("Season", "LTeamID")
        .agg(F.count("*").alias("away_losses"))
        .withColumnRenamed("LTeamID", "TeamID")
    )

    away_record = (
        away_wins.join(away_losses, ["Season", "TeamID"], "outer")
        .fillna(0)
        .withColumn("away_games", F.col("away_wins") + F.col("away_losses"))
        .withColumn(
            "away_win_pct",
            F.when(F.col("away_games") > 0, F.col("away_wins") / F.col("away_games")).otherwise(0.5),
        )
        .select("Season", "TeamID", "away_win_pct")
    )

# COMMAND ----------

# DBTITLE 1,Detailed Stats Features (Offensive/Defensive Efficiency)
if not SKIP_COMPUTE:
    detailed_wins = (
        detailed
        .groupBy("Season", "WTeamID")
        .agg(
            F.avg("WFGM").alias("w_fgm"), F.avg("WFGA").alias("w_fga"),
            F.avg("WFGM3").alias("w_fgm3"), F.avg("WFGA3").alias("w_fga3"),
            F.avg("WFTM").alias("w_ftm"), F.avg("WFTA").alias("w_fta"),
            F.avg("WOR").alias("w_or"), F.avg("WDR").alias("w_dr"),
            F.avg("WAst").alias("w_ast"), F.avg("WTO").alias("w_to"),
            F.avg("WStl").alias("w_stl"), F.avg("WBlk").alias("w_blk"),
            F.avg("WPF").alias("w_pf"),
            F.count("*").alias("w_detail_games"),
        )
        .withColumnRenamed("WTeamID", "TeamID")
    )

    detailed_losses = (
        detailed
        .groupBy("Season", "LTeamID")
        .agg(
            F.avg("LFGM").alias("l_fgm"), F.avg("LFGA").alias("l_fga"),
            F.avg("LFGM3").alias("l_fgm3"), F.avg("LFGA3").alias("l_fga3"),
            F.avg("LFTM").alias("l_ftm"), F.avg("LFTA").alias("l_fta"),
            F.avg("LOR").alias("l_or"), F.avg("LDR").alias("l_dr"),
            F.avg("LAst").alias("l_ast"), F.avg("LTO").alias("l_to"),
            F.avg("LStl").alias("l_stl"), F.avg("LBlk").alias("l_blk"),
            F.avg("LPF").alias("l_pf"),
            F.count("*").alias("l_detail_games"),
        )
        .withColumnRenamed("LTeamID", "TeamID")
    )

    detail_stats = (
        detailed_wins.join(detailed_losses, ["Season", "TeamID"], "outer")
        .fillna(0)
        .withColumn("detail_games", F.col("w_detail_games") + F.col("l_detail_games"))
    )

    for stat in ["fgm", "fga", "fgm3", "fga3", "ftm", "fta", "or", "dr", "ast", "to", "stl", "blk", "pf"]:
        w_col, l_col = f"w_{stat}", f"l_{stat}"
        detail_stats = detail_stats.withColumn(
            f"avg_{stat}",
            F.when(
                F.col("detail_games") > 0,
                (F.col(w_col) * F.col("w_detail_games") + F.col(l_col) * F.col("l_detail_games"))
                / F.col("detail_games"),
            ).otherwise(0),
        )

    detail_stats = (
        detail_stats
        .withColumn("fg_pct",      F.when(F.col("avg_fga")  > 0, F.col("avg_fgm")  / F.col("avg_fga")).otherwise(0))
        .withColumn("fg3_pct",     F.when(F.col("avg_fga3") > 0, F.col("avg_fgm3") / F.col("avg_fga3")).otherwise(0))
        .withColumn("ft_pct",      F.when(F.col("avg_fta")  > 0, F.col("avg_ftm")  / F.col("avg_fta")).otherwise(0))
        .withColumn("ast_to_ratio",F.when(F.col("avg_to")   > 0, F.col("avg_ast")  / F.col("avg_to")).otherwise(1))
        .withColumn("total_rebounds", F.col("avg_or") + F.col("avg_dr"))
        .select(
            "Season", "TeamID",
            "fg_pct", "fg3_pct", "ft_pct",
            "avg_or", "avg_dr", "total_rebounds",
            "avg_ast", "avg_to", "ast_to_ratio",
            "avg_stl", "avg_blk", "avg_pf",
        )
    )
    print(f"Detail stats: {detail_stats.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Elo Ratings
if not SKIP_COMPUTE:
    import pandas as pd
    import numpy as np

    compact_pd = compact.orderBy("Season", "DayNum").toPandas()

    K              = 32
    HOME_ADV       = 100
    INIT_ELO       = 1500
    SEASON_REGRESS = 0.75

    elo         = {}
    current_elo = {}
    prev_season = None

    for _, row in compact_pd.iterrows():
        season = int(row["Season"])
        w      = int(row["WTeamID"])
        l      = int(row["LTeamID"])
        loc    = row.get("WLoc", "N")

        if season != prev_season:
            if prev_season is not None:
                for team_id, e in current_elo.items():
                    elo[(prev_season, team_id)] = e
                mean_elo = np.mean(list(current_elo.values())) if current_elo else INIT_ELO
                for team_id in current_elo:
                    current_elo[team_id] = SEASON_REGRESS * current_elo[team_id] + (1 - SEASON_REGRESS) * mean_elo
            prev_season = season

        for t in [w, l]:
            if t not in current_elo:
                current_elo[t] = INIT_ELO

        elo_w = current_elo[w]
        elo_l = current_elo[l]
        elo_w_adj = elo_w + HOME_ADV if loc == "H" else elo_w
        elo_l_adj = elo_l + HOME_ADV if loc == "A" else elo_l

        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l_adj - elo_w_adj) / 400.0))
        current_elo[w] = elo_w + K * (1 - expected_w)
        current_elo[l] = elo_l + K * (0 - (1 - expected_w))

    if prev_season is not None:
        for team_id, e in current_elo.items():
            elo[(prev_season, team_id)] = e

    elo_df = spark.createDataFrame(
        pd.DataFrame([{"Season": s, "TeamID": t, "elo_rating": e} for (s, t), e in elo.items()])
    )
    print(f"Elo ratings: {elo_df.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Last 10 Games Form
if not SKIP_COMPUTE:
    w_spec = Window.partitionBy("Season", "TeamID").orderBy("DayNum").rowsBetween(-9, 0)

    games_as_team = (
        compact
        .select(F.col("Season"), F.col("DayNum"), F.col("WTeamID").alias("TeamID"), F.lit(1).alias("won"))
        .union(compact.select(F.col("Season"), F.col("DayNum"), F.col("LTeamID").alias("TeamID"), F.lit(0).alias("won")))
    )

    last_10 = (
        games_as_team
        .withColumn("recent_wins",  F.sum("won").over(w_spec))
        .withColumn("recent_games", F.count("won").over(w_spec))
        .withColumn("recent_form",  F.col("recent_wins") / F.col("recent_games"))
        .groupBy("Season", "TeamID")
        .agg(F.last("recent_form").alias("last10_win_pct"))
    )
    print(f"Last 10 form: {last_10.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Parse Tournament Seeds
if not SKIP_COMPUTE:
    seed_parsed = (
        seeds
        .withColumn("seed_region", F.substring("Seed", 1, 1))
        .withColumn("seed_num", F.regexp_extract("Seed", r"[WXYZ](\d+)", 1).cast(IntegerType()))
        .select("Season", "TeamID", "Seed", "seed_region", "seed_num")
    )
    print(f"Seeds: {seed_parsed.count():,} rows")

# COMMAND ----------

# DBTITLE 1,Massey Ordinal Rankings
if not SKIP_COMPUTE:
    try:
        massey = spark.table("mmassey_ordinals")
        massey_agg = (
            massey
            .groupBy("Season", "TeamID")
            .agg(
                F.avg("OrdinalRank").alias("avg_massey_rank"),
                F.min("OrdinalRank").alias("best_massey_rank"),
                F.count(F.lit(1)).alias("massey_count"),
            )
        )
        has_massey = True
        print(f"Massey ordinals: {massey_agg.count():,} rows")
    except Exception:
        has_massey = False
        print("Massey ordinals not available, skipping")

# COMMAND ----------

# DBTITLE 1,Assemble Team-Season Feature Table
if SKIP_TEAM_FEATURES:
    print("✓ team_season_features already exists — loading from table")
    team_features = spark.table(f"{CATALOG}.{SCHEMA}.team_season_features")
elif not SKIP_COMPUTE:
    team_features = (
        season_records
        .join(away_record,  ["Season", "TeamID"], "left")
        .join(detail_stats, ["Season", "TeamID"], "left")
        .join(elo_df,       ["Season", "TeamID"], "left")
        .join(last_10,      ["Season", "TeamID"], "left")
        .join(seed_parsed,  ["Season", "TeamID"], "left")
    )
    if has_massey:
        team_features = team_features.join(massey_agg, ["Season", "TeamID"], "left")

    team_features = team_features.fillna(0)
    print(f"Team features shape: {team_features.count():,} rows × {len(team_features.columns)} cols")

    team_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{CATALOG}.{SCHEMA}.team_season_features"
    )
    print(f"Saved → {CATALOG}.{SCHEMA}.team_season_features")

# COMMAND ----------

# DBTITLE 1,Build Pairwise Matchup Training Data
if SKIP_COMPUTE:
    print("✓ training_features already exists — loading from table")
    training_data = spark.table(f"{CATALOG}.{SCHEMA}.training_features")
else:
    tourney_matchups = (
        tourney
        .withColumn("TeamA", F.when(F.col("WTeamID") < F.col("LTeamID"), F.col("WTeamID")).otherwise(F.col("LTeamID")))
        .withColumn("TeamB", F.when(F.col("WTeamID") < F.col("LTeamID"), F.col("LTeamID")).otherwise(F.col("WTeamID")))
        .withColumn("team1_won", F.when(F.col("WTeamID") < F.col("LTeamID"), F.lit(1)).otherwise(F.lit(0)))
        .select("Season", "TeamA", "TeamB", "team1_won")
    )
    print(f"Tournament matchups: {tourney_matchups.count():,}")

    feature_cols = [c for c in team_features.columns if c not in ("Season", "TeamID", "Seed", "seed_region")]

    fa = team_features.select("Season", "TeamID", *[F.col(c).alias(f"a_{c}") for c in feature_cols]).withColumnRenamed("TeamID", "TeamA")
    fb = team_features.select("Season", "TeamID", *[F.col(c).alias(f"b_{c}") for c in feature_cols]).withColumnRenamed("TeamID", "TeamB")

    training_data = tourney_matchups.join(fa, ["Season", "TeamA"], "left").join(fb, ["Season", "TeamB"], "left")

    for c in feature_cols:
        training_data = training_data.withColumn(
            f"diff_{c}",
            F.coalesce(F.col(f"a_{c}"), F.lit(0)) - F.coalesce(F.col(f"b_{c}"), F.lit(0)),
        )

    training_data = training_data.filter(F.col("Season") >= 2003).fillna(0)
    print(f"Training data (2003+): {training_data.count():,} rows × {len(training_data.columns)} cols")

    training_data.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{CATALOG}.{SCHEMA}.training_features"
    )
    print(f"Saved → {CATALOG}.{SCHEMA}.training_features")

# COMMAND ----------

# DBTITLE 1,Feature Summary Statistics
display(
    training_data
    .select(*[c for c in training_data.columns if c.startswith("diff_")])
    .summary("count", "mean", "stddev", "min", "max")
)
