# Databricks notebook source
# MAGIC %md
# MAGIC # NCAA March Madness 2026 - Data Ingestion
# MAGIC
# MAGIC Unzips pre-uploaded competition data from UC Volume and loads into Delta tables.
# MAGIC Skips assets that already exist unless `FORCE_RERUN = True`.
# MAGIC
# MAGIC **Catalog:** `serverless_stable_82l7qq_catalog`
# MAGIC **Schema:** `march_madness_2026`
# MAGIC **Volume:** `raw_data` (zip already uploaded via `databricks fs cp`)

# COMMAND ----------

# DBTITLE 1,Config
CATALOG = "serverless_stable_82l7qq_catalog"
SCHEMA  = "march_madness_2026"
VOLUME  = "raw_data"

# Set True to re-unzip and overwrite all tables even if they already exist
FORCE_RERUN = False

# COMMAND ----------

# DBTITLE 1,Setup Unity Catalog Namespace
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {VOLUME}")

volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
zip_path    = f"{volume_path}/march-machine-learning-mania-2026.zip"
print(f"Volume path: {volume_path}")
print(f"Zip path:    {zip_path}")

# COMMAND ----------

# DBTITLE 1,Collect Existing Assets
import os

existing_tables = {
    row.tableName
    for row in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}
print(f"Tables already in schema: {len(existing_tables)}")
if existing_tables:
    for t in sorted(existing_tables):
        print(f"  • {t}")

# Check whether CSVs are already extracted by probing a key file
KEY_CSV = f"{volume_path}/MTeams.csv"
csvs_extracted = os.path.exists(KEY_CSV)
print(f"\nCSVs already extracted: {csvs_extracted}")

# COMMAND ----------

# DBTITLE 1,Unzip Competition Data into Volume
import subprocess

if csvs_extracted and not FORCE_RERUN:
    print("✓ CSVs already present in volume — skipping unzip")
    print("  (Set FORCE_RERUN = True to force re-extract)")
else:
    assert os.path.exists(zip_path), (
        f"Zip not found at {zip_path}.\n"
        "Upload it first with:\n"
        f"  databricks fs cp march-machine-learning-mania-2026.zip dbfs:{zip_path} --profile sandbox"
    )
    print(f"Extracting {zip_path} …")
    result = subprocess.run(
        ["unzip", "-o", zip_path, "-d", volume_path],
        capture_output=True, text=True,
    )
    output = result.stdout
    print(output[-3000:] if len(output) > 3000 else output)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"unzip failed (exit {result.returncode})")

# COMMAND ----------

# DBTITLE 1,List CSV Files in Volume
csv_files = [f for f in dbutils.fs.ls(f"dbfs:{volume_path}") if f.name.endswith(".csv")]
print(f"Found {len(csv_files)} CSV files in volume:")
for f in sorted(csv_files, key=lambda x: x.name):
    print(f"  {f.name}  ({f.size:,} bytes)")

# COMMAND ----------

# DBTITLE 1,Load M* and Submission CSVs into Delta Tables
# Focus on Men's tournament data (M prefix) + shared tables + submission files.
# Women's data (W prefix) is skipped to keep the schema clean.

TABLE_MAP = {
    "Cities.csv":                          "cities",
    "Conferences.csv":                     "conferences",
    "MConferenceTourneyGames.csv":         "mconference_tourney_games",
    "MGameCities.csv":                     "mgame_cities",
    "MMasseyOrdinals.csv":                 "mmassey_ordinals",
    "MNCAATourneyCompactResults.csv":      "mncaatourney_compact",
    "MNCAATourneyDetailedResults.csv":     "mncaatourney_detailed",
    "MNCAATourneySeedRoundSlots.csv":      "mncaatourney_seed_round_slots",
    "MNCAATourneySeeds.csv":               "mncaatourney_seeds",
    "MNCAATourneySlots.csv":               "mncaatourney_slots",
    "MRegularSeasonCompactResults.csv":    "mregularseason_compact",
    "MRegularSeasonDetailedResults.csv":   "mregularseason_detailed",
    "MSeasons.csv":                        "mseasons",
    "MSecondaryTourneyCompactResults.csv": "msecondary_tourney_compact",
    "MSecondaryTourneyTeams.csv":          "msecondary_tourney_teams",
    "MTeamCoaches.csv":                    "mteam_coaches",
    "MTeamConferences.csv":                "mteam_conferences",
    "MTeamSpellings.csv":                  "mteam_spellings",
    "MTeams.csv":                          "mteams",
    "SampleSubmissionStage1.csv":          "sample_submission_stage1",
    "SampleSubmissionStage2.csv":          "sample_submission_stage2",
}

not_in_zip = []

for csv_name, table_name in TABLE_MAP.items():
    # Skip if table already exists (and not forcing rerun)
    if table_name in existing_tables and not FORCE_RERUN:
        print(f"  → {table_name} already exists, skipping")
        continue

    csv_path = f"{volume_path}/{csv_name}"
    if not os.path.exists(csv_path):
        not_in_zip.append(csv_name)
        continue

    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_path)
        )
        full_table = f"{CATALOG}.{SCHEMA}.{table_name}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table)
        print(f"  ✓ {table_name}: {df.count():,} rows × {len(df.columns)} cols")
    except Exception as e:
        print(f"  ✗ {csv_name}: {e}")

if not_in_zip:
    print(f"\nNot found in zip: {not_in_zip}")

# COMMAND ----------

# DBTITLE 1,Verify Key Tables
print("=== Data Quality Check ===")
key_tables = [
    "mteams", "mregularseason_compact", "mregularseason_detailed",
    "mncaatourney_compact", "mncaatourney_seeds",
    "mmassey_ordinals", "sample_submission_stage1", "sample_submission_stage2",
]
for tbl in key_tables:
    try:
        cnt = spark.table(f"{CATALOG}.{SCHEMA}.{tbl}").count()
        seasons = spark.sql(
            f"SELECT MIN(Season) as min_s, MAX(Season) as max_s FROM {CATALOG}.{SCHEMA}.{tbl}"
        ).collect()
        season_range = (
            f"  seasons {seasons[0].min_s}–{seasons[0].max_s}"
            if seasons[0].min_s else ""
        )
        print(f"  ✓ {tbl}: {cnt:,} rows{season_range}")
    except Exception as e:
        print(f"  ✗ {tbl}: {e}")

print("\nData ingestion complete!")
