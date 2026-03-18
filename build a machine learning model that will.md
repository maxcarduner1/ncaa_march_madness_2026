build a machine learning model that will help me predict my bracket this year
use my databricks sandbox workspace (serverless-stable-82l7qq) and mlflow best practices for experiment tracking
see how to download data and descriptions of data: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data
create a new UC Schema in the serverless_stable_82l7qq_catalog catalog called "march_madness_2026" and download all the csvs into a volume in here, convert only the ones you need to tables in the same schema

Once we evaluate a series of models, pick the best ones based on historical performance using a holdout dataset that's appropriate for this tournament.

Next see how the predictions will need to work, so adapt model predictions appropriately:
Data Section 1 file: SampleSubmissionStage1.csv and SampleSubmissionStage2.csv
These files illustrate the submission file format. They reflect the simplest possible submission: a 50% winning percentage is predicted for each possible matchup. The Stage1 submission file lists all possible matchups from the last four years (seasons 2022-2025), and can be used to help develop your model. The Stage2 submission file illustrates all possible matchups for the current season, and you will need to predict these matchups for the actual tournament in March.

also build a databricks app to showcase the predictions for this year's ncaa march madness bracket

structure this repo using databricks asset bundles that you can deploy to the workspace and iterate on when model training


