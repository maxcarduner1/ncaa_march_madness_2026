from fastapi import APIRouter, HTTPException
from ..db import query_df
from ..config import CATALOG, SCHEMA
import pandas as pd

router = APIRouter()


@router.get("/predictions")
def get_predictions():
    """All 2026 matchup win probabilities."""
    try:
        df = query_df(f"""
            SELECT p.id, p.pred,
                   t1.TeamName as team1_name,
                   t2.TeamName as team2_name,
                   s1.Seed as team1_seed,
                   s2.Seed as team2_seed
            FROM {CATALOG}.{SCHEMA}.predictions_2026 p
            JOIN {CATALOG}.{SCHEMA}.mteams t1
              ON CAST(split(p.id, '_')[1] AS INT) = t1.TeamID
            JOIN {CATALOG}.{SCHEMA}.mteams t2
              ON CAST(split(p.id, '_')[2] AS INT) = t2.TeamID
            LEFT JOIN {CATALOG}.{SCHEMA}.mncaatourney_seeds s1
              ON s1.Season = 2026 AND CAST(split(p.id, '_')[1] AS INT) = s1.TeamID
            LEFT JOIN {CATALOG}.{SCHEMA}.mncaatourney_seeds s2
              ON s2.Season = 2026 AND CAST(split(p.id, '_')[2] AS INT) = s2.TeamID
            ORDER BY p.id
        """)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{team1_id}/{team2_id}")
def get_matchup_prediction(team1_id: int, team2_id: int):
    """Win probability for a specific matchup."""
    lo, hi = min(team1_id, team2_id), max(team1_id, team2_id)
    matchup_id = f"2026_{lo}_{hi}"
    try:
        df = query_df(f"""
            SELECT p.id, p.pred,
                   t1.TeamName as team1_name,
                   t2.TeamName as team2_name
            FROM {CATALOG}.{SCHEMA}.predictions_2026 p
            JOIN {CATALOG}.{SCHEMA}.mteams t1
              ON {lo} = t1.TeamID
            JOIN {CATALOG}.{SCHEMA}.mteams t2
              ON {hi} = t2.TeamID
            WHERE p.id = '{matchup_id}'
        """)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Matchup {matchup_id} not found")
        row = df.iloc[0].to_dict()
        # Flip probability if caller put teams in reverse order
        if team1_id != lo:
            row["pred"] = 1.0 - row["pred"]
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teams")
def get_teams():
    """All 2026 tournament teams with seeds."""
    try:
        df = query_df(f"""
            SELECT t.TeamID, t.TeamName, s.Seed, s.Season
            FROM {CATALOG}.{SCHEMA}.mteams t
            JOIN {CATALOG}.{SCHEMA}.mncaatourney_seeds s ON t.TeamID = s.TeamID
            WHERE s.Season = 2026
            ORDER BY s.Seed
        """)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
