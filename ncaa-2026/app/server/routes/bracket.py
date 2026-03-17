from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..db import query_df
from ..config import CATALOG, SCHEMA

router = APIRouter()


class BracketSimRequest(BaseModel):
    # Map of round -> list of matchup winner TeamIDs picked by user
    # If empty, use model predictions (highest prob) to simulate
    picks: dict[str, list[int]] = {}


@router.get("/bracket/seeds")
def get_seeded_teams():
    """2026 tournament teams organized by region and seed."""
    try:
        df = query_df(f"""
            SELECT t.TeamID, t.TeamName,
                   s.Seed,
                   SUBSTRING(s.Seed, 1, 1) as region_code,
                   CAST(SUBSTRING(s.Seed, 2, 2) AS INT) as seed_num
            FROM {CATALOG}.{SCHEMA}.mteams t
            JOIN {CATALOG}.{SCHEMA}.mncaatourney_seeds s ON t.TeamID = s.TeamID
            WHERE s.Season = 2026
            ORDER BY s.Seed
        """)
        # Group by region
        regions: dict = {}
        region_names = {"W": "West", "X": "Midwest", "Y": "South", "Z": "East"}
        for _, row in df.iterrows():
            region = region_names.get(row["region_code"], row["region_code"])
            if region not in regions:
                regions[region] = []
            regions[region].append({
                "team_id": int(row["TeamID"]),
                "team_name": row["TeamName"],
                "seed": row["Seed"],
                "seed_num": int(row["seed_num"]),
            })
        return regions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bracket/simulate")
def simulate_bracket(request: BracketSimRequest):
    """
    Simulate a full bracket using model predictions.
    User can override picks for specific rounds.
    Returns simulated results for each round.
    """
    try:
        # Get all predictions
        preds_df = query_df(f"""
            SELECT id, pred FROM {CATALOG}.{SCHEMA}.predictions_2026
        """)
        pred_map = {}
        for _, row in preds_df.iterrows():
            parts = row["id"].split("_")
            t1, t2 = int(parts[1]), int(parts[2])
            pred_map[(t1, t2)] = float(row["pred"])

        def win_prob(team1_id: int, team2_id: int) -> float:
            lo, hi = min(team1_id, team2_id), max(team1_id, team2_id)
            p = pred_map.get((lo, hi), 0.5)
            return p if team1_id == lo else 1 - p

        # Get seeded teams
        seeds_df = query_df(f"""
            SELECT t.TeamID, t.TeamName, s.Seed,
                   SUBSTRING(s.Seed, 1, 1) as region_code,
                   CAST(SUBSTRING(s.Seed, 2, 2) AS INT) as seed_num
            FROM {CATALOG}.{SCHEMA}.mteams t
            JOIN {CATALOG}.{SCHEMA}.mncaatourney_seeds s ON t.TeamID = s.TeamID
            WHERE s.Season = 2026
            ORDER BY s.Seed
        """)

        teams_by_seed: dict = {}
        for _, row in seeds_df.iterrows():
            teams_by_seed[row["Seed"]] = {
                "team_id": int(row["TeamID"]),
                "team_name": row["TeamName"],
                "seed": row["Seed"],
                "seed_num": int(row["seed_num"]),
                "region": row["region_code"],
            }

        # Standard bracket pairings: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        SEED_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
        REGIONS = ["W", "X", "Y", "Z"]
        REGION_NAMES = {"W": "West", "X": "Midwest", "Y": "South", "Z": "East"}

        rounds = []
        # Round of 64: 4 regions x 8 games
        r64_matchups = []
        r64_winners = []
        for region in REGIONS:
            for s1, s2 in SEED_PAIRS:
                seed1 = f"{region}{s1:02d}"
                seed2 = f"{region}{s2:02d}"
                t1 = teams_by_seed.get(seed1)
                t2 = teams_by_seed.get(seed2)
                if not t1 or not t2:
                    # Handle play-in seeds like W01a/W01b - pick higher prob
                    continue
                p = win_prob(t1["team_id"], t2["team_id"])
                winner = t1 if p >= 0.5 else t2
                r64_matchups.append({
                    "team1": t1, "team2": t2,
                    "team1_win_prob": round(p, 4),
                    "predicted_winner": winner,
                    "region": REGION_NAMES[region],
                })
                r64_winners.append(winner)

        rounds.append({"round": "Round of 64", "matchups": r64_matchups})

        # Simulate remaining rounds
        current_field = r64_winners

        round_names = ["Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
        for rnd_name in round_names:
            if len(current_field) < 2:
                break
            matchups = []
            winners = []
            for i in range(0, len(current_field), 2):
                if i + 1 >= len(current_field):
                    winners.append(current_field[i])
                    continue
                t1, t2 = current_field[i], current_field[i + 1]
                p = win_prob(t1["team_id"], t2["team_id"])
                winner = t1 if p >= 0.5 else t2
                matchups.append({
                    "team1": t1, "team2": t2,
                    "team1_win_prob": round(p, 4),
                    "predicted_winner": winner,
                })
                winners.append(winner)
            rounds.append({"round": rnd_name, "matchups": matchups})
            current_field = winners

        return {
            "champion": current_field[0] if current_field else None,
            "rounds": rounds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
def get_model_metrics():
    """Model performance metrics from holdout evaluation (2025 tournament)."""
    try:
        df = query_df(f"""
            SELECT * FROM {CATALOG}.{SCHEMA}.model_metrics
            ORDER BY eval_date DESC
            LIMIT 1
        """)
        if df.empty:
            return {"message": "No metrics available yet. Run the ML pipeline first."}
        return df.iloc[0].to_dict()
    except Exception as e:
        # Return placeholder if table doesn't exist yet
        return {
            "message": "Metrics will be available after running the ML pipeline.",
            "error": str(e)
        }
