export interface Team {
  team_id: number
  team_name: string
  seed: string
  seed_num: number
  region?: string
}

export interface Prediction {
  id: string
  pred: number
  team1_name: string
  team2_name: string
  team1_seed?: string
  team2_seed?: string
}

export interface Matchup {
  team1: Team
  team2: Team
  team1_win_prob: number
  predicted_winner: Team
  region?: string
}

export interface BracketRound {
  round: string
  matchups: Matchup[]
}

export interface BracketResult {
  champion: Team | null
  rounds: BracketRound[]
}

export interface ModelMetrics {
  log_loss?: number
  accuracy?: number
  brier_score?: number
  baseline_log_loss?: number
  model_name?: string
  eval_date?: string
  holdout_season?: number
  message?: string
}

export interface RegionTeams {
  [region: string]: Team[]
}
