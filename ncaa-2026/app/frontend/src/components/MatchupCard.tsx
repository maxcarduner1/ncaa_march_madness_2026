import type { Matchup } from '../types'

interface Props {
  matchup: Matchup
  compact?: boolean
}

export default function MatchupCard({ matchup, compact = false }: Props) {
  const { team1, team2, team1_win_prob, predicted_winner } = matchup
  const t2_prob = 1 - team1_win_prob

  const isT1Winner = predicted_winner.team_id === team1.team_id

  if (compact) {
    return (
      <div className="matchup-card text-xs">
        <div className={`team-row ${isT1Winner ? 'winner' : 'text-slate-400'}`}>
          <span className="seed-badge">{team1.seed_num}</span>
          <span className="flex-1 truncate">{team1.team_name}</span>
          <span className="text-slate-400">{(team1_win_prob * 100).toFixed(0)}%</span>
        </div>
        <div className="h-px bg-slate-700 my-0.5" />
        <div className={`team-row ${!isT1Winner ? 'winner' : 'text-slate-400'}`}>
          <span className="seed-badge">{team2.seed_num}</span>
          <span className="flex-1 truncate">{team2.team_name}</span>
          <span className="text-slate-400">{(t2_prob * 100).toFixed(0)}%</span>
        </div>
        <div className="mt-1.5 bg-slate-700 rounded-full h-1.5">
          <div className="prob-bar" style={{ width: `${team1_win_prob * 100}%` }} />
        </div>
      </div>
    )
  }

  return (
    <div className="matchup-card">
      <div className={`team-row ${isT1Winner ? 'winner' : 'text-slate-400'}`}>
        <span className="seed-badge">{team1.seed_num}</span>
        <span className="flex-1">{team1.team_name}</span>
        <span className="font-mono text-sm">{(team1_win_prob * 100).toFixed(1)}%</span>
      </div>
      <div className="my-1 bg-slate-700 rounded-full h-2">
        <div className="prob-bar h-2" style={{ width: `${team1_win_prob * 100}%` }} />
      </div>
      <div className={`team-row ${!isT1Winner ? 'winner' : 'text-slate-400'}`}>
        <span className="seed-badge">{team2.seed_num}</span>
        <span className="flex-1">{team2.team_name}</span>
        <span className="font-mono text-sm">{(t2_prob * 100).toFixed(1)}%</span>
      </div>
      <div className="mt-2 text-center text-xs text-slate-500">
        Predicted: <span className="text-ncaa-gold font-semibold">{predicted_winner.team_name}</span>
      </div>
    </div>
  )
}
