import { useEffect, useState } from 'react'
import { api } from '../api'
import type { BracketResult, BracketRound } from '../types'
import MatchupCard from './MatchupCard'
import { Trophy, RefreshCw } from 'lucide-react'

const ROUND_COLORS: Record<string, string> = {
  'Round of 64': 'border-slate-600',
  'Round of 32': 'border-blue-700',
  'Sweet 16': 'border-purple-700',
  'Elite 8': 'border-orange-700',
  'Final Four': 'border-yellow-600',
  'Championship': 'border-ncaa-gold',
}

export default function BracketView() {
  const [result, setResult] = useState<BracketResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeRound, setActiveRound] = useState<string>('Round of 64')

  const load = () => {
    setLoading(true)
    setError(null)
    api.simulateBracket()
      .then(r => {
        setResult(r)
        setLoading(false)
      })
      .catch(e => {
        setError(e.message)
        setLoading(false)
      })
  }

  useEffect(() => { load() }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        <RefreshCw className="w-6 h-6 animate-spin mr-2" />
        Simulating bracket...
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-400 mb-4">{error}</p>
        <p className="text-slate-400 text-sm">Run the ML pipeline notebooks to generate predictions first.</p>
      </div>
    )
  }

  if (!result) return null

  const rounds = result.rounds
  const roundNames = rounds.map(r => r.round)

  const activeRoundData = rounds.find(r => r.round === activeRound)

  return (
    <div>
      {/* Champion Banner */}
      {result.champion && (
        <div className="mb-6 p-4 bg-gradient-to-r from-ncaa-blue/30 to-ncaa-gold/20 border border-ncaa-gold/40 rounded-xl flex items-center gap-4">
          <Trophy className="w-8 h-8 text-ncaa-gold flex-shrink-0" />
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-widest">Predicted Champion</div>
            <div className="text-2xl font-bold text-ncaa-gold">{result.champion.team_name}</div>
            <div className="text-sm text-slate-400">Seed #{result.champion.seed_num} · {result.champion.region}</div>
          </div>
        </div>
      )}

      {/* Round Tabs */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {roundNames.map(name => (
          <button
            key={name}
            onClick={() => setActiveRound(name)}
            className={`px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
              activeRound === name
                ? 'bg-ncaa-blue text-white'
                : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
            }`}
          >
            {name}
            <span className="ml-1.5 text-xs opacity-60">
              ({rounds.find(r => r.round === name)?.matchups.length ?? 0})
            </span>
          </button>
        ))}
      </div>

      {/* Matchups Grid */}
      {activeRoundData && (
        <RoundGrid round={activeRoundData} />
      )}
    </div>
  )
}

function RoundGrid({ round }: { round: BracketRound }) {
  const byRegion: Record<string, typeof round.matchups> = {}
  for (const m of round.matchups) {
    const region = m.region ?? m.team1.region ?? 'National'
    if (!byRegion[region]) byRegion[region] = []
    byRegion[region].push(m)
  }

  const hasRegions = Object.keys(byRegion).length > 1

  if (!hasRegions) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {round.matchups.map((m, i) => (
          <MatchupCard key={i} matchup={m} />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {Object.entries(byRegion).map(([region, matchups]) => (
        <div key={region}>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">{region}</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {matchups.map((m, i) => (
              <MatchupCard key={i} matchup={m} compact={round.matchups.length > 16} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
