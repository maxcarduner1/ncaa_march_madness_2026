import type { Prediction, BracketResult, ModelMetrics, RegionTeams } from './types'

const BASE = '/api'

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`API ${path}: ${res.status} ${err}`)
  }
  return res.json()
}

export const api = {
  getPredictions: () => fetchJSON<Prediction[]>('/predictions'),
  getTeams: () => fetchJSON<RegionTeams>('/bracket/seeds'),
  simulateBracket: (picks = {}) =>
    fetchJSON<BracketResult>('/bracket/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ picks }),
    }),
  getMetrics: () => fetchJSON<ModelMetrics>('/metrics'),
}
