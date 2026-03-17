import { useEffect, useState } from 'react'
import { api } from '../api'
import type { ModelMetrics } from '../types'
import { TrendingUp, Target, Award, AlertCircle } from 'lucide-react'

export default function ModelMetricsPanel() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getMetrics()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-slate-400 animate-pulse">Loading metrics...</div>

  if (error || metrics?.message) {
    return (
      <div className="flex items-center gap-2 text-slate-400 text-sm p-4 border border-slate-700 rounded-lg">
        <AlertCircle className="w-4 h-4 flex-shrink-0" />
        <span>{metrics?.message || error}</span>
      </div>
    )
  }

  const improvement = metrics?.baseline_log_loss && metrics?.log_loss
    ? ((metrics.baseline_log_loss - metrics.log_loss) / metrics.baseline_log_loss * 100).toFixed(1)
    : null

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard
        icon={<TrendingUp className="w-5 h-5 text-green-400" />}
        label="Log Loss"
        value={metrics?.log_loss?.toFixed(4) ?? '—'}
        sub={improvement ? `${improvement}% better than baseline` : undefined}
        good={true}
      />
      <MetricCard
        icon={<Target className="w-5 h-5 text-blue-400" />}
        label="Accuracy"
        value={metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
        sub="on 2025 holdout"
      />
      <MetricCard
        icon={<Award className="w-5 h-5 text-yellow-400" />}
        label="Brier Score"
        value={metrics?.brier_score?.toFixed(4) ?? '—'}
        sub="lower is better"
      />
      <MetricCard
        icon={<TrendingUp className="w-5 h-5 text-slate-400" />}
        label="Baseline Log Loss"
        value={metrics?.baseline_log_loss?.toFixed(4) ?? '—'}
        sub="50% prediction"
      />
      {metrics?.model_name && (
        <div className="col-span-2 md:col-span-4 text-xs text-slate-400">
          Model: <span className="text-slate-300">{metrics.model_name}</span>
          {metrics.holdout_season && <> | Holdout: <span className="text-slate-300">{metrics.holdout_season} tournament</span></>}
        </div>
      )}
    </div>
  )
}

function MetricCard({ icon, label, value, sub, good }: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  good?: boolean
}) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}
