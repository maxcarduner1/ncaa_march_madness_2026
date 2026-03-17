import { useState } from 'react'
import BracketView from './components/Bracket'
import ModelMetricsPanel from './components/ModelMetrics'
import { Trophy, BarChart2, GitBranch } from 'lucide-react'

type Tab = 'bracket' | 'metrics'

export default function App() {
  const [tab, setTab] = useState<Tab>('bracket')

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-ncaa-blue border-b border-slate-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Trophy className="w-7 h-7 text-ncaa-gold" />
            <div>
              <h1 className="text-xl font-bold text-white">NCAA March Madness 2026</h1>
              <p className="text-xs text-blue-200">ML-Powered Bracket Predictions</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-blue-200">
            <GitBranch className="w-4 h-4" />
            <span>Databricks AutoML + MLflow</span>
          </div>
        </div>
      </header>

      {/* Nav Tabs */}
      <nav className="bg-slate-800 border-b border-slate-700 px-6">
        <div className="max-w-7xl mx-auto flex gap-1">
          <TabButton
            active={tab === 'bracket'}
            onClick={() => setTab('bracket')}
            icon={<Trophy className="w-4 h-4" />}
            label="Bracket Predictions"
          />
          <TabButton
            active={tab === 'metrics'}
            onClick={() => setTab('metrics')}
            icon={<BarChart2 className="w-4 h-4" />}
            label="Model Performance"
          />
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {tab === 'bracket' && <BracketView />}
        {tab === 'metrics' && (
          <div>
            <h2 className="text-xl font-semibold text-white mb-6">Model Performance Metrics</h2>
            <p className="text-slate-400 text-sm mb-6">
              Evaluated on the 2025 NCAA tournament (holdout set). Model trained on seasons 2003–2024.
            </p>
            <ModelMetricsPanel />
          </div>
        )}
      </main>
    </div>
  )
}

function TabButton({ active, onClick, icon, label }: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
        active
          ? 'border-ncaa-gold text-ncaa-gold'
          : 'border-transparent text-slate-400 hover:text-white'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
