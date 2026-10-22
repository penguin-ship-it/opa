import { useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Brain, RefreshCw, Upload, CheckCircle, AlertTriangle, Clock, HardDrive, Table2 } from 'lucide-react'
import api from '../services/api'
import { describeLoadError } from '../utils/loadError'

interface ModelInfo {
  model_name: string
  artifact_path: string
  exists: boolean
  last_modified: string | null
  size_kb: number | null
  feature_cols: string[]
}

interface TrainResult {
  success: boolean
  method: string
  accuracy: number
  positive_rate: number
  training_rows: number
  providers_updated: number
  feature_importance: Record<string, number>
  provider_scores: Record<string, number>
  trained_at: string
}

const CSV_COLUMNS = [
  { name: 'provider_npi',            example: '1234567890', type: 'string'    },
  { name: 'avg_units_per_line',      example: '2.3',        type: 'float ≥1'  },
  { name: 'high_value_cpt_ratio',    example: '0.45',       type: 'float 0–1' },
  { name: 'multi_line_claim_ratio',  example: '0.52',       type: 'float 0–1' },
  { name: 'modifier_usage_rate',     example: '0.38',       type: 'float 0–1' },
  { name: 'same_day_multi_cpt_rate', example: '0.27',       type: 'float 0–1' },
  { name: 'prior_overpayment_rate',  example: '0.14',       type: 'float 0–1' },
  { name: 'specialty_peer_deviation',example: '1.2',        type: 'float'     },
  { name: 'had_confirmed_overpayment', example: '0',        type: '0 or 1'    },
]

async function fetchInfo(): Promise<ModelInfo> {
  const res = await api.get<ModelInfo>('/ml/info')
  return res.data
}

async function runTrain(): Promise<TrainResult> {
  const res = await api.post<TrainResult>('/ml/train')
  return res.data
}

async function uploadAndTrain(file: File): Promise<TrainResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post<TrainResult>('/ml/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

function FeatureBar({ name, value }: { name: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-600 font-mono">{name}</span>
        <span className="font-semibold text-gray-800">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2">
        <div
          className="h-2 rounded-full bg-indigo-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function ProviderScoreRow({ npi, score }: { npi: string; score: number }) {
  const color = score >= 0.7 ? 'text-red-600' : score >= 0.4 ? 'text-yellow-600' : 'text-green-600'
  const bar   = score >= 0.7 ? 'bg-red-400'   : score >= 0.4 ? 'bg-yellow-400'   : 'bg-green-400'
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-xs text-gray-500 w-28 flex-shrink-0">{npi}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${bar}`} style={{ width: `${Math.round(score * 100)}%` }} />
      </div>
      <span className={`text-xs font-bold w-10 text-right ${color}`}>{Math.round(score * 100)}%</span>
    </div>
  )
}

export default function TrainModelPage() {
  const fileRef = useRef<HTMLInputElement>(null)
  const [result, setResult] = useState<TrainResult | null>(null)

  const { data: info, refetch: refetchInfo, error: infoError } = useQuery<ModelInfo>({
    queryKey: ['ml-info'],
    queryFn: fetchInfo,
  })

  const trainMutation = useMutation({
    mutationFn: runTrain,
    onSuccess: (data) => { setResult(data); refetchInfo() },
  })

  const uploadMutation = useMutation({
    mutationFn: uploadAndTrain,
    onSuccess: (data) => { setResult(data); refetchInfo() },
  })

  const isPending = trainMutation.isPending || uploadMutation.isPending
  const error = trainMutation.error || uploadMutation.error

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    uploadMutation.mutate(file)
    e.target.value = ''
  }

  const sortedFeatures = result
    ? Object.entries(result.feature_importance).sort((a, b) => b[1] - a[1])
    : []

  const sortedProviders = result
    ? Object.entries(result.provider_scores).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="flex flex-col gap-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">ML Model Training</h1>
        <p className="text-sm text-gray-500 mt-1">
          Retrain the billing variance RandomForest classifier and push updated scores to all providers.
        </p>
      </div>

      {/* Model status card */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-4 h-4 text-indigo-500" />
          <h2 className="text-sm font-bold text-gray-900">Current Model</h2>
        </div>

        {info ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5">
            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Status</p>
              {info.exists ? (
                <span className="inline-flex items-center gap-1 text-sm font-bold text-green-600">
                  <CheckCircle className="w-4 h-4" /> Loaded
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-sm font-bold text-red-500">
                  <AlertTriangle className="w-4 h-4" /> Missing
                </span>
              )}
            </div>

            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Last Trained</p>
              <p className="text-sm font-bold text-gray-800">
                {info.last_modified
                  ? new Date(info.last_modified).toLocaleDateString()
                  : '—'}
              </p>
            </div>

            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Artifact Size</p>
              <p className="text-sm font-bold text-gray-800">
                {info.size_kb != null ? `${info.size_kb} KB` : '—'}
              </p>
            </div>

            <div className="bg-gray-50 rounded-xl p-3 text-center">
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Features</p>
              <p className="text-sm font-bold text-gray-800">{info.feature_cols.length}</p>
            </div>
          </div>
        ) : infoError ? (
          <p className="text-sm text-gray-400 mb-5">{describeLoadError(infoError, 'Model artifact info')}</p>
        ) : (
          <div className="h-20 bg-gray-50 rounded-xl animate-pulse mb-5" />
        )}

        {/* Feature list */}
        {info && (
          <div className="border-t border-gray-100 pt-4">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Input Features</p>
            <div className="flex flex-wrap gap-2">
              {info.feature_cols.map((f) => (
                <span key={f} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono font-medium bg-indigo-50 text-indigo-700 border border-indigo-100">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
        <h2 className="text-sm font-bold text-gray-900">Training Actions</h2>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => trainMutation.mutate()}
            disabled={isPending}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#FE017D] text-white
                       text-sm font-semibold rounded-lg hover:bg-[#e5006f]
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-sm"
          >
            {trainMutation.isPending ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Training…
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Retrain on Seed Data
              </>
            )}
          </button>

          <button
            onClick={() => fileRef.current?.click()}
            disabled={isPending}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-gray-300 text-gray-700
                       text-sm font-semibold rounded-lg hover:bg-gray-50
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {uploadMutation.isPending ? (
              <>
                <span className="w-4 h-4 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                Uploading…
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Upload CSV &amp; Retrain
              </>
            )}
          </button>
          <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
        </div>

        <p className="text-xs text-gray-400">
          Seed data: 5,000 synthetic rows across 10 provider profiles (RandomForest, 100 estimators, 5-fold CV).
          Upload a CSV with the same 7 feature columns + <span className="font-mono">had_confirmed_overpayment</span> to train on real data.
        </p>

        {/* CSV schema reference */}
        <div className="border-t border-gray-100 pt-4">
          <div className="flex items-center gap-2 mb-3">
            <Table2 className="w-3.5 h-3.5 text-gray-400" />
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Expected CSV Structure</p>
          </div>
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  {CSV_COLUMNS.map((col) => (
                    <th
                      key={col.name}
                      className="px-3 py-2 text-left font-semibold text-gray-700 whitespace-nowrap border-r border-gray-200 last:border-r-0"
                    >
                      {col.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="bg-white border-b border-gray-100">
                  {CSV_COLUMNS.map((col) => (
                    <td
                      key={col.name}
                      className="px-3 py-2 text-gray-400 whitespace-nowrap border-r border-gray-200 last:border-r-0"
                    >
                      {col.example}
                    </td>
                  ))}
                </tr>
                <tr className="bg-gray-50/50">
                  {CSV_COLUMNS.map((col) => (
                    <td
                      key={col.name}
                      className="px-3 py-1.5 text-gray-300 whitespace-nowrap border-r border-gray-200 last:border-r-0 italic"
                    >
                      {col.type}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-gray-400 mt-2">
            All feature columns are <span className="font-mono">float</span> except <span className="font-mono">provider_npi</span> (string) and <span className="font-mono">had_confirmed_overpayment</span> (0 or 1).
            <span className="font-mono ml-1">specialty_peer_deviation</span> may be negative.
          </p>
        </div>

        {error && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-3 py-2.5">
            <AlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-red-700">
              {(error as any)?.response?.data?.detail ?? (error as Error)?.message ?? 'Training failed.'}
            </p>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Summary metrics */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <div className="flex items-center gap-2 mb-4">
              <CheckCircle className="w-4 h-4 text-green-500" />
              <h2 className="text-sm font-bold text-gray-900">Training Results</h2>
              <span className="ml-auto text-xs text-gray-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {new Date(result.trained_at).toLocaleTimeString()}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
              {[
                { label: 'CV Accuracy',       value: `${(result.accuracy * 100).toFixed(1)}%` },
                { label: 'Positive Rate',     value: `${(result.positive_rate * 100).toFixed(1)}%` },
                { label: 'Training Rows',     value: result.training_rows.toLocaleString() },
                { label: 'Providers Updated', value: result.providers_updated.toString() },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-50 rounded-xl p-3 text-center">
                  <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">{label}</p>
                  <p className="text-lg font-bold text-gray-900">{value}</p>
                </div>
              ))}
            </div>

            {/* Feature importance */}
            <div className="border-t border-gray-100 pt-4 space-y-3">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Feature Importance</p>
              {sortedFeatures.map(([name, val]) => (
                <FeatureBar key={name} name={name} value={val} />
              ))}
            </div>
          </div>

          {/* Provider scores */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <div className="flex items-center gap-2 mb-4">
              <HardDrive className="w-4 h-4 text-indigo-400" />
              <h2 className="text-sm font-bold text-gray-900">Updated Provider Scores</h2>
            </div>
            <div className="space-y-2.5">
              {sortedProviders.map(([npi, score]) => (
                <ProviderScoreRow key={npi} npi={npi} score={score} />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
