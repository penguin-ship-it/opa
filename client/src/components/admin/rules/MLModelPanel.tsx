import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import api from '../../../services/api'
import { formatDate } from '../../../utils/dateUtils'
import { card } from '../../../utils/designSystem'
import ModelTuningPanel from '../ModelTuningPanel'
import Stage2ModelPanel from '../Stage2ModelPanel'
import CapacityPlanPanel from '../CapacityPlanPanel'
import TrainModelPage from '../../../pages/TrainModelPage'
import { describeLoadError } from '../../../utils/loadError'

interface MLModelInfo {
  version: string; trained_at: string; accuracy: number
  precision: number; recall: number; f1_score: number
  auc_roc: number; training_samples: number
  brier_raw?: number | null; brier_calibrated?: number | null
  decision_threshold?: number | null
  feature_importance?: Record<string, number>
}

const FEATURE_LABEL: Record<string, string> = {
  avg_units_per_line: 'Avg units/line', high_value_cpt_ratio: 'High-risk CPT',
  multi_line_claim_ratio: 'Multi-line claims', modifier_usage_rate: 'Modifier usage',
  same_day_multi_cpt_rate: 'Same-day multi-CPT', prior_overpayment_rate: 'Prior overpayment',
  specialty_peer_deviation: 'Peer deviation',
}

export default function MLModelPanel() {
  const qc = useQueryClient()
  const [confirmRetrain, setConfirmRetrain] = useState(false)

  const { data: modelInfo, isLoading, error } = useQuery<MLModelInfo>({
    queryKey: ['admin', 'model'],
    queryFn: async () => (await api.get<MLModelInfo>('/admin/model')).data,
  })

  const retrainMutation = useMutation({
    mutationFn: async () => (await api.post<MLModelInfo>('/admin/model/retrain')).data,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['admin', 'model'] }); setConfirmRetrain(false) },
  })

  return (
    <div className="space-y-4">
      {isLoading ? (
        <div className="h-56 bg-white rounded-xl border border-gray-200 animate-pulse" />
      ) : modelInfo ? (
        <div className={card}>
          <div className="flex items-start justify-between mb-5">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-gray-900">Stage 1 — Provider Reputation</h2>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 font-mono">v{modelInfo.version}</span>
              </div>
              <p className="text-sm text-gray-500 mt-0.5">
                Scores each provider’s billing behavior (P_risk) · trained {formatDate(modelInfo.trained_at)} · {modelInfo.training_samples.toLocaleString()} samples
              </p>
            </div>
            <button onClick={() => setConfirmRetrain(true)}
              className="inline-flex items-center gap-1.5 px-4 py-2 bg-[#FE017D] text-white text-sm rounded-lg hover:bg-[#e5006f] transition-colors">
              <RefreshCw className="w-3.5 h-3.5" /> Retrain
            </button>
          </div>

          {/* Primary — Stage 1 outputs a calibrated score used as a FEATURE, so
              ranking (AUC) and calibration (Brier) are what matter. */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">AUC-ROC</p>
              <p className="text-2xl font-bold text-gray-900">{(modelInfo.auc_roc * 100).toFixed(1)}%</p>
              <p className="text-[11px] text-gray-400 mt-1">how well it ranks providers</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Brier (raw)</p>
              <p className="text-2xl font-bold text-gray-900">{modelInfo.brier_raw != null ? modelInfo.brier_raw.toFixed(4) : '—'}</p>
              <p className="text-[11px] text-gray-400 mt-1">before calibration</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Brier (calibrated)</p>
              <p className="text-2xl font-bold text-gray-900">{modelInfo.brier_calibrated != null ? modelInfo.brier_calibrated.toFixed(4) : '—'}</p>
              <p className="text-[11px] text-gray-400 mt-1">lower = truer probabilities</p>
            </div>
          </div>
          <p className="text-[11px] text-gray-400 mt-2">
            Stage 1 outputs a calibrated score (P_risk) consumed as a <span className="font-semibold">feature</span> by Stage 2 — it never flags claims,
            so ranking (AUC) and calibration (Brier) are the metrics that matter.
          </p>

          {/* Secondary — cutoff diagnostics, de-emphasized (no flagging happens here). */}
          <div className="mt-4">
            <p className="text-[11px] text-gray-400 uppercase tracking-wider mb-2">
              Diagnostics at a nominal cutoff{modelInfo.decision_threshold != null ? ` (${modelInfo.decision_threshold.toFixed(2)})` : ''} — not an operating point
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {([
                ['Accuracy', modelInfo.accuracy], ['Precision', modelInfo.precision],
                ['Recall', modelInfo.recall], ['F1 Score', modelInfo.f1_score],
              ] as [string, number][]).map(([label, val]) => (
                <div key={label} className="bg-gray-50/60 rounded-lg p-3 text-center">
                  <p className="text-[11px] text-gray-400 uppercase tracking-wider">{label}</p>
                  <p className="text-base font-semibold text-gray-500">{(val * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>

          {modelInfo.feature_importance && Object.keys(modelInfo.feature_importance).length > 0 && (() => {
            const entries = Object.entries(modelInfo.feature_importance)
            const sum = entries.reduce((a, [, v]) => a + v, 0) || 1
            const max = Math.max(...entries.map(([, v]) => v))
            const sorted = [...entries].sort(([, a], [, b]) => b - a)
            return (
              <div className="mt-6 pt-5 border-t border-gray-100">
                <h3 className="text-sm font-bold text-gray-900 mb-3">Feature importances</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                  {sorted.map(([feat, val]) => {
                    const pct = (val / sum) * 100
                    const barPct = max > 0 ? (val / max) * 100 : 0
                    return (
                      <div key={feat} className="bg-gray-50 rounded-xl p-4 text-center">
                        <p className="text-xs text-gray-400 uppercase tracking-wider mb-1 truncate">
                          {FEATURE_LABEL[feat] ?? feat}
                        </p>
                        <p className="text-lg font-bold text-gray-900">{pct.toFixed(1)}%</p>
                        <div className="mt-2.5 w-full bg-gray-200 rounded-full h-1.5">
                          <div className="h-1.5 rounded-full bg-[#FE017D]" style={{ width: `${barPct}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })()}
        </div>
      ) : (
        <p className="text-sm text-gray-400">{describeLoadError(error, 'Stage 1 model info')}</p>
      )}

      {confirmRetrain && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-sm w-full p-6">
            <h3 className="font-semibold text-gray-900 mb-2">Retrain Model?</h3>
            <p className="text-sm text-gray-500 mb-5">This will trigger a new training job. May take several minutes.</p>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setConfirmRetrain(false)} className="px-4 py-2 text-sm border border-gray-200 rounded-lg hover:bg-gray-50">Cancel</button>
              <button onClick={() => retrainMutation.mutate()} disabled={retrainMutation.isPending}
                className="px-4 py-2 text-sm bg-[#FE017D] text-white rounded-lg hover:bg-[#e5006f] disabled:opacity-60">
                {retrainMutation.isPending ? 'Retraining…' : 'Start Retraining'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stage 1 tuning */}
      <ModelTuningPanel />

      {/* Pipeline divider → Stage 2 */}
      <div className="flex items-center gap-3 pt-2">
        <div className="h-px flex-1 bg-gray-200" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Pipeline · Stage 2</span>
        <div className="h-px flex-1 bg-gray-200" />
      </div>
      <Stage2ModelPanel />

      {/* Pipeline divider → Audit plan */}
      <div className="flex items-center gap-3 pt-2">
        <div className="h-px flex-1 bg-gray-200" />
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Prioritization · Audit plan</span>
        <div className="h-px flex-1 bg-gray-200" />
      </div>
      <CapacityPlanPanel />

      <div className="pt-2"><TrainModelPage /></div>
    </div>
  )
}
