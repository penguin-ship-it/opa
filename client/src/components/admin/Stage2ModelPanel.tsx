import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Layers, FlaskConical, Save, CheckCircle, AlertTriangle } from 'lucide-react'
import api from '../../services/api'

// ── Types ────────────────────────────────────────────────────────────────────

interface Stage2Result {
  features: string[]
  params: Record<string, unknown>
  positive_rate: number
  training_rows: number
  accuracy: number
  precision: number | null
  recall: number | null
  f1_score: number | null
  f2_score: number | null
  auc_roc: number | null
  threshold: number | null
  calibration_method: string
  brier_raw: number | null
  brier_calibrated: number | null
  feature_importance: Record<string, number>
  trained_at?: string
}

interface Stage2Config {
  n_estimators: number
  max_depth: number | null
  min_samples_split: number
  min_samples_leaf: number
  max_features: string | null
  max_leaf_nodes: number | null
  bootstrap: boolean
  class_weight: string | null
  criterion: string
  decision_threshold_mode: string
  manual_threshold: number | null
  calibration_method: string
}

const DEFAULT_CONFIG: Stage2Config = {
  n_estimators: 300, max_depth: null, min_samples_split: 2, min_samples_leaf: 20,
  max_features: 'sqrt', max_leaf_nodes: null, bootstrap: true, class_weight: null,
  criterion: 'gini', decision_threshold_mode: 'auto_f2', manual_threshold: null,
  calibration_method: 'sigmoid',
}

const HINT: Record<string, string> = {
  n_estimators: 'Trees in the forest. More = stabler but slower (10–2000).',
  max_depth: 'Max depth per tree. Empty = unlimited. Lower fights overfitting (1–100).',
  min_samples_split: 'Min samples to split an internal node (≥2).',
  min_samples_leaf: 'Min samples at a leaf. Higher smooths predictions (1–200).',
  max_leaf_nodes: 'Cap on leaves per tree. Empty = unlimited (2–10000).',
  max_features: 'Features considered at each split. sqrt/log2 add randomness; "all" uses every feature.',
  criterion: 'Split-quality function.',
  class_weight: 'Re-weight classes. The pipeline already SMOTE-balances, so muted effect.',
  bootstrap: 'Sample rows with replacement per tree. Off = each tree sees the full set.',
  decision_threshold_mode: 'auto_f2 sweeps for the F2-optimal cutoff; manual pins it.',
  manual_threshold: 'Probability cutoff used when mode = manual (0–1).',
}

interface TrialRun extends Stage2Result {
  attempt: number
}

// F-code → plain-English feature description (spec v1.3 §6).
const FEATURES: { key: string; name: string; desc: string }[] = [
  { key: 'P_risk', name: 'Provider reputation (P_risk)', desc: 'The calibrated Stage 1 provider score, injected as a feature — the cross-stage hand-off.' },
  { key: 'F4', name: 'High-Value CPT Ratio', desc: 'Share of the claim’s lines that use a high-risk CPT code.' },
  { key: 'F5', name: 'Modifier Value-Add Ratio', desc: '$ from payment-driving modifiers above the fee schedule ÷ total allowed.' },
  { key: 'F7', name: 'Avg Units / Line (specialty-adj.)', desc: 'Mean units per line ÷ the provider specialty’s expected norm.' },
  { key: 'F8', name: 'Multi-Line Flag', desc: '1 if the claim has more than one service line, else 0.' },
]

const card2 = 'bg-white rounded-xl border border-gray-200 shadow-sm p-5'
const labelCls = 'block text-xs font-medium text-gray-600 mb-1'
const inputCls = 'w-full rounded-lg border border-gray-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#FE017D]/30 focus:border-[#FE017D]'

const fmtPct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const fmt4 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(4))

// ── Component ─────────────────────────────────────────────────────────────────

export default function Stage2ModelPanel() {
  const qc = useQueryClient()
  const [form, setForm] = useState<Stage2Config>(DEFAULT_CONFIG)
  const set = <K extends keyof Stage2Config>(k: K, v: Stage2Config[K]) => setForm((f) => ({ ...f, [k]: v }))
  const manualMode = form.decision_threshold_mode === 'manual'
  const [trials, setTrials] = useState<TrialRun[]>([])
  const [committed, setCommitted] = useState<number | null>(null)

  // Current committed model (404 → null before first retrain).
  const { data: current } = useQuery<Stage2Result | null>({
    queryKey: ['admin', 'model2'],
    queryFn: async () => {
      try {
        return (await api.get<Stage2Result>('/admin/model2')).data
      } catch {
        return null
      }
    },
  })

  const trialMutation = useMutation({
    mutationFn: async (body: Stage2Config) => (await api.post<Stage2Result>('/admin/model2/trial', body)).data,
    onSuccess: (data) => setTrials((prev) => [{ ...data, attempt: prev.length + 1 }, ...prev]),
  })

  const retrainMutation = useMutation({
    mutationFn: async (body: Stage2Config) =>
      (await api.post<{ claims_scored: number }>('/admin/model2/retrain', body)).data,
    onSuccess: (data) => {
      setCommitted(data.claims_scored)
      qc.invalidateQueries({ queryKey: ['admin', 'model2'] })
      setTimeout(() => setCommitted(null), 6000)
    },
  })

  const busy = trialMutation.isPending || retrainMutation.isPending
  const error = trialMutation.error || retrainMutation.error
  // Feature importances to show: latest trial, else the committed model.
  const importance = trials[0]?.feature_importance ?? current?.feature_importance ?? {}
  const metrics = trials[0] ?? current

  const numField = (
    key: keyof Stage2Config, label: string, opts: { min?: number; max?: number; nullable?: boolean } = {}
  ) => (
    <div>
      <label className={labelCls}>{label}</label>
      <input type="number" className={inputCls} min={opts.min} max={opts.max}
        value={form[key] == null ? '' : (form[key] as number)}
        placeholder={opts.nullable ? 'unlimited' : undefined}
        onChange={(e) => {
          const raw = e.target.value
          set(key, (raw === '' ? (opts.nullable ? null : 0) : Number(raw)) as never)
        }} />
      <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT[key as string]}</p>
    </div>
  )

  return (
    <div className="space-y-4">
      {/* ── Model header + features ─────────────────────────────── */}
      <div className={card2}>
        <div className="flex items-center gap-2 mb-1">
          <Layers className="w-4 h-4 text-[#FE017D]" />
          <h2 className="text-lg font-bold text-gray-900">Stage 2 — Claim Predictor</h2>
          <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#FE017D]/10 text-[#FE017D] font-semibold">calibrated</span>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Scores each <span className="font-semibold">claim</span> for overpayment risk from its own payment features plus
          the Stage 1 provider score. Its calibrated probability drives the audit plan below.
          {current?.trained_at && <span className="text-gray-400"> · last committed {new Date(current.trained_at).toLocaleString()}</span>}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FEATURES.map((f) => {
            const imp = importance[f.key]
            return (
              <div key={f.key} className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-gray-900">{f.name}</span>
                  {imp != null && (
                    <span className="text-[11px] font-mono text-[#FE017D]">{(imp * 100).toFixed(0)}%</span>
                  )}
                </div>
                <p className="text-[11px] text-gray-500 mt-1 leading-snug">{f.desc}</p>
                {imp != null && (
                  <div className="mt-1.5 w-full bg-gray-200 rounded-full h-1">
                    <div className="h-1 rounded-full bg-[#FE017D]" style={{ width: `${Math.min(imp * 100, 100)}%` }} />
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Metrics */}
        {metrics ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-4 pt-4 border-t border-gray-100">
            {([
              ['AUC-ROC', fmtPct(metrics.auc_roc)], ['F2', fmtPct(metrics.f2_score)],
              ['Precision', fmtPct(metrics.precision)], ['Recall', fmtPct(metrics.recall)],
              ['Brier (raw)', fmt4(metrics.brier_raw)], ['Brier (cal.)', fmt4(metrics.brier_calibrated)],
            ] as [string, string][]).map(([label, val]) => (
              <div key={label} className="text-center">
                <p className="text-[11px] text-gray-400 uppercase tracking-wider">{label}</p>
                <p className="text-lg font-bold text-gray-900">{val}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-400 mt-4 pt-4 border-t border-gray-100">
            No model committed yet — run a trial or retrain to see metrics.
          </p>
        )}
        {metrics && (
          <p className="text-[11px] text-gray-400 mt-2">
            {trials[0] ? 'Showing the latest trial. ' : 'Showing the committed model. '}
            Positive rate {fmtPct(metrics.positive_rate)} · calibration {metrics.calibration_method} · {metrics.training_rows.toLocaleString()} training rows.
            Lower Brier after calibration = probabilities read as true frequencies (needed for EV).
          </p>
        )}
      </div>

      {/* ── Hyperparameters ─────────────────────────────────────── */}
      <div className={card2}>
        <div className="flex items-center gap-2 mb-1">
          <FlaskConical className="w-4 h-4 text-[#FE017D]" />
          <h3 className="text-sm font-bold text-gray-900">Hyperparameters</h3>
          <span className="text-[11px] text-gray-400">· trial runs are not persisted; Retrain saves the model & re-scores every claim</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-3">
          {numField('n_estimators', 'n_estimators', { min: 10, max: 2000 })}
          {numField('max_depth', 'max_depth', { min: 1, max: 100, nullable: true })}
          {numField('min_samples_split', 'min_samples_split', { min: 2, max: 200 })}
          {numField('min_samples_leaf', 'min_samples_leaf', { min: 1, max: 200 })}
          {numField('max_leaf_nodes', 'max_leaf_nodes', { min: 2, max: 10000, nullable: true })}

          <div>
            <label className={labelCls}>max_features</label>
            <select className={inputCls} value={form.max_features ?? 'none'}
              onChange={(e) => set('max_features', e.target.value === 'none' ? null : e.target.value)}>
              <option value="sqrt">sqrt</option>
              <option value="log2">log2</option>
              <option value="none">all features</option>
            </select>
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT.max_features}</p>
          </div>
          <div>
            <label className={labelCls}>criterion</label>
            <select className={inputCls} value={form.criterion}
              onChange={(e) => set('criterion', e.target.value)}>
              <option value="gini">gini</option>
              <option value="entropy">entropy</option>
              <option value="log_loss">log_loss</option>
            </select>
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT.criterion}</p>
          </div>
          <div>
            <label className={labelCls}>class_weight</label>
            <select className={inputCls} value={form.class_weight ?? 'none'}
              onChange={(e) => set('class_weight', e.target.value === 'none' ? null : e.target.value)}>
              <option value="none">none</option>
              <option value="balanced">balanced</option>
              <option value="balanced_subsample">balanced_subsample</option>
            </select>
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT.class_weight}</p>
          </div>
          <div>
            <label className={labelCls}>bootstrap</label>
            <div className="flex items-center gap-2 h-[38px]">
              <button type="button" onClick={() => set('bootstrap', !form.bootstrap)}
                className={`relative w-11 h-6 rounded-full transition-colors ${form.bootstrap ? 'bg-[#FE017D]' : 'bg-gray-300'}`}>
                <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${form.bootstrap ? 'translate-x-5' : ''}`} />
              </button>
              <span className="text-sm text-gray-600">{form.bootstrap ? 'on' : 'off'}</span>
            </div>
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT.bootstrap}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4 pt-4 border-t border-gray-100">
          <div>
            <label className={labelCls}>decision_threshold_mode</label>
            <select className={inputCls} value={form.decision_threshold_mode}
              onChange={(e) => set('decision_threshold_mode', e.target.value)}>
              <option value="auto_f2">auto_f2 (sweep)</option>
              <option value="manual">manual</option>
            </select>
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT.decision_threshold_mode}</p>
          </div>
          <div>
            <label className={labelCls}>manual_threshold</label>
            <input type="number" step="0.01" min={0} max={1} className={inputCls}
              disabled={!manualMode}
              value={form.manual_threshold == null ? '' : form.manual_threshold}
              placeholder={manualMode ? '0.50' : 'auto'}
              onChange={(e) => set('manual_threshold', e.target.value === '' ? null : Number(e.target.value))} />
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">{HINT.manual_threshold}</p>
          </div>
          <div>
            <label className={labelCls}>calibration</label>
            <select className={inputCls} value={form.calibration_method}
              onChange={(e) => set('calibration_method', e.target.value)}>
              <option value="sigmoid">sigmoid (Platt)</option>
              <option value="isotonic">isotonic</option>
              <option value="none">none</option>
            </select>
            <p className="text-[11px] text-gray-400 mt-1 leading-snug">Maps raw scores to true probabilities. Keep on — EV multiplies probability × dollars.</p>
          </div>
        </div>

        {manualMode && form.manual_threshold == null && (
          <p className="text-xs text-amber-600 mt-3">manual_threshold is required when mode = manual.</p>
        )}

        <div className="flex flex-wrap items-center gap-3 mt-4 pt-4 border-t border-gray-100">
          <button onClick={() => trialMutation.mutate(form)} disabled={busy || (manualMode && form.manual_threshold == null)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-[#FE017D] text-[#FE017D] text-sm font-semibold rounded-lg hover:bg-[#FE017D]/5 disabled:opacity-40 transition-colors">
            <FlaskConical className={`w-4 h-4 ${trialMutation.isPending ? 'animate-pulse' : ''}`} />
            {trialMutation.isPending ? 'Running trial…' : 'Run Trial'}
          </button>
          <button onClick={() => retrainMutation.mutate(form)} disabled={busy || (manualMode && form.manual_threshold == null)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#FE017D] text-white text-sm font-semibold rounded-lg hover:bg-[#e5006f] disabled:opacity-40 transition-colors">
            <Save className="w-4 h-4" />
            {retrainMutation.isPending ? 'Retraining…' : 'Retrain & re-score claims'}
          </button>
        </div>

        {committed != null && (
          <div className="flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg px-3 py-2.5 mt-4">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <p className="text-xs text-green-700">Retrained and re-scored <span className="font-semibold">{committed.toLocaleString()}</span> claims. The audit plan below now reflects the new model.</p>
          </div>
        )}
        {error && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-3 py-2.5 mt-4">
            <AlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-red-700">
              {(error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? (error as Error)?.message ?? 'Request failed.'}
            </p>
          </div>
        )}

        {trials.length > 0 && (
          <div className="overflow-x-auto mt-4">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-[11px] text-gray-400 uppercase tracking-wider border-b border-gray-100">
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">n_est</th>
                  <th className="py-2 pr-3">cal.</th>
                  <th className="py-2 pr-3">AUC-ROC</th>
                  <th className="py-2 pr-3">F2</th>
                  <th className="py-2 pr-3">Precision</th>
                  <th className="py-2 pr-3">Recall</th>
                  <th className="py-2 pr-3">Brier raw→cal</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {trials.map((t) => (
                  <tr key={t.attempt}>
                    <td className="py-2 pr-3 font-mono text-gray-500">{t.attempt}</td>
                    <td className="py-2 pr-3 font-mono text-gray-500">{String(t.params?.n_estimators ?? '—')}</td>
                    <td className="py-2 pr-3 text-gray-500">{t.calibration_method}</td>
                    <td className="py-2 pr-3 font-semibold text-gray-900">{fmtPct(t.auc_roc)}</td>
                    <td className="py-2 pr-3">{fmtPct(t.f2_score)}</td>
                    <td className="py-2 pr-3">{fmtPct(t.precision)}</td>
                    <td className="py-2 pr-3">{fmtPct(t.recall)}</td>
                    <td className="py-2 pr-3 font-mono text-xs text-gray-500">{fmt4(t.brier_raw)} → {fmt4(t.brier_calibrated)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
