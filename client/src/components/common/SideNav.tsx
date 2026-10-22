import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useRef, useState, useEffect, useCallback } from 'react'
import {
  LayoutDashboard,
  ListChecks,
  Settings,
  Archive,
  ScanLine,
  Users,
  Table,
  ShieldCheck,
  UserCog,
  AlertTriangle,
  ShieldAlert,
  Inbox,
  FileInput,
  FileOutput,
  Send,
  Flag,
  Search,
  X,
  MonitorPlay,
  ExternalLink,
  RotateCcw,
  LayoutGrid,
} from 'lucide-react'
import api from '../../services/api'
import { APP_URLS } from '../../config/appUrls'
import { getStatusCounts } from '../../services/caseService'
import { useCurrentUser } from '../../hooks/useCurrentUser'
import type { CaseStatus, CaseSummary, ReferenceDataFreshness, UserRole } from '../../types'

// Typography + palette mirror ClaimGuard's left nav so the platform reads
// consistently. Accent stays PayGuard pink for the active icon + row tint.
const BRAND    = '#FE017D'   // PayGuard accent — active icon
const BRAND_BG = '#fdf2f8'   // pink-50 — active row background

type LinkSpec = {
  to: string
  label: string
  icon: any
  // When set, this is a lifecycle stage tab on the Worklist (/worklist?stage=…).
  stage?: string
  // Statuses whose live counts sum into this item's badge.
  badgeStatuses?: CaseStatus[]
  // Active when pathname starts with `to` (covers nested routes).
  matchPrefix?: boolean
  supervisorOnly?: boolean
  adminOnly?: boolean
}

type NavSection = { heading?: string; links: LinkSpec[]; kind?: 'demo-tools' }

// The nav is organized around the case LIFECYCLE rather than a flat list.
// "Cases" maps each stage to the statuses it rolls up (the badge counts);
// everything that isn't a case queue is pushed into its own section.
const SECTIONS: NavSection[] = [
  { links: [
    { to: '/overview', label: 'Overview', icon: LayoutGrid, matchPrefix: true },
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/control-room', label: 'Control Room', icon: MonitorPlay, matchPrefix: true },
  ] },
  {
    heading: 'Cases',
    links: [
      { to: '/worklist', stage: 'intake',    label: 'Intake',    icon: Inbox,      badgeStatuses: ['new', 'awaiting_837'] },
      { to: '/worklist', stage: 'review',    label: 'Review',    icon: ListChecks, badgeStatuses: ['assigned', 'in_review', 'ready_for_notice'] },
      { to: '/worklist', stage: 'approvals', label: 'Approvals', icon: ShieldCheck, badgeStatuses: ['pending_supervisor'], supervisorOnly: true },
      { to: '/worklist', stage: 'recovery',  label: 'Recovery',  icon: Send,       badgeStatuses: ['notice_sent', 'provider_responded', 'reconciling'] },
      { to: '/closed-cases', label: 'Closed', icon: Archive, matchPrefix: true },
    ],
  },
  {
    heading: 'Smart Views',
    links: [
      { to: '/worklist', stage: 'jeopardy', label: 'Jeopardy / Overdue', icon: AlertTriangle },
      { to: '/escalations', label: 'Escalations', icon: Flag, matchPrefix: true, supervisorOnly: true },
    ],
  },
  {
    heading: 'Insights',
    links: [
      { to: '/provider-risk', label: 'Provider Risk', icon: ShieldAlert, matchPrefix: true, supervisorOnly: true },
      { to: '/assignments',   label: 'Assignments',   icon: UserCog,     matchPrefix: true, supervisorOnly: true },
    ],
  },
  {
    heading: 'Tools',
    links: [
      { to: '/analyze-835',      label: 'Analyze 835',      icon: ScanLine,   matchPrefix: true },
      { to: '/delivery-queue',   label: 'Delivery Queue',   icon: Send,       matchPrefix: true },
      { to: '/file-intake',      label: 'File Intake',      icon: FileInput,  matchPrefix: true, adminOnly: true },
      { to: '/output-files',     label: 'Output Files',     icon: FileOutput, matchPrefix: true, adminOnly: true },
    ],
  },
  {
    heading: 'Reference',
    links: [
      { to: '/members',   label: 'Members',   icon: Users, matchPrefix: true },
      { to: '/providers', label: 'Providers', icon: Users, matchPrefix: true },
    ],
  },
  // Demo tools — custom-rendered (external portal link + reset action), so this
  // section carries no LinkSpec entries; see the `demo-tools` branch in render.
  { heading: 'Demo', kind: 'demo-tools', links: [] },
  { links: [{ to: '/admin', label: 'Admin', icon: Settings, matchPrefix: true, adminOnly: true }] },
]

const STATUS_DOT: Record<string, string> = {
  new:                   'bg-blue-400',
  awaiting_837:          'bg-slate-300',
  assigned:              'bg-yellow-400',
  in_review:             'bg-orange-400',
  ready_for_notice:      'bg-purple-400',
  pending_supervisor:    'bg-pink-400',
  notice_sent:           'bg-indigo-400',
  provider_responded:    'bg-teal-400',
  reconciling:           'bg-cyan-400',
  closed_recovered:      'bg-green-400',
  closed_no_action:      'bg-slate-300',
  closed_disputed:       'bg-red-400',
}

function CaseSearch() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [debouncedQ, setDebouncedQ] = useState('')

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => setDebouncedQ(query.trim()), 220)
  }, [query])

  const { data, isFetching } = useQuery<{ items: CaseSummary[] }>({
    queryKey: ['case-search', debouncedQ],
    queryFn: async () => {
      if (!debouncedQ) return { items: [] }
      // The list endpoint returns { items: [...] }; page_size caps it (limit is ignored).
      const res = await api.get('/cases', { params: { search: debouncedQ, page_size: 8 } })
      return res.data
    },
    enabled: debouncedQ.length >= 2,
    staleTime: 15_000,
  })

  const results = data?.items ?? []

  const pick = useCallback((c: CaseSummary) => {
    navigate(`/cases/${c.id}`)
    setQuery('')
    setDebouncedQ('')
    setOpen(false)
  }, [navigate])

  const clear = () => { setQuery(''); setDebouncedQ(''); setOpen(false) }

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open || !results.length) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(a => Math.min(a + 1, results.length - 1)) }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setActive(a => Math.max(a - 1, 0)) }
    if (e.key === 'Enter')     { e.preventDefault(); pick(results[active]) }
    if (e.key === 'Escape')    { setOpen(false); inputRef.current?.blur() }
  }

  return (
    <div ref={containerRef} className="relative px-3 mb-2">
      <div className="relative flex items-center">
        <Search className="absolute left-2.5 w-3.5 h-3.5 text-slate-400 pointer-events-none" />
        <input
          ref={inputRef}
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true); setActive(0) }}
          onFocus={() => query.length >= 2 && setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder="Search cases…"
          className="w-full pl-7 pr-6 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg
                     placeholder-slate-400 text-slate-700
                     focus:outline-none focus:border-[#FE017D] focus:ring-1 focus:ring-pink-100 focus:bg-white
                     transition-colors"
        />
        {query && (
          <button onClick={clear} className="absolute right-2 text-slate-400 hover:text-slate-600">
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      {open && debouncedQ.length >= 2 && (
        <div className="absolute left-3 right-3 top-full mt-1 z-50 bg-white rounded-xl border border-slate-200
                        shadow-lg shadow-slate-200/80 overflow-hidden">
          {isFetching && !results.length ? (
            <p className="text-xs text-slate-400 px-3 py-2.5">Searching…</p>
          ) : results.length === 0 ? (
            <p className="text-xs text-slate-400 px-3 py-2.5">No cases found</p>
          ) : (
            <ul>
              {results.map((c, i) => {
                const memberName = c.claim?.member?.name ?? '—'
                const dot = STATUS_DOT[c.status] ?? 'bg-slate-300'
                return (
                  <li key={c.id}>
                    <button
                      onMouseDown={() => pick(c)}
                      onMouseEnter={() => setActive(i)}
                      className={`w-full text-left px-3 py-2 flex items-start gap-2.5 transition-colors ${
                        i === active ? 'bg-pink-50' : 'hover:bg-slate-50'
                      }`}
                    >
                      <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}`} />
                      <div className="min-w-0">
                        <p className="text-xs font-medium text-slate-800 truncate">{memberName}</p>
                        <p className="text-[10px] text-slate-400 truncate">{c.case_number} · {c.status.replace(/_/g, ' ')}</p>
                      </div>
                      <span className="ml-auto text-[10px] font-semibold text-slate-400 flex-shrink-0 mt-0.5">
                        ${(c.amount_at_risk ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

// Demo tools row: open the mock provider portal (the recoup-notice upload
// target) in a new tab, and reset the demo-created data. Reset hits
// POST /api/demo/reset, which removes only demo-stamped rows (seeded data is
// untouched) — the same clear a demo re-run performs.
function DemoTools() {
  const [resetting, setResetting] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  const rowClass =
    'w-full flex items-center gap-3 px-3 py-2.5 rounded-2xl text-sm ' +
    'text-slate-600 hover:bg-slate-100 hover:text-slate-900 transition-colors'

  const handleReset = async () => {
    if (resetting) return
    if (!window.confirm('Reset demo data? This clears demo-created cases and claims. Seeded reference data is kept.')) {
      return
    }
    setResetting(true)
    setNote(null)
    try {
      await api.post('/demo/reset')
      setNote('✓ Demo data reset')
      // Reload so every open view refetches against the cleared data.
      setTimeout(() => window.location.reload(), 600)
    } catch {
      setResetting(false)
      setNote('Reset failed — see console')
    }
  }

  return (
    <div className="space-y-0.5">
      <a
        href={APP_URLS.providerPortal}
        target="_blank"
        rel="noopener noreferrer"
        data-tour="nav-provider-portal"
        className={rowClass}
      >
        <ExternalLink className="w-[18px] h-[18px] flex-shrink-0" style={{ color: '#94a3b8' }} />
        <span className="flex-1">Provider Portal</span>
      </a>
      <button
        type="button"
        onClick={handleReset}
        disabled={resetting}
        data-tour="nav-reset-data"
        className={`${rowClass} disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        <RotateCcw className="w-[18px] h-[18px] flex-shrink-0" style={{ color: '#94a3b8' }} />
        <span className="flex-1 text-left">{resetting ? 'Resetting…' : 'Reset Data'}</span>
      </button>
      {note && <p className="px-3 pt-1 text-[10px] text-slate-400">{note}</p>}
    </div>
  )
}

export default function SideNav() {
  // Role of the authenticated user, used only to filter which nav links show —
  // a UX nicety, not security; the server enforces real access. Read from the
  // CurrentUser context, not localStorage `opa_role`: that key is written only
  // after the async auth lookup, so a previous user's role could leak into the
  // first render. Until the user loads, treat them as an analyst (fail closed).
  const { currentUser } = useCurrentUser()
  const role = (currentUser?.role ?? 'analyst') as UserRole
  const { pathname, search } = useLocation()
  const currentStage = new URLSearchParams(search).get('stage') ?? ''

  const { data: freshness = [] } = useQuery<ReferenceDataFreshness[]>({
    queryKey: ['freshness-banner'],
    queryFn: async () => {
      const res = await api.get<ReferenceDataFreshness[]>('/admin/reference-freshness')
      return res.data
    },
    staleTime: 5 * 60 * 1000,
    enabled: role === 'admin',
  })

  // Live per-status counts drive the stage badges — the at-a-glance "where is
  // the work piling up" signal. Org-wide so the nav reads as a pipeline overview.
  const { data: counts = {} } = useQuery<Record<string, number>>({
    queryKey: ['status-counts'],
    queryFn: () => getStatusCounts(),
    staleTime: 30 * 1000,
  })

  const alertCount = freshness.filter((f) => f.status === 'critical' || f.status === 'stale').length

  const visible = (l: LinkSpec) =>
    (!l.supervisorOnly || role === 'supervisor' || role === 'admin') &&
    (!l.adminOnly || role === 'admin')

  const isActive = (l: LinkSpec): boolean => {
    if (l.stage !== undefined) return pathname === '/worklist' && currentStage === l.stage
    if (l.to === '/') return pathname === '/'
    if (l.matchPrefix) return pathname === l.to || pathname.startsWith(l.to + '/')
    return pathname === l.to
  }

  const badgeFor = (l: LinkSpec): number =>
    (l.badgeStatuses ?? []).reduce((sum, s) => sum + (counts[s] ?? 0), 0)

  const href = (l: LinkSpec) => (l.stage ? `/worklist?stage=${l.stage}` : l.to)

  return (
    <aside className="w-full h-full bg-white flex flex-col overflow-y-auto">
      {/* Header */}
      <div className="px-5 pt-5 pb-4 flex items-center justify-between">
        <p className="text-slate-900 text-lg font-semibold tracking-tight leading-none">PayGuard</p>
        <div className="relative">
          <button className="w-8 h-8 flex items-center justify-center">
            <img src="/favicon.svg" alt="PayGuard" className="w-7 h-7" />
          </button>
          {alertCount > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] bg-red-500 rounded-full
                             text-white text-[10px] font-bold flex items-center justify-center px-1">
              {alertCount}
            </span>
          )}
        </div>
      </div>

      <div className="mx-5 border-t border-slate-100" />

      {/* Navigation */}
      <nav className="flex-1 px-3 pt-3 overflow-y-auto" data-tour="main-nav">
        {SECTIONS.map((section, si) => {
          if (section.kind === 'demo-tools') {
            return (
              <div key="demo-tools" className="mt-4 mb-1">
                {section.heading && (
                  <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider px-3 mb-1.5">
                    {section.heading}
                  </p>
                )}
                <DemoTools />
              </div>
            )
          }
          const links = section.links.filter(visible)
          if (!links.length) return null
          return (
            <div key={section.heading ?? `sec-${si}`} className={si === 0 ? 'mb-1' : 'mt-4 mb-1'}>
              {section.heading && (
                <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider px-3 mb-1.5">
                  {section.heading}
                </p>
              )}
              {section.heading === 'Cases' && <CaseSearch />}
              <div className="space-y-0.5">
                {links.map((l) => {
                  const active = isActive(l)
                  const badge = badgeFor(l)
                  const Icon = l.icon
                  return (
                    <Link
                      key={l.label}
                      to={href(l)}
                      data-tour={`nav-${l.label.toLowerCase().replace(/\s+/g, '-')}`}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-2xl text-sm transition-colors ${
                        active
                          ? 'text-slate-900 font-medium'
                          : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                      }`}
                      style={active ? { backgroundColor: BRAND_BG } : undefined}
                    >
                      <Icon
                        className="w-[18px] h-[18px] flex-shrink-0"
                        style={{ color: active ? BRAND : '#94a3b8' }}
                      />
                      <span className="flex-1">{l.label}</span>
                      {badge > 0 && (
                        <span
                          className={`min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-semibold
                                      flex items-center justify-center ${
                            active ? 'bg-[#FE017D] text-white' : 'bg-slate-100 text-slate-500'
                          }`}
                        >
                          {badge}
                        </span>
                      )}
                    </Link>
                  )
                })}
              </div>
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-100">
        <p className="text-[11px] text-slate-400">PayGuard v0.1.0</p>
      </div>
    </aside>
  )
}
