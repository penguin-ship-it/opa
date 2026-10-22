import { useState } from 'react'
import { ShieldX } from 'lucide-react'
import { useCurrentUser } from '../hooks/useCurrentUser'
import AdminSidebar, { type AdminSection } from '../components/admin/AdminSidebar'
import ReferenceFreshnessPanel from '../components/admin/system/ReferenceFreshnessPanel'
import UsersPanel from '../components/admin/system/UsersPanel'
import DetectorRulesPanel from '../components/admin/rules/DetectorRulesPanel'
import PrioritizationPanel from '../components/admin/rules/PrioritizationPanel'
import MLModelPanel from '../components/admin/rules/MLModelPanel'
import CptCodesPanel from '../components/admin/reference/CptCodesPanel'
import IcdCodesPanel from '../components/admin/reference/IcdCodesPanel'
import DrgCodesPanel from '../components/admin/reference/DrgCodesPanel'
import ModifierCodesPanel from '../components/admin/reference/ModifierCodesPanel'
import ExcludedProvidersPanel from '../components/admin/reference/ExcludedProvidersPanel'
import LetterTemplatesTab from '../components/admin/LetterTemplatesTab'
import RulePromptsPanel from '../components/admin/RulePromptsPanel'

const TITLES: Record<AdminSection, string> = {
  freshness:          'Reference Data',
  users:              'Users',
  rules:              'Detector Rules',
  'rule-prompts':     'Rule Prompts',
  prioritization:     'Prioritization',
  model:              'ML Model',
  cpt:                'CPT / HCPCS Codes',
  icd:                'ICD-10 Codes',
  drg:                'DRG Codes',
  modifiers:          'Modifier Codes',
  excluded:           'Excluded Providers',
  'letter-templates': 'Letter Templates',
  'doc-templates':    'Document Templates',
}

function DocTemplatesPlaceholder() {
  return (
    <div className="flex-1 flex items-center justify-center text-sm text-gray-400">
      Document templates — coming soon
    </div>
  )
}

export default function AdminPage() {
  const [section, setSection] = useState<AdminSection>('freshness')
  const { currentUser, isLoading } = useCurrentUser()

  // Every /api/admin/* route is admin-only server-side; without this gate a
  // non-admin sees each panel's 403 rendered as a misleading empty state.
  if (isLoading) {
    return <div className="flex h-full items-center justify-center text-slate-400">Checking access…</div>
  }
  if (currentUser?.role !== 'admin') {
    return <AdminAccessRequired role={currentUser?.role} />
  }

  const content: Record<AdminSection, React.ReactNode> = {
    freshness:          <ReferenceFreshnessPanel />,
    users:              <UsersPanel />,
    rules:              <DetectorRulesPanel />,
    'rule-prompts':     <RulePromptsPanel />,
    prioritization:     <PrioritizationPanel />,
    model:              <MLModelPanel />,
    cpt:                <CptCodesPanel />,
    icd:                <IcdCodesPanel />,
    drg:                <DrgCodesPanel />,
    modifiers:          <ModifierCodesPanel />,
    excluded:           <ExcludedProvidersPanel />,
    'letter-templates': <LetterTemplatesTab />,
    'doc-templates':    <DocTemplatesPlaceholder />,
  }

  // Code table panels manage their own full-height layout
  const fullHeight = ['cpt', 'icd', 'drg', 'modifiers', 'excluded'].includes(section)

  return (
    <div className="flex h-full min-h-0 -m-5">
      <AdminSidebar active={section} onChange={setSection} />

      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Section header */}
        {!fullHeight && (
          <div className="px-6 py-4 border-b border-gray-200 bg-white flex-shrink-0">
            <h1 className="text-lg font-bold text-gray-900">{TITLES[section]}</h1>
          </div>
        )}

        {/* Section content */}
        <div className={fullHeight ? 'flex-1 min-h-0 flex flex-col' : 'flex-1 overflow-y-auto p-6'}>
          {content[section]}
        </div>
      </div>
    </div>
  )
}

function AdminAccessRequired({ role }: { role?: string }) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-amber-50 text-amber-600 mb-4">
          <ShieldX className="w-7 h-7" />
        </div>
        <h1 className="text-2xl font-semibold text-slate-900 mb-2">Admin access required</h1>
        <p className="text-sm text-slate-600">
          The Admin area is limited to users with the <span className="font-medium">admin</span> role.
          {role && <> You're signed in as <span className="font-medium">{role}</span>.</>}
        </p>
        <p className="text-xs text-slate-400 mt-6">
          Ask an administrator to grant you the admin role in the IAM admin app.
        </p>
      </div>
    </div>
  )
}
