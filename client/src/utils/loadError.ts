/** Human reason a model-info request failed, instead of a silent empty state. */
export function describeLoadError(err: unknown, what: string): string {
  const status = (err as { response?: { status?: number } })?.response?.status
  if (status === 403) return `${what} requires the admin role.`
  if (status === 404) return `${what}: no trained model yet — run Retrain (or \`make seed\`).`
  if (status) return `Couldn't load ${what.toLowerCase()} (HTTP ${status}).`
  return `Couldn't load ${what.toLowerCase()} — is the backend running?`
}
