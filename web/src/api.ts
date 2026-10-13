import type { Report } from './types'

// Same-origin by default: `agenteval serve` mounts this bundle itself, and
// `npm run dev` proxies /api to :8000. Either way a relative path avoids CORS
// entirely -- an absolute URL breaks when the page is opened via a different
// host string (localhost vs 127.0.0.1) than the one in the allow-list.
const BASE = import.meta.env.VITE_API ?? ''

export async function fetchReport(
  params: { n_tasks: number; n_seeds: number; n_agents: number; seed: number },
  signal?: AbortSignal,
): Promise<Report> {
  const q = new URLSearchParams({
    n_tasks: String(params.n_tasks),
    n_seeds: String(params.n_seeds),
    n_agents: String(params.n_agents),
    seed: String(params.seed),
    n_boot: '800',
  })
  const res = await fetch(`${BASE}/api/report?${q}`, { signal })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`report failed (${res.status}): ${body.slice(0, 200)}`)
  }
  return res.json()
}

export const fmt = (x: number | null | undefined, d = 3): string =>
  x === null || x === undefined || Number.isNaN(x) ? 'n/a' : x.toFixed(d)

export const pct = (x: number | null | undefined, d = 1): string =>
  x === null || x === undefined || Number.isNaN(x) ? 'n/a' : `${(x * 100).toFixed(d)}%`
