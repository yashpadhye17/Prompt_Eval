import type {
  ConfigResponse,
  Estimate,
  GenerationDetail,
  RunListItem,
  RunSummary,
} from './types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'content-type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      const raw = body.detail ?? detail
      detail = typeof raw === 'object' && raw !== null ? raw.message ?? JSON.stringify(raw) : raw
    } catch {
      // response had no JSON body; the status text will do
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export interface RunPayload {
  models?: string[]
  prompt_ids?: string[]
  repeats?: number
  judge_enabled?: boolean
  judge_model?: string
  judge_repeats_per_cell?: number
}

export const api = {
  config: () => request<ConfigResponse>('/config'),
  runs: () => request<RunListItem[]>('/runs'),
  run: (id: string) => request<RunSummary>(`/runs/${encodeURIComponent(id)}`),
  estimate: (payload: RunPayload) =>
    request<Estimate>('/runs/estimate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  createRun: (payload: RunPayload) =>
    request<{ run_id: string; total_tasks: number }>('/runs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  cancelRun: (id: string) =>
    request<{ cancelled: boolean }>(`/runs/${encodeURIComponent(id)}/cancel`, {
      method: 'POST',
    }),
  deleteRun: (id: string) =>
    request<{ deleted: string }>(`/runs/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  generation: (id: number) => request<GenerationDetail>(`/generations/${id}`),
  buildReport: (id: string) =>
    request<{ path: string; filename: string }>(
      `/runs/${encodeURIComponent(id)}/report`,
      { method: 'POST' },
    ),
  reportUrl: (id: string) => `${BASE}/runs/${encodeURIComponent(id)}/report`,
  eventsUrl: (id: string) => `${BASE}/runs/${encodeURIComponent(id)}/events`,
}
