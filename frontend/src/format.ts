import type { Stat } from './types'

export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined ? '—' : v.toFixed(digits)

export const pct = (v: number | null | undefined): string =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`

export const money = (v: number | null | undefined): string =>
  v === null || v === undefined ? '—' : `$${v.toFixed(4)}`

export const tokens = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return '—'
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return `${Math.round(v)}`
}

export const secs = (ms: number | null | undefined, digits = 1): string =>
  ms === null || ms === undefined ? '—' : `${(ms / 1000).toFixed(digits)}s`

/** Mean with its 95% interval, or an n= note when the interval is undefined. */
export const ci = (stat: Stat | undefined): string => {
  if (!stat || stat.mean === null) return '—'
  if (stat.ci95 === null) return `${stat.mean.toFixed(3)} (n=${stat.n})`
  return `${stat.mean.toFixed(3)} ± ${stat.ci95.toFixed(3)}`
}

/** Red→green scale used by the heatmap. */
export const scoreColor = (v: number | null): string => {
  if (v === null) return '#243140'
  const clamped = Math.max(0, Math.min(1, v))
  const hue = clamped * 120 // 0 = red, 120 = green
  return `hsl(${hue} 52% 58%)`
}

export const shortModel = (model: string): string =>
  model.includes('/') ? model.split('/').slice(1).join('/') : model

export const statusBadge = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'good'
    case 'running':
      return 'accent'
    case 'failed':
      return 'bad'
    case 'cancelled':
      return 'warn'
    default:
      return ''
  }
}
