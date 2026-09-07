import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { fmt, money, pct, secs, shortModel } from '../format'
import type { GenerationDetail, NumberItem } from '../types'

interface Props {
  generationId: number
  onClose: () => void
}

const CLASS_LABEL: Record<string, string> = {
  supported: 'matches a supplied KEY FACT',
  prompt: 'provided elsewhere in the prompt',
  derived: 'derivable from supplied facts',
  unsupported: 'not supplied and not derivable',
  contradicting: 'contradicts a supplied fact',
}

/**
 * Rebuild the response with every detected quantity wrapped in a <mark>.
 * Offsets come from the backend, which normalizes only 1:1 characters, so
 * they line up with the stored text.
 */
function highlight(text: string, items: NumberItem[]) {
  const sorted = [...items].sort((a, b) => a.start - b.start)
  const nodes: React.ReactNode[] = []
  let cursor = 0

  sorted.forEach((item, i) => {
    if (item.start < cursor || item.end > text.length) return // overlapping or stale span
    if (item.start > cursor) nodes.push(text.slice(cursor, item.start))
    nodes.push(
      <mark
        key={i}
        className={`q q-${item.classification}`}
        title={`${item.classification}: ${item.evidence ?? CLASS_LABEL[item.classification] ?? ''}`}
      >
        {text.slice(item.start, item.end)}
      </mark>,
    )
    cursor = item.end
  })
  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

export default function GenerationDrawer({ generationId, onClose }: Props) {
  const [detail, setDetail] = useState<GenerationDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'output' | 'judge' | 'numbers' | 'prompt' | 'reasoning'>('output')

  useEffect(() => {
    let active = true
    setDetail(null)
    setError(null)
    api
      .generation(generationId)
      .then((d) => active && setDetail(d))
      .catch((e) => active && setError(String((e as Error).message ?? e)))
    return () => {
      active = false
    }
  }, [generationId])

  // Escape closes the drawer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const items = detail?.details?.numbers?.items ?? []
  const highlighted = useMemo(
    () => (detail ? highlight(detail.output, items) : null),
    [detail, items],
  )

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="spread">
          <h2 style={{ margin: 0 }}>
            {detail ? `${shortModel(detail.model)} — ${detail.prompt_id}` : 'Loading…'}
          </h2>
          <button onClick={onClose}>Close</button>
        </div>

        {error && <div className="banner error">{error}</div>}
        {!detail && !error && <p className="muted">Loading response…</p>}

        {detail && (
          <>
            <div className="row tight" style={{ marginTop: 10 }}>
              <span className="badge accent">{detail.technique}</span>
              <span className="badge">repeat {detail.repeat_index}</span>
              <span className="badge">{detail.word_count} words</span>
              {detail.truncated ? <span className="badge bad">truncated</span> : null}
              {detail.self_graded ? <span className="badge warn">self-graded</span> : null}
              <span className="badge">{secs(detail.latency_ms)}</span>
              <span className="badge">{money(detail.cost_usd)}</span>
            </div>

            <div className="stat-tiles" style={{ marginTop: 14 }}>
              <Tile label="Grounding" value={pct(detail.numeric_grounding)} />
              <Tile label="Fact recall" value={pct(detail.fact_recall)} />
              <Tile label="Structure" value={pct(detail.structural_compliance)} />
              <Tile label="Length" value={fmt(detail.length_compliance, 2)} />
              <Tile label="Markers" value={pct(detail.required_tokens)} />
              <Tile label="Judge" value={fmt(detail.judge_overall, 2)} />
            </div>

            <div className="row tight" style={{ marginTop: 18, marginBottom: 10 }}>
              {(['output', 'numbers', 'judge', 'prompt', 'reasoning'] as const).map((t) => (
                <button
                  key={t}
                  className={tab === t ? 'primary' : ''}
                  onClick={() => setTab(t)}
                  disabled={t === 'reasoning' && !detail.reasoning}
                >
                  {t === 'output'
                    ? 'Response'
                    : t === 'numbers'
                      ? `Numbers (${items.length})`
                      : t === 'judge'
                        ? 'Judge'
                        : t === 'prompt'
                          ? 'Prompt'
                          : 'Reasoning'}
                </button>
              ))}
            </div>

            {tab === 'output' && (
              <>
                <div className="legend" style={{ marginBottom: 8 }}>
                  {Object.entries(CLASS_LABEL).map(([k, label]) => (
                    <span key={k}>
                      <span className={`swatch q-${k}`} style={{ background: swatch(k) }} />
                      {label}
                    </span>
                  ))}
                </div>
                <div className="output">{highlighted}</div>
              </>
            )}

            {tab === 'numbers' && (
              <>
                <p className="small muted" style={{ marginTop: 0 }}>
                  Every quantity found in the response, with the verdict and the
                  evidence behind it.
                </p>
                <table>
                  <thead>
                    <tr>
                      <th>Quantity</th>
                      <th>Verdict</th>
                      <th>Evidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item, i) => (
                      <tr key={i}>
                        <td className="mono">{item.raw}</td>
                        <td>
                          <span className={`badge ${badgeClass(item.classification)}`}>
                            {item.classification}
                          </span>
                        </td>
                        <td className="small">
                          {item.evidence ?? CLASS_LABEL[item.classification]}
                        </td>
                      </tr>
                    ))}
                    {items.length === 0 && (
                      <tr>
                        <td colSpan={3} className="muted">
                          No quantities detected in this response.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>

                <h3>KEY FACTS coverage</h3>
                <div className="row tight">
                  {(detail.details.facts?.found ?? []).map((f, i) => (
                    <span key={`f${i}`} className="badge good">
                      {f}
                    </span>
                  ))}
                  {(detail.details.facts?.missing ?? []).map((f, i) => (
                    <span key={`m${i}`} className="badge bad" title="Not cited in the response">
                      {f}
                    </span>
                  ))}
                </div>

                {(detail.details.structure?.missing?.length ?? 0) > 0 && (
                  <>
                    <h3>Missing sections</h3>
                    <ul className="small">
                      {detail.details.structure!.missing.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </>
                )}

                {(detail.details.format?.problems?.length ?? 0) > 0 && (
                  <>
                    <h3>Format problems</h3>
                    <ul className="small">
                      {detail.details.format!.problems.map((p) => (
                        <li key={p}>{p}</li>
                      ))}
                    </ul>
                  </>
                )}
              </>
            )}

            {tab === 'judge' && (
              <>
                {detail.judge_error && (
                  <div className="banner error">Judge failed: {detail.judge_error}</div>
                )}
                {detail.self_graded ? (
                  <div className="banner">
                    This response was graded by the same model that produced it, so the
                    score may be inflated by self-preference bias.
                  </div>
                ) : null}
                <table>
                  <thead>
                    <tr>
                      <th>Dimension</th>
                      <th className="num">Score</th>
                      <th>Justification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(detail.judge_scores).map(([dim, score]) => (
                      <tr key={dim}>
                        <td>{dim.replace(/_/g, ' ')}</td>
                        <td className="num">{score} / 5</td>
                        <td className="small">{detail.judge_justifications[dim]}</td>
                      </tr>
                    ))}
                    {Object.keys(detail.judge_scores).length === 0 && (
                      <tr>
                        <td colSpan={3} className="muted">
                          No judge scores for this response.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </>
            )}

            {tab === 'prompt' && (
              <>
                {detail.prompt_meta?.key_facts_raw && (
                  <>
                    <h3>KEY FACTS the response was held to</h3>
                    <div className="output" style={{ maxHeight: 130 }}>
                      {detail.prompt_meta.key_facts_raw}
                    </div>
                  </>
                )}
                <h3>Full prompt</h3>
                <div className="output">{detail.prompt_text}</div>
              </>
            )}

            {tab === 'reasoning' && (
              <>
                <p className="small muted" style={{ marginTop: 0 }}>
                  Internal reasoning returned separately by the model. It is billed as
                  output tokens but excluded from the graded response.
                </p>
                <div className="output">{detail.reasoning}</div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function swatch(kind: string): string {
  switch (kind) {
    case 'supported':
      return '#4f9d69'
    case 'prompt':
      return '#6d9d7f'
    case 'derived':
      return '#6fa8cd'
    case 'unsupported':
      return '#d9a03f'
    case 'contradicting':
      return '#d4645c'
    default:
      return '#2a3947'
  }
}

function badgeClass(kind: string): string {
  if (kind === 'contradicting') return 'bad'
  if (kind === 'unsupported') return 'warn'
  if (kind === 'derived') return 'accent'
  return 'good'
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  )
}
