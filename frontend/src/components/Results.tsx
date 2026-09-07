import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ci, fmt, money, pct, scoreColor, secs, shortModel } from '../format'
import type { RunSummary } from '../types'

const AXIS = { stroke: '#8ba0b3', fontSize: 11 }
const TOOLTIP_STYLE = {
  background: '#16202b',
  border: '1px solid #2a3947',
  borderRadius: 6,
  fontSize: 12,
}

interface Props {
  summary: RunSummary
  onOpenGeneration: (id: number) => void
}

export default function Results({ summary, onOpenGeneration }: Props) {
  const { models, techniques, heatmap, overall, operational, totals, metric_labels } = summary

  const metricKeys = Object.keys(metric_labels)
  const modelBars = models.map((m) => ({
    name: shortModel(m.model),
    score: m.metrics.composite?.mean ?? 0,
    error: m.metrics.composite?.ci95 ?? 0,
  }))
  const techniqueBars = techniques.map((t) => ({
    name: t.technique,
    score: t.metrics.composite?.mean ?? 0,
    error: t.metrics.composite?.ci95 ?? 0,
  }))

  const radarData = metricKeys
    .filter((k) => k !== 'judge_overall')
    .map((key) => {
      const row: Record<string, string | number> = { metric: metric_labels[key] }
      models.forEach((m) => {
        row[shortModel(m.model)] = m.metrics[key]?.mean ?? 0
      })
      return row
    })

  const palette = ['#4a9fd8', '#4f9d69', '#d9a03f', '#d4645c', '#9d7fc0', '#5fb8b0']

  return (
    <>
      <h2>Summary</h2>
      <div className="stat-tiles">
        <Tile
          label="Composite"
          value={fmt(overall.composite?.mean)}
          sub={ci(overall.composite)}
        />
        <Tile
          label="Numeric grounding"
          value={pct(overall.numeric_grounding?.mean)}
          sub={`${pct(1 - (overall.numeric_grounding?.mean ?? 0))} unsupported`}
        />
        <Tile label="Fact recall" value={pct(overall.fact_recall?.mean)} />
        <Tile
          label="Structure"
          value={pct(overall.structural_compliance?.mean)}
          sub="required sections present"
        />
        <Tile
          label="Length compliance"
          value={fmt(overall.length_compliance?.mean)}
          sub={`${fmt(operational.words?.mean, 0)} words avg`}
        />
        <Tile
          label="Judge"
          value={fmt(overall.judge_overall?.mean)}
          sub={
            totals.self_graded > 0
              ? `${totals.self_graded} self-graded`
              : `${totals.judged} judged`
          }
        />
        <Tile
          label="Cost"
          value={money((operational.cost_usd ?? 0) + (operational.judge_cost_usd ?? 0))}
          sub={`${money(operational.cost_usd)} gen + ${money(operational.judge_cost_usd)} judge`}
        />
        <Tile
          label="Errors"
          value={String(operational.errors + operational.truncated)}
          sub={`${operational.errors} failed, ${operational.truncated} truncated`}
        />
      </div>

      {totals.self_graded > 0 && (
        <div className="banner" style={{ marginTop: 16 }}>
          {totals.self_graded} of {totals.judged} judged responses were graded by the
          model that produced them. Models tend to favour their own output, so those
          rows are marked and should not be read as neutral.
        </div>
      )}

      <div className="grid2">
        <div className="card">
          <h3>Model ranking (composite, 95% CI)</h3>
          <ResponsiveContainer width="100%" height={40 + modelBars.length * 46}>
            <BarChart data={modelBars} layout="vertical" margin={{ left: 6, right: 26 }}>
              <CartesianGrid stroke="#243140" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} {...AXIS} />
              <YAxis type="category" dataKey="name" width={104} {...AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => fmt(v)} />
              <Bar dataKey="score" fill="#4a9fd8" radius={[0, 4, 4, 0]}>
                <ErrorBar dataKey="error" stroke="#e6edf3" width={5} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Technique ranking (composite, 95% CI)</h3>
          <ResponsiveContainer width="100%" height={40 + techniqueBars.length * 46}>
            <BarChart data={techniqueBars} layout="vertical" margin={{ left: 6, right: 26 }}>
              <CartesianGrid stroke="#243140" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} {...AXIS} />
              <YAxis type="category" dataKey="name" width={104} {...AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => fmt(v)} />
              <Bar dataKey="score" fill="#4f9d69" radius={[0, 4, 4, 0]}>
                <ErrorBar dataKey="error" stroke="#e6edf3" width={5} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <h2>Model detail</h2>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Composite</th>
              {metricKeys.map((k) => (
                <th className="num" key={k}>
                  {metric_labels[k].replace('Compliance', '').replace('Overall', '')}
                </th>
              ))}
              <th className="num">Cost</th>
              <th className="num">Latency</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model}>
                <td>
                  {shortModel(m.model)}{' '}
                  {m.self_graded && (
                    <span className="badge warn" title="Graded by itself">
                      self
                    </span>
                  )}
                </td>
                <td className="num">{ci(m.metrics.composite)}</td>
                {metricKeys.map((k) => (
                  <td className="num" key={k}>
                    {fmt(m.metrics[k]?.mean, 2)}
                  </td>
                ))}
                <td className="num">{money(m.operational.cost_usd)}</td>
                <td className="num">{secs(m.operational.latency_ms?.mean)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {heatmap.grid.length > 0 && (
        <>
          <h2>Technique against model</h2>
          <div className="card">
            <p className="small muted" style={{ marginTop: 0 }}>
              Mean composite per pairing. Shows whether a technique's advantage holds
              across models or is specific to one.
            </p>
            <table className="heatmap">
              <thead>
                <tr>
                  <th />
                  {heatmap.models.map((m) => (
                    <th key={m} className="num" style={{ fontSize: 11 }}>
                      {shortModel(m)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {heatmap.techniques.map((t, i) => (
                  <tr key={t}>
                    <th style={{ whiteSpace: 'nowrap' }}>{t}</th>
                    {heatmap.grid[i].map((v, j) => (
                      <td
                        key={j}
                        className="cell"
                        style={{ background: scoreColor(v) }}
                        title={`${t} × ${heatmap.models[j]}`}
                      >
                        {v === null ? 'n/a' : v.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {models.length > 0 && radarData.length > 0 && (
        <>
          <h2>Metric profile</h2>
          <div className="card">
            <ResponsiveContainer width="100%" height={330}>
              <RadarChart data={radarData} outerRadius="72%">
                <PolarGrid stroke="#2a3947" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#8ba0b3', fontSize: 10.5 }} />
                <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#8ba0b3', fontSize: 9 }} />
                {models.map((m, i) => (
                  <Radar
                    key={m.model}
                    name={shortModel(m.model)}
                    dataKey={shortModel(m.model)}
                    stroke={palette[i % palette.length]}
                    fill={palette[i % palette.length]}
                    fillOpacity={0.1}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 11.5 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => fmt(v)} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      <h2>Cost and speed</h2>
      <div className="grid2">
        <div className="card">
          <h3>Cost per model (USD)</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart
              data={models.map((m) => ({
                name: shortModel(m.model),
                cost: m.operational.cost_usd ?? 0,
              }))}
            >
              <CartesianGrid stroke="#243140" vertical={false} />
              <XAxis dataKey="name" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => money(v)} />
              <Bar dataKey="cost" fill="#d9a03f" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Mean latency and time-to-first-token</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart
              data={models.map((m) => ({
                name: shortModel(m.model),
                latency: (m.operational.latency_ms?.mean ?? 0) / 1000,
                ttft: (m.operational.ttft_ms?.mean ?? 0) / 1000,
              }))}
            >
              <CartesianGrid stroke="#243140" vertical={false} />
              <XAxis dataKey="name" {...AXIS} />
              <YAxis {...AXIS} unit="s" />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => `${v.toFixed(2)}s`} />
              <Legend wrapperStyle={{ fontSize: 11.5 }} />
              <Bar dataKey="latency" name="Total" fill="#4a9fd8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ttft" name="TTFT" fill="#5fb8b0" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {summary.consistency.by_model.length > 0 && (
        <>
          <h2>Consistency across repeats</h2>
          <div className="card">
            <p className="small muted" style={{ marginTop: 0 }}>
              Lexical similarity between repeats of the same prompt. Higher means more
              repeatable wording. This measures phrasing overlap, not semantic
              equivalence, because no embedding model is available on Groq.
            </p>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart
                data={summary.consistency.by_model.map((c) => ({
                  name: shortModel(c.model),
                  cosine: c.tfidf_cosine?.mean ?? 0,
                  rouge: c.rouge_l?.mean ?? 0,
                }))}
              >
                <CartesianGrid stroke="#243140" vertical={false} />
                <XAxis dataKey="name" {...AXIS} />
                <YAxis domain={[0, 1]} {...AXIS} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => fmt(v)} />
                <Legend wrapperStyle={{ fontSize: 11.5 }} />
                <Bar dataKey="cosine" name="TF-IDF cosine" fill="#4a9fd8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rouge" name="ROUGE-L" fill="#9d7fc0" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      <h2>Every cell</h2>
      <div className="card">
        <p className="small muted" style={{ marginTop: 0 }}>
          One row per prompt and model, averaged over repeats. Click to inspect a
          response with its numbers highlighted.
        </p>
        <table>
          <thead>
            <tr>
              <th>Prompt</th>
              <th>Technique</th>
              <th>Model</th>
              <th className="num">Composite</th>
              <th className="num">Grounding</th>
              <th className="num">Structure</th>
              <th className="num">Judge</th>
              <th className="num">Stability</th>
            </tr>
          </thead>
          <tbody>
            {summary.cells.map((c) => (
              <tr
                key={`${c.prompt_id}-${c.model}`}
                className="clickable"
                onClick={() => onOpenGeneration(c.generation_ids[0])}
              >
                <td className="mono">{c.prompt_id}</td>
                <td>{c.technique}</td>
                <td>{shortModel(c.model)}</td>
                <td className="num">{ci(c.composite)}</td>
                <td className="num">{pct(c.metrics.numeric_grounding?.mean)}</td>
                <td className="num">{pct(c.metrics.structural_compliance?.mean)}</td>
                <td className="num">{fmt(c.metrics.judge_overall?.mean, 2)}</td>
                <td className="num">{fmt(c.consistency?.tfidf_cosine, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {summary.failures.length > 0 && (
        <>
          <h2>Worst grounding failures</h2>
          <div className="card">
            <p className="small muted" style={{ marginTop: 0 }}>
              Ranked by severity, weighting contradictions of supplied facts above
              invented figures.
            </p>
            {summary.failures.map((f) => (
              <div key={f.generation_id} style={{ marginBottom: 18 }}>
                <div className="spread">
                  <strong>
                    {shortModel(f.model)} — {f.prompt_id} ({f.technique}, repeat{' '}
                    {f.repeat_index})
                  </strong>
                  <button className="link" onClick={() => onOpenGeneration(f.generation_id)}>
                    inspect
                  </button>
                </div>
                <div className="small muted">
                  {f.contradictions ?? 0} contradiction(s), {f.unsupported ?? 0}{' '}
                  unsupported figure(s), {f.word_count} words vs {f.word_limit} limit
                </div>
                <div className="row tight" style={{ marginTop: 6 }}>
                  {f.examples.slice(0, 8).map((ex, i) => (
                    <span
                      key={i}
                      className={`badge ${ex.classification === 'contradicting' ? 'bad' : 'warn'}`}
                      title={ex.evidence ?? ''}
                    >
                      {ex.raw}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}
