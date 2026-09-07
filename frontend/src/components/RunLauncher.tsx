import { useEffect, useState } from 'react'
import { api } from '../api'
import { money, shortModel, tokens } from '../format'
import type { ConfigResponse, Estimate, RunPreset } from '../types'

interface Props {
  config: ConfigResponse
  disabled: boolean
  onLaunched: (runId: string) => void
}

export default function RunLauncher({ config, disabled, onLaunched }: Props) {
  const [models, setModels] = useState<string[]>(config.models.map((m) => m.id))
  const [queries, setQueries] = useState<string[]>([
    ...new Set(config.prompts.map((p) => p.query_id)),
  ])
  const [techniques, setTechniques] = useState<string[]>([
    ...new Set(config.prompts.map((p) => p.technique)),
  ])
  const [repeats, setRepeats] = useState(config.defaults.repeats)
  const [judge, setJudge] = useState(config.defaults.judge_enabled)
  const [judgeModel, setJudgeModel] = useState(config.defaults.judge_model)
  const [judgeRepeats, setJudgeRepeats] = useState(
    config.defaults.judge_repeats_per_cell ?? 1,
  )
  const [activePreset, setActivePreset] = useState<string | null>('free_tier')
  const [estimate, setEstimate] = useState<Estimate | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const promptIds = config.prompts
    .filter((p) => queries.includes(p.query_id) && techniques.includes(p.technique))
    .map((p) => p.prompt_id)

  const allQueries = [...new Set(config.prompts.map((p) => p.query_id))]
  const allTechniques = [...new Set(config.prompts.map((p) => p.technique))]

  // Re-price whenever the matrix changes, so cost is visible before spending.
  useEffect(() => {
    if (!models.length || !promptIds.length) {
      setEstimate(null)
      return
    }
    let active = true
    api
      .estimate({
        models,
        prompt_ids: promptIds,
        repeats,
        judge_enabled: judge,
        judge_model: judgeModel,
        judge_repeats_per_cell: judgeRepeats,
      })
      .then((e) => active && setEstimate(e))
      .catch((e) => active && setError(String(e.message ?? e)))
    return () => {
      active = false
    }
  }, [models.join(','), promptIds.join(','), repeats, judge, judgeModel, judgeRepeats])

  const toggle = (list: string[], setter: (v: string[]) => void, value: string) =>
    setter(list.includes(value) ? list.filter((v) => v !== value) : [...list, value])

  const launch = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await api.createRun({
        models,
        prompt_ids: promptIds,
        repeats,
        judge_enabled: judge,
        judge_model: judgeModel,
        judge_repeats_per_cell: judgeRepeats,
      })
      onLaunched(res.run_id)
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setBusy(false)
    }
  }

  const applyPreset = (preset: RunPreset) => {
    const allQueries = [...new Set(config.prompts.map((p) => p.query_id))]
    const allTechniques = [...new Set(config.prompts.map((p) => p.technique))]
    if (preset.models) setModels(preset.models)
    if (preset.prompt_ids) {
      const selected = config.prompts.filter((p) => preset.prompt_ids?.includes(p.prompt_id))
      setQueries([...new Set(selected.map((p) => p.query_id))])
      setTechniques([...new Set(selected.map((p) => p.technique))])
    } else {
      setQueries(allQueries)
      setTechniques(allTechniques)
    }
    if (preset.repeats) setRepeats(preset.repeats)
    if (preset.judge_enabled !== undefined) setJudge(preset.judge_enabled)
    if (preset.judge_model) setJudgeModel(preset.judge_model)
    if (preset.judge_repeats_per_cell !== undefined) {
      setJudgeRepeats(preset.judge_repeats_per_cell)
    }
    setActivePreset(preset.id)
  }

  const budgetFits = estimate?.budget?.fits !== false
  const canLaunch =
    !disabled && !busy && models.length > 0 && promptIds.length > 0 && budgetFits

  return (
    <div className="card">
      <div className="spread">
        <h2 style={{ margin: 0 }}>New benchmark run</h2>
        <span className="badge">
          {models.length} models × {promptIds.length} prompts × {repeats} repeats ={' '}
          {models.length * promptIds.length * repeats} generations
        </span>
      </div>

      {error && <div className="banner error" style={{ marginTop: 12 }}>{error}</div>}

      {config.presets.length > 0 && (
        <>
          <h3>Preset</h3>
          <div className="row tight">
            {config.presets.map((preset) => (
              <button
                key={preset.id}
                className={activePreset === preset.id ? 'primary' : undefined}
                type="button"
                onClick={() => applyPreset(preset)}
                title={preset.description}
              >
                {preset.label}
              </button>
            ))}
          </div>
          {activePreset && (
            <p className="small muted" style={{ marginTop: 8 }}>
              {config.presets.find((p) => p.id === activePreset)?.description}
            </p>
          )}
        </>
      )}

      <h3>Models</h3>
      <div className="row">
        {config.models.map((m) => (
          <label className="check" key={m.id}>
            <input
              type="checkbox"
              checked={models.includes(m.id)}
              onChange={() => toggle(models, setModels, m.id)}
            />
            {m.label}
            {m.id === config.defaults.judge_model && (
              <span className="badge warn" title="Also the judge model; its own rows are flagged as self-graded">
                judge
              </span>
            )}
          </label>
        ))}
      </div>

      <h3>Queries</h3>
      <div className="row">
        {allQueries.map((q) => (
          <label className="check" key={q}>
            <input
              type="checkbox"
              checked={queries.includes(q)}
              onChange={() => toggle(queries, setQueries, q)}
            />
            {q}
          </label>
        ))}
      </div>

      <h3>Techniques</h3>
      <div className="row">
        {allTechniques.map((t) => (
          <label className="check" key={t}>
            <input
              type="checkbox"
              checked={techniques.includes(t)}
              onChange={() => toggle(techniques, setTechniques, t)}
            />
            {t}
          </label>
        ))}
      </div>

      <h3>Settings</h3>
      <div className="row">
        <label className="field">
          Repeats per cell
          <input
            type="number"
            min={1}
            max={10}
            value={repeats}
            style={{ width: 80 }}
            onChange={(e) => setRepeats(Math.max(1, Math.min(10, Number(e.target.value))))}
          />
        </label>
        <label className="check" style={{ marginTop: 14 }}>
          <input type="checkbox" checked={judge} onChange={() => setJudge(!judge)} />
          LLM-as-judge
        </label>
        {judge && (
          <>
            <label className="field">
              Judge model
              <select value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)}>
                {config.judge_options.map((id) => (
                  <option key={id} value={id}>
                    {shortModel(id)}
                    {models.includes(id) ? ' (also a candidate → self-graded rows)' : ''}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Judge repeats per cell
              <input
                type="number"
                min={0}
                max={repeats}
                value={judgeRepeats}
                style={{ width: 80 }}
                onChange={(e) =>
                  setJudgeRepeats(Math.max(0, Math.min(repeats, Number(e.target.value))))
                }
              />
            </label>
          </>
        )}
      </div>
      <p className="small muted" style={{ marginBottom: 0 }}>
        Repeats above 1 are what make the consistency and confidence-interval
        metrics possible; a single run per cell reports no variance.
        {judge && (
          <>
            {' '}
            Judge repeats of 1 grades the first output of every cell and is
            what keeps a full matrix inside the free-tier daily token cap; 0
            means grade every repeat.
          </>
        )}
        {judge && models.includes(judgeModel) && (
          <>
            {' '}
            The judge is also under test, so its own rows will be flagged as
            self-graded; pick a different judge to avoid that.
          </>
        )}
      </p>

      {estimate && (
        <>
          <h3>Cost estimate</h3>
          <div className="stat-tiles">
            <div className="tile">
              <div className="label">Generation</div>
              <div className="value">{money(estimate.generation_cost_usd)}</div>
              <div className="sub">{estimate.generations} calls</div>
            </div>
            <div className="tile">
              <div className="label">Judging</div>
              <div className="value">{money(estimate.judge_cost_usd)}</div>
              <div className="sub">
                {estimate.judge_calls} calls
                {estimate.judge_sampled ? ' (sampled)' : ''}
              </div>
            </div>
            <div className="tile">
              <div className="label">Total</div>
              <div className="value">{money(estimate.total_cost_usd)}</div>
              <div className="sub">
                {estimate.per_model[0]?.completion_estimate_source === 'observed'
                  ? 'from observed history'
                  : 'nominal estimate'}
              </div>
            </div>
          </div>
          {estimate.unpriced_models.length > 0 && (
            <div className="banner" style={{ marginTop: 12 }}>
              No pricing configured for {estimate.unpriced_models.join(', ')}; their
              cost is excluded from this estimate.
            </div>
          )}
          {estimate.budget && estimate.budget.enabled && (
            <>
              <h3>Daily token budget</h3>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="num">Used today</th>
                    <th className="num">This run</th>
                    <th className="num">Left</th>
                    <th>Fits</th>
                  </tr>
                </thead>
                <tbody>
                  {estimate.budget.models
                    .filter((m) => (m.projected ?? 0) > 0 || m.used > 0)
                    .map((m) => (
                      <tr key={m.model}>
                        <td>{shortModel(m.model)}</td>
                        <td className="num">
                          {tokens(m.used)}
                          {m.limit ? ` / ${tokens(m.limit)}` : ''}
                        </td>
                        <td className="num">{tokens(m.projected)}</td>
                        <td className="num">{tokens(m.remaining)}</td>
                        <td>
                          <span className={`badge ${m.fits ? 'good' : 'bad'}`}>
                            {m.fits ? 'yes' : `short ${tokens(m.shortfall)}`}
                          </span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
              {!estimate.budget.fits && (
                <div className="banner error" style={{ marginTop: 12 }}>
                  This matrix does not fit in today&apos;s remaining allowance
                  for {(estimate.budget.blocking_models || []).map(shortModel).join(', ')}.
                  Lower repeats, sample the judge, pick a different judge, or wait
                  for the rolling 24h window to reset.
                </div>
              )}
            </>
          )}
          <p className="small muted">{estimate.note}</p>
        </>
      )}

      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" disabled={!canLaunch} onClick={launch}>
          {busy ? 'Starting…' : 'Start run'}
        </button>
        {disabled && <span className="small muted">A run is already in progress.</span>}
        {promptIds.length === 0 && (
          <span className="small muted">Select at least one query and technique.</span>
        )}
      </div>
    </div>
  )
}
