import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import GenerationDrawer from './components/GenerationDrawer'
import Results from './components/Results'
import RunLauncher from './components/RunLauncher'
import { fmt, shortModel, statusBadge } from './format'
import type { ConfigResponse, ProgressEvent, RunListItem, RunSummary } from './types'

export default function App() {
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [progress, setProgress] = useState<{ completed: number; total: number } | null>(null)
  const [log, setLog] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [openGeneration, setOpenGeneration] = useState<number | null>(null)
  const [reportBusy, setReportBusy] = useState(false)
  const sourceRef = useRef<EventSource | null>(null)

  const refreshRuns = useCallback(async () => {
    try {
      const list = await api.runs()
      setRuns(list)
      return list
    } catch (e) {
      setError(String((e as Error).message ?? e))
      return []
    }
  }, [])

  const loadSummary = useCallback(async (runId: string) => {
    try {
      setSummary(await api.run(runId))
    } catch (e) {
      setError(String((e as Error).message ?? e))
    }
  }, [])

  useEffect(() => {
    api.config().then(setConfig).catch((e) => setError(String(e.message ?? e)))
    refreshRuns().then((list) => {
      const active = list.find((r) => r.running) ?? list[0]
      if (active) setSelected(active.id)
    })
  }, [refreshRuns])

  // One SSE subscription at a time, following whichever run is selected.
  useEffect(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    setProgress(null)
    setLog([])

    if (!selected) {
      setSummary(null)
      return
    }
    loadSummary(selected)

    const source = new EventSource(api.eventsUrl(selected))
    sourceRef.current = source

    const handle = (raw: MessageEvent) => {
      let event: ProgressEvent
      try {
        event = JSON.parse(raw.data)
      } catch {
        return
      }

      if (event.completed !== undefined && event.total !== undefined) {
        setProgress({ completed: event.completed, total: event.total })
      }

      if (event.type === 'snapshot' && event.total) {
        setProgress({ completed: event.completed ?? 0, total: event.total })
      }

      if (event.type === 'task_done') {
        const label =
          event.status === 'ok'
            ? `${shortModel(event.model ?? '')} ${event.prompt_id} r${event.repeat_index} · ` +
              `grounding ${fmt(event.scores?.numeric_grounding, 2)} · judge ${fmt(event.judge_overall ?? null, 2)}`
            : `${shortModel(event.model ?? '')} ${event.prompt_id} r${event.repeat_index} · FAILED: ${event.error}`
        setLog((prev) => [...prev.slice(-160), label])
      } else if (event.type === 'phase') {
        setLog((prev) => [...prev.slice(-160), `computing ${event.phase}…`])
      } else if (event.type === 'run_finished' || event.type === 'run_cancelled') {
        setLog((prev) => [...prev.slice(-160), `run ${event.status ?? 'cancelled'}`])
        loadSummary(selected)
        refreshRuns()
        source.close()
      }
    }

    for (const name of [
      'snapshot',
      'run_started',
      'task_done',
      'phase',
      'run_finished',
      'run_cancelled',
    ]) {
      source.addEventListener(name, handle as EventListener)
    }
    source.onerror = () => source.close()

    return () => source.close()
  }, [selected, loadSummary, refreshRuns])

  const anyRunning = runs.some((r) => r.running)

  const onLaunched = async (runId: string) => {
    await refreshRuns()
    setSelected(runId)
  }

  const cancel = async () => {
    if (!selected) return
    await api.cancelRun(selected)
    await refreshRuns()
    loadSummary(selected)
  }

  const remove = async (runId: string) => {
    await api.deleteRun(runId)
    const list = await refreshRuns()
    if (selected === runId) setSelected(list[0]?.id ?? null)
  }

  const makeReport = async () => {
    if (!selected) return
    setReportBusy(true)
    setError(null)
    try {
      await api.buildReport(selected)
      await refreshRuns()
      window.open(api.reportUrl(selected), '_blank')
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setReportBusy(false)
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Prompt Eval</h1>
        <p className="tagline">
          Technique benchmark with fact-grounding metrics, a judge rubric and PDF
          reporting.
        </p>

        <h3>Runs</h3>
        {runs.length === 0 && <p className="small muted">No runs yet.</p>}
        <ul className="runlist">
          {runs.map((r) => (
            <li
              key={r.id}
              className={selected === r.id ? 'active' : ''}
              onClick={() => setSelected(r.id)}
            >
              <div className="spread">
                <span className="mono" style={{ fontSize: 11 }}>
                  {r.id.replace('run-', '')}
                </span>
                <span className={`badge ${statusBadge(r.status)}`}>{r.status}</span>
              </div>
              <div className="small muted" style={{ marginTop: 3 }}>
                {r.models.length} models · {r.prompt_ids.length} prompts · ×{r.repeats}
              </div>
              <div className="small muted">
                {r.completed_tasks}/{r.total_tasks} done
                {r.failed_tasks > 0 && ` · ${r.failed_tasks} failed`}
              </div>
              <div className="row tight" style={{ marginTop: 6 }}>
                <button
                  className="link small"
                  onClick={(e) => {
                    e.stopPropagation()
                    remove(r.id)
                  }}
                >
                  delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        {error && <div className="banner error">{error}</div>}

        {config && (
          <RunLauncher config={config} disabled={anyRunning} onLaunched={onLaunched} />
        )}

        {selected && (
          <div className="card">
            <div className="spread">
              <div>
                <h2 style={{ margin: 0 }}>{selected}</h2>
                <div className="small muted">
                  {summary?.run?.status ?? '…'}
                  {summary?.run?.judge_model && ` · judge ${shortModel(summary.run.judge_model)}`}
                </div>
              </div>
              <div className="row tight">
                {summary?.running && (
                  <button className="danger" onClick={cancel}>
                    Cancel run
                  </button>
                )}
                <button
                  className="primary"
                  disabled={reportBusy || !summary || summary.totals.ok === 0}
                  onClick={makeReport}
                >
                  {reportBusy ? 'Building…' : 'Generate PDF report'}
                </button>
                {summary?.has_report && (
                  <a href={api.reportUrl(selected)} target="_blank" rel="noreferrer">
                    <button>Download last report</button>
                  </a>
                )}
              </div>
            </div>

            {progress && progress.total > 0 && (
              <div style={{ marginTop: 14 }}>
                <div className="spread small muted" style={{ marginBottom: 5 }}>
                  <span>
                    {progress.completed} / {progress.total} generations
                  </span>
                  <span>{Math.round((progress.completed / progress.total) * 100)}%</span>
                </div>
                <div className="progress">
                  <div style={{ width: `${(progress.completed / progress.total) * 100}%` }} />
                </div>
              </div>
            )}

            {log.length > 0 && (
              <div className="log" style={{ marginTop: 14 }}>
                {log.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {summary && summary.totals.ok > 0 && (
          <Results summary={summary} onOpenGeneration={setOpenGeneration} />
        )}

        {summary && summary.totals.ok === 0 && !summary.running && (
          <div className="card muted">
            This run produced no successful generations.
            {summary.run?.total_tasks ? ` ${summary.totals.failed} failed.` : ''}
          </div>
        )}

        {!selected && !anyRunning && (
          <div className="card muted">
            Select a run on the left, or start a new benchmark above.
          </div>
        )}
      </main>

      {openGeneration !== null && (
        <GenerationDrawer
          generationId={openGeneration}
          onClose={() => setOpenGeneration(null)}
        />
      )}
    </div>
  )
}
