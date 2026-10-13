import { useCallback, useEffect, useState } from 'react'
import { fetchReport, fmt, pct } from './api'
import type { Report } from './types'
import './App.css'

/** One agent's bar, with the honest interval drawn around the point estimate. */
function IntervalBar({
  label,
  mean,
  lo,
  hi,
  baseline,
  sig,
  note,
}: {
  label: string
  mean: number
  lo: number
  hi: number
  baseline?: boolean
  sig?: boolean
  note?: string
}) {
  // scale to the widest interval plus its point, with a little headroom
  const scale = Math.max(0.25, hi, mean) * 1.15
  const w = (v: number) => `${Math.max(0, Math.min(100, (v / scale) * 100))}%`
  return (
    <div className="bar-row">
      <div className="bar-label" title={note}>
        {label}
        {baseline && <span className="tag">baseline</span>}
        {sig && <span className="tag sig">significant</span>}
      </div>
      <div className="bar-track">
        <div
          className="bar-ci"
          style={{ left: w(lo), width: `calc(${w(hi)} - ${w(lo)})` }}
        />
        <div className="bar-mean" style={{ left: w(mean) }} />
      </div>
      <div className="bar-value">
        {fmt(mean)} <span className="dim">[{fmt(lo)}, {fmt(hi)}]</span>
      </div>
    </div>
  )
}

function EstimatorTable({ name, effect, highlight }: {
  name: string
  effect: Report['comparisons'][0]['estimators']['naive']
  highlight?: boolean
}) {
  return (
    <tr className={highlight ? 'hl' : ''}>
      <td>{name}</td>
      <td>{fmt(effect.se, 4)}</td>
      <td>{fmt(effect.p_value === null ? null : effect.p_value, 4)}</td>
      <td className="mono">[{fmt(effect.ci_low, 3)}, {fmt(effect.ci_high, 3)}]</td>
    </tr>
  )
}

export default function App() {
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [tasks, setTasks] = useState(48)
  const [seeds, setSeeds] = useState(3)
  const [seed, setSeed] = useState(11)

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const next = await fetchReport(
        { n_tasks: tasks, n_seeds: seeds, n_agents: 6, seed },
        signal,
      )
      setReport(next)
      setError(null)
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (!signal?.aborted) setBusy(false)
    }
  }, [tasks, seeds, seed])

  // Fetching on mount is a genuine effect, but the synchronous setState that used
  // to live here re-rendered before the request was even sent. `busy` is now set by
  // the event that changes a parameter, and the abort stops a slow response from
  // overwriting a newer one.
  useEffect(() => {
    const ctrl = new AbortController()
    void load(ctrl.signal)
    return () => ctrl.abort()
  }, [load])

  // Changing a parameter re-runs the effect; set `busy` here, in the event, so the
  // effect itself never calls setState synchronously.
  const edit = <T,>(setter: (v: T) => void) => (v: T) => {
    setBusy(true)
    setter(v)
  }

  const rebuild = () => {
    setBusy(true)
    void load()
  }

  if (error) {
    return (
      <div className="wrap">
        <h1>agenteval</h1>
        <div className="panel err">
          <strong>Could not reach the API.</strong>
          <p className="mono">{error}</p>
          <p className="dim">
            Start it with <code>python -m agenteval.cli serve</code>, then reload.
          </p>
        </div>
      </div>
    )
  }

  if (!report) {
    return <div className="wrap"><h1>agenteval</h1><p className="dim">building report…</p></div>
  }

  const { meta, leaderboard, judges, planner } = report
  const judgeList = Object.values(judges.judges).sort(
    (a, b) => b.sensitivity - a.sensitivity,
  )
  const worstInflation = Math.max(
    ...report.comparisons.map((c) => c.se_inflation_naive_to_crossed),
  )
  const rec = planner.recommended
  const mdd = planner.min_detectable_delta

  return (
    <div className="wrap">
      <header>
        <h1>agenteval</h1>
        <p className="dim">
          {meta.n_tasks} tasks × {meta.n_seeds} seeds × {meta.n_judges} judges ={' '}
          {meta.n_cells} cells · baseline <b>{meta.baseline}</b> · built in{' '}
          {fmt(meta.elapsed_s, 1)}s
        </p>
      </header>

      {report.warnings.map((w) => (
        <div className="panel warn" key={w}>
          <strong>Design warning.</strong> {w}
        </div>
      ))}

      <div className="panel">
        <h2>Design</h2>
        <div className="controls">
          <label>
            tasks
            <input type="number" min={8} max={256} value={tasks}
              onChange={(e) => edit(setTasks)(Number(e.target.value))} />
          </label>
          <label>
            seeds
            <input type="number" min={2} max={16} value={seeds}
              onChange={(e) => edit(setSeeds)(Number(e.target.value))} />
          </label>
          <label>
            seed
            <input type="number" min={0} value={seed}
              onChange={(e) => edit(setSeed)(Number(e.target.value))} />
          </label>
          <button onClick={rebuild} disabled={busy}>
            {busy ? 'running…' : 'rebuild'}
          </button>
        </div>
        <p className="dim">
          Seeds must be ≥ 2. With a single seed its variance is indistinguishable from
          task difficulty, and no interval can be corrected after the fact.
        </p>
      </div>

      <div className="panel">
        <h2>Leaderboard</h2>
        <p className="dim">
          Bars are the point estimate; the bar behind is the 95% interval from the
          crossed variance components. A wide bar means the ranking is not yet
          resolved.
        </p>
        {leaderboard.map((r) => (
          <IntervalBar key={r.agent} label={r.agent} mean={r.mean}
            lo={r.ci_low} hi={r.ci_high} baseline={r.is_baseline}
            sig={r.significant} note={r.ci_note} />
        ))}
      </div>

      <div className="panel">
        <h2>What the naive interval would have told you</h2>
        <p className="dim">
          Every contrast, three ways. The i.i.d. interval treats all{' '}
          {meta.n_cells} cells as independent, which they are not — and the axes do
          not share a denominator either.
        </p>
        <table>
          <thead>
            <tr><th>contrast</th><th>naive se</th><th>honest se</th><th>inflation</th><th>p (naive)</th><th>p (honest)</th></tr>
          </thead>
          <tbody>
            {report.comparisons.map((c) => (
              <tr key={c.arm}>
                <td>{c.arm} − {c.baseline}</td>
                <td>{fmt(c.estimators.naive.se, 4)}</td>
                <td>{fmt(c.estimators.crossed.se, 4)}</td>
                <td className="hl">{fmt(c.se_inflation_naive_to_crossed, 2)}×</td>
                <td>{fmt(c.estimators.naive.p_value, 4)}</td>
                <td>{fmt(c.estimators.crossed.p_value, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="callout">
          The naive interval is up to <b>{fmt(worstInflation, 1)}× too narrow</b>.
          That is the gap between a published result and one that survives
          replication.
        </p>
      </div>

      {report.comparisons[0] && (
        <div className="panel">
          <h2>Where the variance lives — {report.comparisons[0].arm}</h2>
          <p className="dim">
            Each axis averages down on its own: a seed's offset is shared by every
            task and judge run on it.
          </p>
          <table>
            <thead><tr><th>estimator</th><th>se</th><th>p</th><th>interval</th></tr></thead>
            <tbody>
              <EstimatorTable name="i.i.d. cells" effect={report.comparisons[0].estimators.naive} />
              <EstimatorTable name="task-clustered" effect={report.comparisons[0].estimators.cluster} />
              <EstimatorTable name="task bootstrap" effect={report.comparisons[0].estimators.bootstrap} />
              <EstimatorTable name="crossed components" effect={report.comparisons[0].estimators.crossed} highlight />
            </tbody>
          </table>
          <p className="dim">
            Task-clustering and the bootstrap agree with each other and both remain
            too narrow: resampling tasks cannot see the shared seed and judge
            offsets.
          </p>
        </div>
      )}

      <div className="panel">
        <h2>Judges are instruments</h2>
        <p className="dim">
          Planted operating points next to Dawid-Skene estimates, then the observed
          rate after inverting that bias. All four land on{' '}
          {fmt(judges.true_positive_rate)} once corrected, despite raw rates spread
          over {pct(judges.observed_positive_rate)} to{' '}
          {pct(Math.max(...judgeList.map((j) => j.observed_positive_rate)))}.
        </p>
        <table>
          <thead>
            <tr><th>judge</th><th>sens</th><th>spec</th><th>κ</th><th>observed</th><th>corrected</th></tr>
          </thead>
          <tbody>
            {judgeList.map((j) => (
              <tr key={j.name}>
                <td>{j.name}</td>
                <td className="mono">{fmt(j.planted.sensitivity, 2)} → {fmt(j.sensitivity)}</td>
                <td className="mono">{fmt(j.planted.specificity, 2)} → {fmt(j.specificity)}</td>
                <td>{fmt(j.cohen_kappa_vs_truth, 2)}</td>
                <td>{fmt(j.observed_positive_rate)}</td>
                <td className="hl">{fmt(j.bias_corrected_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>What to buy next</h2>
        <div className="grid">
          <div>
            <div className="stat-label">min detectable effect now</div>
            <div className="stat">{fmt(mdd.at_design)}</div>
            <div className="dim">at {meta.n_tasks}×{meta.n_seeds}×{meta.n_judges}</div>
          </div>
          <div>
            <div className="stat-label">target</div>
            <div className="stat">{fmt(meta.target_delta)}</div>
            <div className="dim">power now {pct(planner.current_power_at_target, 0)}</div>
          </div>
          {rec && (
            <div>
              <div className="stat-label">recommended</div>
              <div className="stat">
                {rec.n_tasks}×{rec.n_seeds}×{rec.n_judges}
              </div>
              <div className="dim">
                {fmt(rec.cost, 0)} cost · {fmt(rec.cost_ratio_vs_current, 0)}× current
              </div>
            </div>
          )}
        </div>
        <p className="dim">
          Cost-optimal allocation is n ∝ √(σ²/c) per axis, so the expensive axis gets
          fewer levels. Seed and judge variance also put a floor under the smallest
          effect this benchmark can ever resolve, no matter how many tasks are added.
        </p>
        <table>
          <thead>
            <tr><th>design</th><th>cells</th><th>cost</th><th>se</th><th>power</th></tr>
          </thead>
          <tbody>
            {planner.table
              .filter((r) => r.n_seeds === 3 && r.n_judges === 4)
              .map((r) => (
                <tr key={`${r.n_tasks}-${r.n_seeds}-${r.n_judges}`}
                  className={r.n_tasks === meta.n_tasks ? 'hl' : ''}>
                  <td>{r.n_tasks}×{r.n_seeds}×{r.n_judges}</td>
                  <td>{r.n_cells}</td>
                  <td>{fmt(r.cost, 0)}</td>
                  <td>{fmt(r.se, 4)}</td>
                  <td>{pct(r.power, 0)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
