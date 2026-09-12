import { useMemo, useState } from 'react'
import { api, fmt } from '../api.js'
import { Status, useFetch } from '../useFetch.js'

const COUNTED = new Set(['trial', 'ml_hp', 'grid'])

export default function Runs() {
  const { data, error, loading } = useFetch(api.runs, [])
  const [family, setFamily] = useState('')
  const [countedOnly, setCountedOnly] = useState(false)
  const families = useMemo(() => [...new Set((data ?? []).map((r) => r.family))].sort(), [data])
  const rows = useMemo(() => (data ?? []).filter((r) => (!family || r.family === family) && (!countedOnly || COUNTED.has(r.kind))), [data, family, countedOnly])

  return (
    <>
      <h1>Registered runs</h1>
      <p className="muted">Every backtest trial is logged so the deflated Sharpe ratio can count how many things were tried. Kinds <code>trial</code>, <code>ml_hp</code> and <code>grid</code> count toward that; folds, permutations and sanity checks do not.</p>
      <div className="row">
        <label className="muted">Family{' '}
          <select value={family} onChange={(e) => setFamily(e.target.value)}>
            <option value="">all</option>
            {families.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label className="muted"><input type="checkbox" checked={countedOnly} onChange={(e) => setCountedOnly(e.target.checked)} /> counted trials only</label>
      </div>
      <Status loading={loading} error={error}>
        <div className="card">
          <div className="scroll">
            <table>
              <thead><tr><th>run</th><th>family</th><th>kind</th><th>obs</th><th>Sharpe</th><th>skew</th><th>snapshot</th><th>git</th><th>tearsheet</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.run_id}>
                    <td><code>{r.run_id}</code></td>
                    <td>{r.family}</td>
                    <td><span className={`tag ${COUNTED.has(r.kind) ? '' : 'muted'}`}>{r.kind}</span></td>
                    <td className="num">{fmt.int(r.n_obs)}</td>
                    <td className={`num ${r.sharpe > 0 ? 'good' : 'bad'}`}>{fmt.num(r.sharpe)}</td>
                    <td className="num">{fmt.num(r.skew)}</td>
                    <td className="muted"><code>{r.snapshot_id}</code></td>
                    <td className="muted"><code>{(r.git_sha ?? '').slice(0, 7)}</code></td>
                    <td>{r.has_tearsheet ? <a href={`/reports/${r.run_id}/tearsheet.html`} target="_blank" rel="noreferrer">open ↗</a> : <span className="muted">–</span>}</td>
                  </tr>
                ))}
                {rows.length === 0 && <tr><td colSpan="9" className="muted">Registry is empty. Run <code>quantlab run configs/strategies/meanrev_5d.yaml</code>.</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </Status>
    </>
  )
}
