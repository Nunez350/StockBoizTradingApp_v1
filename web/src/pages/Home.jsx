import { Link } from 'react-router-dom'
import { api, fmt } from '../api.js'
import { Status, useFetch } from '../useFetch.js'

export default function Home() {
  const health = useFetch(api.health, [])
  const sugg = useFetch(() => api.suggestions().catch(() => null), [])
  const strategies = useFetch(api.strategies, [])
  const h = health.data
  const s = sugg.data

  return (
    <>
      <section className="hero">
        <h1>Buy and sell suggestions, with the statistics that say how much to trust them.</h1>
        <p>
          StockBoiz v1 and v2 promised trade suggestions on top of a symbol lookup. v3 keeps the name and the pages
          and swaps the backend for <b>quantlab</b>: a point-in-time S&amp;P 500 universe, an honest backtester,
          a run registry that deflates every Sharpe ratio by the number of trials, and a paper-trading job with safety rails.
        </p>
      </section>

      <div className="grid">
        <div className="card">
          <h3>Data</h3>
          <Status loading={health.loading} error={health.error}>
            {h && h.status === 'ok' ? (
              <dl>
                <dt>Snapshot</dt><dd><code>{h.snapshot_id}</code></dd>
                <dt>Sessions</dt><dd>{h.first_session} → {h.last_session}</dd>
                <dt>Symbols</dt><dd>{fmt.int(h.n_symbols)} in panel, {fmt.int(h.n_tradable)} tradable today</dd>
                <dt>Kill switch</dt><dd className={h.kill_switch ? 'bad' : 'good'}>{h.kill_switch ? 'PRESENT: job halted' : 'clear'}</dd>
              </dl>
            ) : <p className="muted">{h?.detail ?? 'No curated snapshot yet. Run the data commands in the README.'}</p>}
          </Status>
        </div>

        <div className="card">
          <h3>Latest suggestions</h3>
          <Status loading={sugg.loading} error={sugg.error}>
            {s ? (
              <>
                <dl>
                  <dt>Date</dt><dd>{s.date} <span className="tag">{s.mode ?? 'no orders'}</span></dd>
                  <dt>Orders</dt><dd><span className="good">{s.n_buy} buy</span> · <span className="bad">{s.n_sell} sell</span></dd>
                  <dt>Gross</dt><dd>{fmt.usd(s.gross_notional)}</dd>
                  {s.safety_abort && <><dt>Safety</dt><dd className="bad">ABORTED: {s.safety_abort}</dd></>}
                </dl>
                <Link className="btn" to="/suggestions">Open the order book</Link>
              </>
            ) : <p className="muted">No live reports yet. Run <code>quantlab live run --dry-run</code>.</p>}
          </Status>
        </div>

        <div className="card">
          <h3>Strategies on file</h3>
          <Status loading={strategies.loading} error={strategies.error}>
            <table className="compact">
              <thead><tr><th>name</th><th>signal</th><th>rebalance</th><th>portfolio</th></tr></thead>
              <tbody>
                {(strategies.data ?? []).map((r) => (
                  <tr key={r.file}>
                    <td>{r.name ?? <span className="bad">{r.error}</span>}</td>
                    <td>{r.signal}{r.meta ? ' + meta' : ''}</td>
                    <td>{r.rebalance}</td>
                    <td>{r.portfolio}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Status>
        </div>
      </div>
    </>
  )
}
