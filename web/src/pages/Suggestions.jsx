import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, fmt } from '../api.js'
import { Status, useFetch } from '../useFetch.js'

export default function Suggestions() {
  const [params, setParams] = useSearchParams()
  const date = params.get('date')
  const { data, error, loading } = useFetch(() => api.suggestions(date), [date])
  const [sort, setSort] = useState({ key: 'notional_est', dir: -1 })
  const [showAllTargets, setShowAllTargets] = useState(false)

  const orders = useMemo(() => {
    const rows = [...(data?.orders ?? [])]
    rows.sort((a, b) => ((a[sort.key] ?? '') > (b[sort.key] ?? '') ? 1 : -1) * sort.dir)
    return rows
  }, [data, sort])

  const targets = useMemo(() => {
    const rows = [...(data?.targets ?? [])].sort((a, b) => Math.abs(b.target_notional) - Math.abs(a.target_notional))
    return showAllTargets ? rows : rows.slice(0, 40)
  }, [data, showAllTargets])

  const th = (key, label) => (
    <th className="sortable" onClick={() => setSort((s) => ({ key, dir: s.key === key ? -s.dir : -1 }))}>
      {label}{sort.key === key ? (sort.dir < 0 ? ' ▾' : ' ▴') : ''}
    </th>
  )

  return (
    <>
      <div className="row">
        <h1>Suggestions</h1>
        {data && (
          <label className="muted">Report date{' '}
            <select value={data.date} onChange={(e) => setParams({ date: e.target.value })}>
              {data.dates.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
        )}
      </div>
      <Status loading={loading} error={error}>
        {data && (
          <>
            {data.kill_switch && <div className="banner bad">Kill switch file present: the daily job exits without trading until it is removed.</div>}
            {data.safety_abort && <div className="banner bad">Safety rail aborted before submission: <b>{data.safety_abort}</b>. No orders were written for this date.</div>}
            {data.mode === 'dry_run' && <div className="banner">Dry run: these orders were computed and journaled but never sent to the broker.</div>}
            <div className="tiles">
              <div className="tile"><span>orders</span><b>{data.orders.length}</b></div>
              <div className="tile"><span>buys</span><b className="good">{data.n_buy}</b></div>
              <div className="tile"><span>sells</span><b className="bad">{data.n_sell}</b></div>
              <div className="tile"><span>gross notional</span><b>{fmt.usd(data.gross_notional)}</b></div>
              <div className="tile"><span>target names</span><b>{data.targets.length}</b></div>
            </div>

            {orders.length > 0 && (
              <div className="card">
                <h3>Orders</h3>
                <div className="scroll">
                  <table>
                    <thead><tr>{th('symbol', 'symbol')}<th>name</th><th>sector</th>{th('side', 'side')}{th('qty', 'shares')}{th('notional_est', 'notional')}</tr></thead>
                    <tbody>
                      {orders.map((o) => (
                        <tr key={o.symbol}>
                          <td><Link to={`/stocks/${o.symbol}`}>{o.symbol}</Link></td>
                          <td>{o.name ?? <span className="muted">–</span>}</td>
                          <td className="muted">{o.sector}</td>
                          <td><span className={`tag ${o.side === 'buy' ? 'good' : 'bad'}`}>{o.side}</span></td>
                          <td className="num">{fmt.int(Math.abs(o.qty))}</td>
                          <td className="num">{fmt.usd(o.notional_est)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {data.targets.length > 0 && (
              <div className="card">
                <div className="row"><h3>Target book</h3>
                  <button className="link" onClick={() => setShowAllTargets((v) => !v)}>{showAllTargets ? 'show top 40' : `show all ${data.targets.length}`}</button>
                </div>
                <div className="scroll">
                  <table>
                    <thead><tr><th>symbol</th><th>name</th><th>weight</th><th>ref price</th><th>target $</th><th>target shares</th></tr></thead>
                    <tbody>
                      {targets.map((t) => (
                        <tr key={t.symbol}>
                          <td><Link to={`/stocks/${t.symbol}`}>{t.symbol}</Link></td>
                          <td>{t.name ?? <span className="muted">–</span>}</td>
                          <td className={`num ${t.target_w > 0 ? 'good' : 'bad'}`}>{fmt.pct(t.target_w)}</td>
                          <td className="num">{fmt.num(t.ref_price)}</td>
                          <td className="num">{fmt.usd(t.target_notional)}</td>
                          <td className="num">{fmt.int(t.target_shares)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {data.reconciliation && (
              <div className="card"><h3>Post-trade reconciliation</h3><pre>{data.reconciliation}</pre></div>
            )}
          </>
        )}
      </Status>
    </>
  )
}
