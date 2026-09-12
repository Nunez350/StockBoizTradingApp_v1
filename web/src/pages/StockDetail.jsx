import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Plotly from 'plotly.js-basic-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import { api, fmt } from '../api.js'
import { Status, useFetch } from '../useFetch.js'

const Plot = createPlotlyComponent(Plotly)
const RANGES = [['6M', 126], ['1Y', 252], ['2Y', 504], ['5Y', 1260], ['Max', 10000]]

export default function StockDetail() {
  const { symbol } = useParams()
  const [sessions, setSessions] = useState(504)
  const { data: d, error, loading } = useFetch(() => api.stock(symbol, sessions), [symbol, sessions])
  const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
  const fg = dark ? '#d6d9de' : '#222'

  return (
    <>
      <p><Link to="/stocks">← Stocks</Link></p>
      <Status loading={loading} error={error}>
        {d && (
          <>
            <div className="row">
              <h1>{d.symbol} <span className="muted">{d.name ?? ''}</span></h1>
              <div className="ranges">{RANGES.map(([l, n]) => <button key={l} className={n === sessions ? 'active' : ''} onClick={() => setSessions(n)}>{l}</button>)}</div>
            </div>
            <div className="tiles">
              <div className="tile"><span>last close</span><b>{fmt.num(d.last_close)}</b><small>{d.last_session}</small></div>
              <div className="tile"><span>sector</span><b>{d.sector}</b></div>
              <div className="tile"><span>S&amp;P 500 member</span><b className={d.member ? 'good' : 'muted'}>{d.member ? 'yes' : 'no'}</b><small>{d.first_member ? `since ${d.first_member}` : ''}</small></div>
              <div className="tile"><span>tradable</span><b className={d.tradable ? 'good' : 'muted'}>{d.tradable ? 'yes' : 'no'}</b><small>price + liquidity filters</small></div>
            </div>
            <div className="card">
              <Plot
                data={[
                  { x: d.history.map((r) => r.date), y: d.history.map((r) => r.adj_close), type: 'scatter', mode: 'lines', name: 'adj close', line: { color: '#3b82f6', width: 1.5 } },
                  { x: d.history.map((r) => r.date), y: d.history.map((r) => r.volume), type: 'bar', name: 'volume', yaxis: 'y2', marker: { color: dark ? '#4b5563' : '#cbd5e1' } },
                ]}
                layout={{
                  autosize: true, height: 420, margin: { l: 50, r: 50, t: 10, b: 40 },
                  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: fg },
                  xaxis: { gridcolor: 'rgba(128,128,128,0.15)' },
                  yaxis: { title: { text: 'total-return adjusted close' }, gridcolor: 'rgba(128,128,128,0.15)' },
                  yaxis2: { overlaying: 'y', side: 'right', showgrid: false, domain: [0, 0.25], title: { text: 'volume' } },
                  legend: { orientation: 'h', y: 1.05 }, showlegend: true,
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
                useResizeHandler
              />
            </div>
            <div className="card">
              <h3>Latest feature values</h3>
              <p className="muted">Computed by the registered feature functions at the last session in the snapshot. Each uses only data up to that date.</p>
              {Object.keys(d.features).length === 0 ? <p className="muted">No feature cache for this snapshot; run <code>quantlab features build</code>.</p> : (
                <div className="features">
                  {Object.entries(d.features).map(([k, v]) => <div key={k}><code>{k}</code><b>{fmt.sig(v)}</b></div>)}
                </div>
              )}
            </div>
          </>
        )}
      </Status>
    </>
  )
}
