import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api.js'
import { Status, useFetch } from '../useFetch.js'

export default function Stocks() {
  const [params, setParams] = useSearchParams()
  const [q, setQ] = useState(params.get('q') ?? '')
  const [query, setQuery] = useState(q)
  useEffect(() => {
    const t = setTimeout(() => { setQuery(q); setParams(q ? { q } : {}, { replace: true }) }, 200)
    return () => clearTimeout(t)
  }, [q, setParams])
  const { data, error, loading } = useFetch(() => api.searchStocks(query, 50), [query])

  return (
    <>
      <h1>Stocks</h1>
      <p className="muted">Search the point-in-time S&amp;P 500 universe by ticker or company name. Names are known for current constituents; historical members show a ticker only.</p>
      <input className="search" autoFocus placeholder="AAPL, Apple, semiconductors…" value={q} onChange={(e) => setQ(e.target.value)} />
      <Status loading={loading} error={error}>
        <div className="card">
          <table>
            <thead><tr><th>symbol</th><th>name</th><th>sector</th><th>member</th><th>tradable</th></tr></thead>
            <tbody>
              {(data ?? []).map((r) => (
                <tr key={r.symbol}>
                  <td><Link to={`/stocks/${r.symbol}`}>{r.symbol}</Link></td>
                  <td>{r.name ?? <span className="muted">historical member</span>}</td>
                  <td className="muted">{r.sector}</td>
                  <td>{r.member ? <span className="tag good">yes</span> : <span className="tag">no</span>}</td>
                  <td>{r.tradable ? <span className="tag good">yes</span> : <span className="tag">no</span>}</td>
                </tr>
              ))}
              {data && data.length === 0 && <tr><td colSpan="5" className="muted">No match.</td></tr>}
            </tbody>
          </table>
        </div>
      </Status>
    </>
  )
}
