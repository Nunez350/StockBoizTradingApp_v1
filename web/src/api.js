// Thin fetch layer over the quantlab API. Every call returns parsed JSON or throws with the server's detail.
async function get(path, params) {
  const url = new URL(path, window.location.origin)
  if (params) Object.entries(params).forEach(([k, v]) => v != null && v !== '' && url.searchParams.set(k, v))
  const res = await fetch(url)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not JSON */ }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export const api = {
  health: () => get('/api/health'),
  suggestions: (date) => get('/api/suggestions', { date }),
  searchStocks: (q, limit = 25) => get('/api/stocks', { q, limit }),
  stock: (symbol, sessions = 504) => get(`/api/stocks/${encodeURIComponent(symbol)}`, { sessions }),
  runs: (family) => get('/api/runs', { family }),
  strategies: () => get('/api/strategies'),
}

export const fmt = {
  num: (v, d = 2) => (v == null || Number.isNaN(v) ? '–' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })),
  int: (v) => (v == null ? '–' : Number(v).toLocaleString()),
  usd: (v) => (v == null ? '–' : Number(v).toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })),
  pct: (v, d = 2) => (v == null ? '–' : `${(Number(v) * 100).toFixed(d)}%`),
  sig: (v) => (v == null ? '–' : Math.abs(v) >= 1000 ? Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 }) : Number(v).toPrecision(4)),
}
