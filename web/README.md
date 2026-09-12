# StockBoiz web UI

Vite + React front end over the quantlab JSON API. Pages: Home (data status, latest suggestions,
strategies on file), Suggestions (the paper-trading job's order book and target weights), Stocks
(search the point-in-time universe, Plotly price chart, latest feature values), Runs (the registry
with tearsheet links), Contact.

```bash
# terminal 1: API on :8000 (serves web/dist too, once built)
uv run quantlab api serve

# terminal 2: dev server on :5173 with /api and /reports proxied to :8000
cd web && npm install && npm run dev

# production: build once, then only the API process is needed
cd web && npm run build && cd .. && uv run quantlab api serve --host 0.0.0.0
```

No Redux, no sagas: each page fetches what it shows with a small `useFetch` hook. The chart uses
`plotly.js-basic-dist-min` so the bundle stays around 1 MB.
