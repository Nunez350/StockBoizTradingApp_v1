import { createElement as h, useEffect, useState } from 'react'

// Run an async loader whenever `deps` change; expose {data, error, loading}. Stale responses are ignored.
export function useFetch(loader, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true, error: null }))
    loader().then(
      (data) => alive && setState({ data, error: null, loading: false }),
      (error) => alive && setState({ data: null, error: error.message, loading: false }),
    )
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}

export function Status({ loading, error, children }) {
  // plain createElement so this file can stay .js (no JSX transform needed)
  if (loading) return h('p', { className: 'muted' }, 'Loading…')
  if (error) return h('p', { className: 'error' }, error)
  return children
}
