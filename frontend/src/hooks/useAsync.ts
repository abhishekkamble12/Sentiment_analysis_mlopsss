import { useCallback, useEffect, useState } from 'react'

interface State<T> {
  data: T | null
  loading: boolean
  error: string | null
}

/** Fetch on mount (and on `refetch`); exposes loading/error state without per-page boilerplate. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [state, setState] = useState<State<T>>({ data: null, loading: true, error: null })
  const [nonce, setNonce] = useState(0)

  const run = useCallback(() => {
    let cancelled = false
    setState((s) => ({ ...s, loading: true, error: null }))
    fn()
      .then((data) => !cancelled && setState({ data, loading: false, error: null }))
      .catch((err) => !cancelled && setState({ data: null, loading: false, error: msg(err) }))
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    const cancel = run()
    return cancel
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps])

  const refetch = useCallback(() => setNonce((n) => n + 1), [])
  return { ...state, refetch }
}

function msg(err: unknown): string {
  if (err instanceof Error) return err.message
  return 'Request failed'
}
