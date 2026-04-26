import { useCallback, useRef } from 'react'

/** Returns a ref and a change handler that keeps a textarea's height fitted to its content. */
export function useAutoResize() {
  const ref = useRef<HTMLTextAreaElement>(null)

  const resize = useCallback(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [])

  return { ref, resize }
}
