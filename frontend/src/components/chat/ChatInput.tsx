import { Link, Search } from 'lucide-react'
import { useState } from 'react'
import { useAutoResize } from '../../hooks/useAutoResize'

type Mode = 'query' | 'url'

interface Props {
  onSearch: (query: string) => void
  onScoreUrl: (url: string) => void
  disabled?: boolean
}

export function ChatInput({ onSearch, onScoreUrl, disabled }: Props) {
  const [mode, setMode] = useState<Mode>('query')
  const [value, setValue] = useState('')
  const { ref, resize } = useAutoResize()

  function handleSubmit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    if (mode === 'url') {
      onScoreUrl(trimmed)
    } else {
      onSearch(trimmed)
    }
    setValue('')
    if (ref.current) ref.current.style.height = 'auto'
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const placeholder =
    mode === 'query'
      ? 'e.g. "chewy brown butter chocolate chip cookies that stay soft for days"'
      : 'Paste a recipe URL to score it'

  return (
    <div className="bg-white rounded-2xl border border-amber-100 shadow-sm p-4 space-y-3">
      {/* Mode toggle */}
      <div className="flex gap-1 bg-stone-100 rounded-xl p-1 w-fit">
        <ModeButton
          active={mode === 'query'}
          onClick={() => setMode('query')}
          icon={<Search size={13} />}
          label="Find a Recipe"
        />
        <ModeButton
          active={mode === 'url'}
          onClick={() => setMode('url')}
          icon={<Link size={13} />}
          label="Score a URL"
        />
      </div>

      {/* Input row */}
      <div className="flex gap-2 items-end">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => { setValue(e.target.value); resize() }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1 text-sm border border-stone-200 rounded-xl px-4 py-2.5 resize-none
            leading-relaxed overflow-hidden
            focus:outline-none focus:ring-2 focus:ring-amber-300
            placeholder:text-stone-300 disabled:bg-stone-50 disabled:text-stone-400"
        />
        <button
          type="button"
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          className="bg-amber-500 hover:bg-amber-600 disabled:opacity-40 shrink-0
            text-white text-sm font-medium px-5 py-2.5 rounded-xl transition-colors"
        >
          {disabled ? 'Baking…' : mode === 'url' ? 'Score' : 'Search'}
        </button>
      </div>
      <p className="text-[11px] text-stone-300 -mt-1">Enter to submit · Shift+Enter for new line</p>
    </div>
  )
}

function ModeButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all
        ${active ? 'bg-white text-amber-800 shadow-sm' : 'text-stone-500 hover:text-stone-700'}`}
    >
      {icon}
      {label}
    </button>
  )
}
