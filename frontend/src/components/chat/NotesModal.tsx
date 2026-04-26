import { X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

interface Props {
  recipeTitle: string
  onSave: (notes: string) => void
  onCancel: () => void
  saving: boolean
}

export function NotesModal({ recipeTitle, onSave, onCancel, saving }: Props) {
  const [notes, setNotes] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape') onCancel()
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel() }}
      onKeyDown={handleKeyDown}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-stone-800">Save to Recipe Box</h2>
            <p className="text-sm text-stone-500 mt-0.5 leading-snug">{recipeTitle}</p>
          </div>
          <button onClick={onCancel} className="text-stone-400 hover:text-stone-600 mt-0.5 shrink-0">
            <X size={18} />
          </button>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-stone-500">
            Notes <span className="font-normal text-stone-400">(optional)</span>
          </label>
          <textarea
            ref={textareaRef}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="What do you like about it? Any tweaks you'd make?"
            rows={4}
            className="w-full text-sm border border-stone-200 rounded-xl px-3 py-2.5 resize-none
              focus:outline-none focus:ring-2 focus:ring-amber-300 placeholder:text-stone-300"
          />
        </div>

        <div className="flex gap-2 pt-1">
          <button
            onClick={() => onSave(notes)}
            disabled={saving}
            className="flex-1 bg-amber-500 hover:bg-amber-600 disabled:opacity-50
              text-white text-sm font-medium py-2.5 rounded-xl transition-colors"
          >
            {saving ? 'Saving…' : 'Save to Recipe Box'}
          </button>
          <button
            onClick={onCancel}
            className="px-4 text-sm text-stone-500 hover:text-stone-700"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
