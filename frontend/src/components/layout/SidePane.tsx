import { Settings, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { getPrefs, updatePrefs } from '../../api/client'
import type { DietaryOption, UserPrefs } from '../../types/api'
import { DIETARY_LABELS } from '../../types/api'

interface Props {
  open: boolean
  onClose: () => void
  recency: string | null
  onRecencyChange: (v: string | null) => void
}

const DIETARY_OPTIONS: DietaryOption[] = [
  'gluten_free', 'vegan', 'dairy_free', 'nut_free', 'paleo', 'keto',
]

export function SidePane({ open, onClose, recency, onRecencyChange }: Props) {
  const [prefs, setPrefs] = useState<UserPrefs>({
    dietary_restrictions: [],
    use_feedback_prefs: true,
    prefer_accessibility: 0,
  })

  useEffect(() => {
    getPrefs().then(setPrefs)
  }, [])

  async function toggleDietary(opt: DietaryOption) {
    const current = prefs.dietary_restrictions
    const next = current.includes(opt)
      ? current.filter((d) => d !== opt)
      : [...current, opt]
    const updated = { ...prefs, dietary_restrictions: next }
    setPrefs(updated)
    await updatePrefs({ dietary_restrictions: next })
  }

  async function toggleMemory() {
    const next = { ...prefs, use_feedback_prefs: !prefs.use_feedback_prefs }
    setPrefs(next)
    await updatePrefs({ use_feedback_prefs: next.use_feedback_prefs })
  }

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 bg-black/20 z-20"
          onClick={onClose}
        />
      )}

      {/* Pane */}
      <aside
        className={`fixed top-0 left-0 h-full w-72 bg-white border-r border-amber-100 shadow-xl z-30
          transform transition-transform duration-300
          ${open ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-amber-100">
          <div className="flex items-center gap-2 text-amber-800 font-semibold">
            <Settings size={18} />
            Settings
          </div>
          <button onClick={onClose} className="text-stone-400 hover:text-stone-600">
            <X size={18} />
          </button>
        </div>

        <div className="px-5 py-5 space-y-7 overflow-y-auto h-[calc(100%-56px)]">
          {/* Dietary restrictions */}
          <section>
            <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-3">
              Dietary Restrictions
            </h3>
            <div className="space-y-2">
              {DIETARY_OPTIONS.map((opt) => (
                <label key={opt} className="flex items-center gap-3 cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={prefs.dietary_restrictions.includes(opt)}
                    onChange={() => toggleDietary(opt)}
                    className="w-4 h-4 rounded accent-amber-600"
                  />
                  <span className="text-sm text-stone-700 group-hover:text-stone-900">
                    {DIETARY_LABELS[opt]}
                  </span>
                </label>
              ))}
            </div>
          </section>

          {/* Recipe Box Memory */}
          <section>
            <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-3">
              Recipe Box Memory
            </h3>
            <label className="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={prefs.use_feedback_prefs}
                onChange={toggleMemory}
                className="w-4 h-4 mt-0.5 rounded accent-amber-600"
              />
              <span className="text-sm text-stone-700 group-hover:text-stone-900">
                Use saved recipes to bias scoring weights toward your taste profile
              </span>
            </label>
          </section>

          {/* Recency filter */}
          <section>
            <h3 className="text-xs font-semibold text-stone-400 uppercase tracking-wider mb-3">
              Recipe Recency
            </h3>
            <div className="space-y-2">
              {[
                { value: null,    label: 'Any time' },
                { value: 'year',  label: 'Past year' },
                { value: 'month', label: 'Past month' },
              ].map(({ value, label }) => (
                <label key={label} className="flex items-center gap-3 cursor-pointer group">
                  <input
                    type="radio"
                    checked={recency === value}
                    onChange={() => onRecencyChange(value)}
                    className="w-4 h-4 accent-amber-600"
                  />
                  <span className="text-sm text-stone-700 group-hover:text-stone-900">
                    {label}
                  </span>
                </label>
              ))}
            </div>
          </section>
        </div>
      </aside>
    </>
  )
}
