import { Search, SortAsc, Trash2, ChevronDown, ChevronUp, Pencil, Check } from 'lucide-react'
import { useEffect, useState } from 'react'
import { deleteRecipe, getRecipeBox, updateNotes } from '../../api/client'
import type { Category, RecipeBoxRow, ScoredRecipe } from '../../types/api'
import { CATEGORY_LABELS } from '../../types/api'
import { CompositeScore, ScoreBar } from '../chat/ScoreBar'

type SortOption = 'date_desc' | 'date_asc' | 'title_asc' | 'title_desc'

const SORT_LABELS: Record<SortOption, string> = {
  date_desc:  'Newest first',
  date_asc:   'Oldest first',
  title_asc:  'A → Z',
  title_desc: 'Z → A',
}

const CATEGORIES: Category[] = [
  'cookie', 'quick_bread', 'cake', 'yeasted_bread', 'pastry', 'brownie', 'other',
]

export function RecipeBoxTab() {
  const [rows, setRows] = useState<RecipeBoxRow[]>([])
  const [sort, setSort] = useState<SortOption>('date_desc')
  const [categoryFilter, setCategoryFilter] = useState<Category | null>(null)
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')

  // Debounce search input
  useEffect(() => {
    const id = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(id)
  }, [search])

  useEffect(() => {
    getRecipeBox({
      sort,
      category: categoryFilter ?? undefined,
      search: debouncedSearch || undefined,
    }).then(setRows)
  }, [sort, categoryFilter, debouncedSearch])

  function handleDeleted(url: string) {
    setRows((prev) => prev.filter((r) => r.url !== url))
  }

  function handleNotesUpdated(url: string, notes: string) {
    setRows((prev) => prev.map((r) => r.url === url ? { ...r, notes } : r))
  }

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-center">
        {/* Search */}
        <div className="relative flex-1 min-w-48">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search title or notes…"
            className="w-full pl-8 pr-3 py-2 text-sm border border-stone-200 rounded-xl
              focus:outline-none focus:ring-2 focus:ring-amber-300 placeholder:text-stone-300"
          />
        </div>

        {/* Sort */}
        <div className="relative">
          <SortAsc size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortOption)}
            className="pl-8 pr-3 py-2 text-sm border border-stone-200 rounded-xl bg-white
              focus:outline-none focus:ring-2 focus:ring-amber-300 appearance-none"
          >
            {(Object.entries(SORT_LABELS) as [SortOption, string][]).map(([v, l]) => (
              <option key={v} value={v}>{l}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Category filter pills */}
      <div className="flex flex-wrap gap-2">
        <CategoryPill
          label="All"
          active={categoryFilter === null}
          onClick={() => setCategoryFilter(null)}
        />
        {CATEGORIES.map((cat) => (
          <CategoryPill
            key={cat}
            label={CATEGORY_LABELS[cat]}
            active={categoryFilter === cat}
            onClick={() => setCategoryFilter(cat)}
          />
        ))}
      </div>

      {/* Empty state */}
      {rows.length === 0 && (
        <div className="text-center py-20 text-stone-400">
          <p className="text-4xl mb-3">🍪</p>
          <p className="font-medium">Your recipe box is empty</p>
          <p className="text-sm mt-1">Heart a recipe from the Find tab to save it here.</p>
        </div>
      )}

      {/* Recipe list */}
      <div className="space-y-3">
        {rows.map((row) => (
          <RecipeBoxItem
            key={row.url}
            row={row}
            onDeleted={handleDeleted}
            onNotesUpdated={handleNotesUpdated}
          />
        ))}
      </div>
    </div>
  )
}

function CategoryPill({
  label,
  active,
  onClick,
}: {
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`text-xs px-3 py-1 rounded-full font-medium transition-colors
        ${active
          ? 'bg-amber-500 text-white'
          : 'bg-stone-100 text-stone-600 hover:bg-amber-100 hover:text-amber-800'}`}
    >
      {label}
    </button>
  )
}

function RecipeBoxItem({
  row,
  onDeleted,
  onNotesUpdated,
}: {
  row: RecipeBoxRow
  onDeleted: (url: string) => void
  onNotesUpdated: (url: string, notes: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [editingNotes, setEditingNotes] = useState(false)
  const [notes, setNotes] = useState(row.notes ?? '')
  const [deleting, setDeleting] = useState(false)

  const scored: ScoredRecipe | null = (() => {
    try { return JSON.parse(row.recipe_json) } catch { return null }
  })()

  const savedDate = new Date(row.liked_at).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
  })

  async function handleDelete() {
    if (!confirm('Remove this recipe from your box?')) return
    setDeleting(true)
    await deleteRecipe(row.url)
    onDeleted(row.url)
  }

  async function handleSaveNotes() {
    await updateNotes(row.url, notes)
    onNotesUpdated(row.url, notes)
    setEditingNotes(false)
  }

  return (
    <div className="bg-white rounded-2xl border border-amber-100 shadow-sm overflow-hidden">
      {/* Condensed header */}
      <div className="flex items-center gap-3 px-4 py-3">
        <div className="flex-1 min-w-0">
          <a
            href={row.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-semibold text-stone-800 hover:text-amber-700 transition-colors truncate block"
          >
            {row.title}
          </a>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-medium">
              {CATEGORY_LABELS[row.category as Category] ?? row.category}
            </span>
            <span className="text-xs text-stone-400">{savedDate}</span>
          </div>
        </div>

        {/* Composite score badge */}
        {scored && (
          <div className={`text-sm font-bold tabular-nums px-2 py-1 rounded-lg
            ${scored.composite_score >= 70 ? 'bg-emerald-50 text-emerald-700' :
              scored.composite_score >= 45 ? 'bg-amber-50 text-amber-700' :
              'bg-rose-50 text-rose-700'}`}>
            {scored.composite_score.toFixed(0)}
          </div>
        )}

        <div className="flex items-center gap-1">
          <button
            onClick={() => { setEditingNotes(true); setExpanded(true) }}
            className="p-1.5 text-stone-400 hover:text-amber-600 rounded-lg hover:bg-amber-50 transition-colors"
            title="Edit notes"
          >
            <Pencil size={14} />
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="p-1.5 text-stone-400 hover:text-rose-500 rounded-lg hover:bg-rose-50 transition-colors"
            title="Remove from box"
          >
            <Trash2 size={14} />
          </button>
          <button
            onClick={() => setExpanded((x) => !x)}
            className="p-1.5 text-stone-400 hover:text-stone-600 rounded-lg hover:bg-stone-50 transition-colors"
            title={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Notes preview (condensed) */}
      {!expanded && notes && (
        <div className="px-4 pb-3">
          <p className="text-xs text-stone-500 italic truncate">{notes}</p>
        </div>
      )}

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-stone-100 px-4 py-4 space-y-4">
          {/* Notes editor */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-stone-500">Notes</p>
            {editingNotes ? (
              <div className="space-y-2">
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="What do you like? Any improvements you'd suggest?"
                  className="w-full text-sm border border-stone-200 rounded-xl px-3 py-2 resize-none h-20
                    focus:outline-none focus:ring-2 focus:ring-amber-300 placeholder:text-stone-300"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveNotes}
                    className="flex items-center gap-1 bg-amber-500 hover:bg-amber-600 text-white
                      text-xs font-medium px-3 py-1.5 rounded-lg transition-colors"
                  >
                    <Check size={12} /> Save
                  </button>
                  <button
                    onClick={() => { setNotes(row.notes ?? ''); setEditingNotes(false) }}
                    className="text-xs text-stone-500 hover:text-stone-700 px-3 py-1.5"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <p
                onClick={() => setEditingNotes(true)}
                className="text-sm text-stone-600 italic cursor-pointer hover:text-stone-800 min-h-[20px]"
              >
                {notes || <span className="text-stone-300">Click to add notes…</span>}
              </p>
            )}
          </div>

          {/* Full scorecard */}
          {scored && (
            <div className="space-y-3">
              <div className="flex items-center gap-4 py-2 px-3 bg-stone-50 rounded-xl">
                <CompositeScore score={scored.composite_score} />
                <div className="flex-1">
                  <p className="text-xs text-stone-500 font-medium mb-1">Composite Score</p>
                  <div className="h-3 bg-stone-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full score-bar-fill ${
                        scored.composite_score >= 70 ? 'bg-emerald-400' :
                        scored.composite_score >= 45 ? 'bg-amber-400' : 'bg-rose-400'
                      }`}
                      style={{ width: `${scored.composite_score}%` }}
                    />
                  </div>
                </div>
              </div>

              {scored.criteria.map((c) => (
                <ScoreBar
                  key={c.name}
                  score={c.score}
                  label={c.name}
                  weight={c.weight}
                  details={c.details}
                />
              ))}

              {scored.explanation && (
                <p className="text-xs text-stone-600 leading-relaxed border-t border-stone-100 pt-3">
                  {scored.explanation}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
