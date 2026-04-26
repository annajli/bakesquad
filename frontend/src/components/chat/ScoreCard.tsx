import { Heart } from 'lucide-react'
import { useState } from 'react'
import { saveRecipe } from '../../api/client'
import type { ScoredRecipe } from '../../types/api'
import { CATEGORY_LABELS } from '../../types/api'
import { NotesModal } from './NotesModal'
import { CompositeScore, ScoreBar } from './ScoreBar'

interface Props {
  scored: ScoredRecipe
  rank: number
  onSaved?: (scored: ScoredRecipe) => void
}

function Tag({ label }: { label: string }) {
  return (
    <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-medium">
      {label}
    </span>
  )
}

export function ScoreCard({ scored, rank, onSaved }: Props) {
  const [hearted, setHearted] = useState(false)
  const [saving, setSaving] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)

  const { recipe, ratios, criteria, composite_score, constraint_violations, explanation } = scored

  const tags: string[] = []
  if (ratios.fat_type === 'oil')    tags.push('Oil-Based')
  if (ratios.fat_type === 'butter') tags.push('Butter-Based')
  if (ratios.has_banana)            tags.push('Banana')
  if (ratios.has_chocolate)         tags.push('Chocolate')
  if (ratios.has_extra_yolks)       tags.push('Extra Yolks')
  if (ratios.flour_type !== 'ap' && ratios.flour_type)
    tags.push(ratios.flour_type.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()))

  async function handleSave(notes: string) {
    setSaving(true)
    await saveRecipe(scored, notes)
    setHearted(true)
    setSaving(false)
    setModalOpen(false)
    onSaved?.(scored)
  }

  return (
    <div className="relative bg-white rounded-2xl border border-amber-100 shadow-sm hover:shadow-md transition-shadow p-5 flex flex-col gap-4">
      {/* Rank badge */}
      <div className="absolute top-4 left-4 w-7 h-7 rounded-full bg-amber-100 text-amber-800 text-xs font-bold flex items-center justify-center">
        #{rank}
      </div>

      {/* Heart button */}
      <button
        onClick={() => !hearted && setModalOpen(true)}
        disabled={hearted}
        className={`absolute top-4 right-4 p-1.5 rounded-full transition-colors
          ${hearted
            ? 'text-rose-500 bg-rose-50'
            : 'text-stone-300 hover:text-rose-400 hover:bg-rose-50'}`}
        title={hearted ? 'Saved to Recipe Box' : 'Save to Recipe Box'}
      >
        <Heart size={18} fill={hearted ? 'currentColor' : 'none'} />
      </button>

      {/* Title + category */}
      <div className="pl-8 pr-8">
        <a
          href={recipe.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-base font-semibold text-stone-800 hover:text-amber-700 transition-colors leading-snug"
        >
          {recipe.title}
        </a>
        <p className="text-xs text-stone-400 mt-0.5">
          {CATEGORY_LABELS[recipe.category]}
          {recipe.yield_description ? ` · ${recipe.yield_description}` : ''}
        </p>
      </div>

      {/* Tags */}
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pl-1">
          {tags.map((t) => <Tag key={t} label={t} />)}
        </div>
      )}

      {/* Composite score */}
      <div className="flex items-center gap-4 py-2 px-3 bg-stone-50 rounded-xl">
        <CompositeScore score={composite_score} />
        <div className="flex-1">
          <p className="text-xs text-stone-500 font-medium mb-1">Composite Score</p>
          <div className="h-3 bg-stone-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full score-bar-fill ${
                composite_score >= 70 ? 'bg-emerald-400' :
                composite_score >= 45 ? 'bg-amber-400' : 'bg-rose-400'
              }`}
              style={{ width: `${composite_score}%` }}
            />
          </div>
        </div>
      </div>

      {/* Per-criterion scores */}
      <div className="space-y-3">
        {criteria.map((c) => (
          <ScoreBar
            key={c.name}
            score={c.score}
            label={c.name}
            weight={c.weight}
            details={c.details}
          />
        ))}
      </div>

      {/* Ratios */}
      <div className="bg-amber-50 rounded-xl p-3 space-y-1">
        <p className="text-xs font-semibold text-amber-800 mb-1.5">Ratios</p>
        {ratios.liquid_to_flour   != null && <RatioRow label="liquid/flour"    value={ratios.liquid_to_flour} />}
        {ratios.fat_to_flour      != null && <RatioRow label="fat/flour"       value={ratios.fat_to_flour} />}
        {ratios.sugar_to_flour    != null && <RatioRow label="sugar/flour"     value={ratios.sugar_to_flour} />}
        {ratios.leavening_to_flour!= null && <RatioRow label="leavening/flour" value={ratios.leavening_to_flour} decimals={4} />}
        {ratios.brown_to_white_sugar != null && <RatioRow label="brown/white sugar" value={ratios.brown_to_white_sugar} />}
        {ratios.fat_type && <RatioRow label="fat source" value={ratios.fat_type} />}
        {ratios.from_cache && (
          <p className="text-[10px] text-amber-600 mt-1">from cache</p>
        )}
      </div>

      {/* Explanation */}
      {explanation && (
        <p className="text-xs text-stone-600 leading-relaxed border-t border-stone-100 pt-3">
          {explanation}
        </p>
      )}

      {/* Constraint violations */}
      {constraint_violations.length > 0 && (
        <div className="space-y-1">
          {constraint_violations.map((v) => (
            <p key={v} className="text-xs text-rose-600 bg-rose-50 rounded-lg px-3 py-1.5">
              ⚠ {v}
            </p>
          ))}
        </div>
      )}

      {modalOpen && (
        <NotesModal
          recipeTitle={recipe.title}
          onSave={handleSave}
          onCancel={() => setModalOpen(false)}
          saving={saving}
        />
      )}
    </div>
  )
}

function RatioRow({
  label,
  value,
  decimals = 3,
}: {
  label: string
  value: number | string
  decimals?: number
}) {
  const formatted = typeof value === 'number' ? value.toFixed(decimals) : value
  return (
    <div className="flex justify-between text-xs">
      <span className="text-amber-700">{label}</span>
      <span className="font-mono text-stone-700">{formatted}</span>
    </div>
  )
}
