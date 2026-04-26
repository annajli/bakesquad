interface Props {
  score: number   // 0–100
  label?: string
  weight?: number
  details?: string
}

function barColor(score: number) {
  if (score >= 70) return 'bg-emerald-400'
  if (score >= 45) return 'bg-amber-400'
  return 'bg-rose-400'
}

export function ScoreBar({ score, label, weight, details }: Props) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs text-stone-600">
        <span className="font-medium">
          {label}
          {weight !== undefined && (
            <span className="text-stone-400 ml-1 font-normal">w={weight.toFixed(2)}</span>
          )}
        </span>
        <span className="font-semibold tabular-nums">{score.toFixed(0)}</span>
      </div>
      <div className="h-2 bg-stone-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full score-bar-fill ${barColor(score)}`}
          style={{ width: `${score}%` }}
        />
      </div>
      {details && (
        <p className="text-[10px] text-stone-400 leading-tight">{details}</p>
      )}
    </div>
  )
}

export function CompositeScore({ score }: { score: number }) {
  const color =
    score >= 70 ? 'text-emerald-600' :
    score >= 45 ? 'text-amber-600' :
    'text-rose-600'

  return (
    <div className="flex flex-col items-center">
      <span className={`text-4xl font-bold tabular-nums ${color}`}>
        {score.toFixed(0)}
      </span>
      <span className="text-xs text-stone-400 font-medium">/ 100</span>
    </div>
  )
}
