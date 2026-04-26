import { useEffect, useState } from 'react'

export type PipelineStep =
  | 'query_plan'
  | 'search'
  | 'fetch'
  | 'parse'
  | 'ratio'
  | 'score'
  | 'done'
  | 'error'

const STEP_ORDER: PipelineStep[] = [
  'query_plan', 'search', 'fetch', 'parse', 'ratio', 'score', 'done',
]

const STEP_LABELS: Record<string, string> = {
  query_plan: 'Query',
  search:     'Search',
  fetch:      'Fetch',
  parse:      'Parse',
  ratio:      'Ratios',
  score:      'Score',
  done:       'Done',
}

const DEFAULT_PUNS = [
  'Preheating the oven...',
  'Sifting the flour...',
  'Separating the eggs...',
  'Measuring the vanilla...',
  'Folding it all together...',
]

interface Props {
  currentStep: PipelineStep
  puns: string[]
}

export function ProgressTimeline({ currentStep, puns }: Props) {
  const [punIndex, setPunIndex] = useState(0)
  const allPuns = puns.length > 0 ? puns : DEFAULT_PUNS

  // Cycle through puns every 2 seconds while in progress
  useEffect(() => {
    if (currentStep === 'done' || currentStep === 'error') return
    const id = setInterval(() => {
      setPunIndex((i) => (i + 1) % allPuns.length)
    }, 2000)
    return () => clearInterval(id)
  }, [currentStep, allPuns.length])

  const currentIdx = STEP_ORDER.indexOf(currentStep)
  const isError = currentStep === 'error'

  return (
    <div className="w-full py-6 px-4">
      {/* Baking pun */}
      <p className="text-center text-amber-700 font-medium text-sm mb-6 min-h-[20px] transition-all">
        {isError ? '❌ Something went wrong' : allPuns[punIndex]}
      </p>

      {/* Step dots */}
      <div className="flex items-center justify-center gap-0">
        {STEP_ORDER.map((step, idx) => {
          const isDone = idx < currentIdx || currentStep === 'done'
          const isCurrent = idx === currentIdx && currentStep !== 'done'

          return (
            <div key={step} className="flex items-center">
              {/* Connector line */}
              {idx > 0 && (
                <div
                  className={`h-0.5 w-8 transition-colors duration-500
                    ${isDone ? 'bg-amber-400' : 'bg-stone-200'}`}
                />
              )}

              {/* Dot */}
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`w-3 h-3 rounded-full transition-all duration-500
                    ${isDone     ? 'bg-amber-400' : ''}
                    ${isCurrent  ? 'bg-amber-500 pulse-dot' : ''}
                    ${!isDone && !isCurrent ? 'bg-stone-200' : ''}
                    ${isError && idx === currentIdx ? 'bg-red-400' : ''}`}
                />
                <span className="text-[10px] text-stone-400 w-10 text-center leading-tight">
                  {STEP_LABELS[step]}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
