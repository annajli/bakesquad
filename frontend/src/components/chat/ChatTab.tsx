import { useEffect, useRef, useState } from 'react'
import { useAutoResize } from '../../hooks/useAutoResize'
import { streamPost } from '../../api/client'
import type {
  ChatSseEvent,
  ScoreUrlSseEvent,
  SearchSseEvent,
  ScoredRecipe,
} from '../../types/api'
import { ChatInput } from './ChatInput'
import type { PipelineStep } from './ProgressTimeline'
import { ProgressTimeline } from './ProgressTimeline'
import { ScoreCard } from './ScoreCard'

interface ResultGroup {
  query: string
  results: ScoredRecipe[]
  type: 'search' | 'url' | 'filter' | 'factual'
  answer?: string          // factual follow-up answer
}

interface Props {
  recency: string | null
}

export function ChatTab({ recency }: Props) {
  const [groups, setGroups] = useState<ResultGroup[]>([])
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState<PipelineStep>('query_plan')
  const [puns, setPuns] = useState<string[]>([])
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [followUp, setFollowUp] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const followUpResize = useAutoResize()

  // Auto-scroll to newest results
  useEffect(() => {
    if (!loading) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [groups, loading])

  async function handleSearch(query: string) {
    setLoading(true)
    setErrorMsg(null)
    setStep('query_plan')
    setPuns([])
    try {
      const stream = streamPost<SearchSseEvent>('/search', {
        query,
        recency: recency ?? undefined,
        session_id: sessionId ?? undefined,
      })

      for await (const event of stream) {
        if ('messages' in event) {
          setPuns(event.messages)
          setStep(event.step as PipelineStep)
        } else if (event.step === 'query_plan_done') {
          setStep('search')
        } else if (event.step === 'search_done') {
          setStep('fetch')
        } else if (event.step === 'fetch_done') {
          setStep('parse')
        } else if (event.step === 'parse_done') {
          setStep('ratio')
        } else if (event.step === 'ratio_done') {
          setStep('score')
        } else if (event.step === 'done') {
          setSessionId(event.session_id)
          setStep('done')
          setGroups((prev) => [
            ...prev,
            { query, results: event.results, type: 'search' },
          ])
        } else if (event.step === 'error') {
          setErrorMsg(event.message)
          setStep('error')
        }
      }
    } catch (e) {
      setErrorMsg(String(e))
      setStep('error')
    } finally {
      setLoading(false)
    }
  }

  async function handleScoreUrl(url: string) {
    setLoading(true)
    setErrorMsg(null)
    setStep('query_plan')
    setPuns([])

    try {
      const stream = streamPost<ScoreUrlSseEvent>('/score-url', { url })

      for await (const event of stream) {
        if (event.step === 'fetch')  { setStep('fetch');  setPuns([event.message]) }
        if (event.step === 'parse')  { setStep('parse');  setPuns([event.message]) }
        if (event.step === 'ratio')  { setStep('ratio');  setPuns([event.message]) }
        if (event.step === 'score')  { setStep('score');  setPuns([event.message]) }
        if (event.step === 'done') {
          setStep('done')
          setGroups((prev) => [
            ...prev,
            { query: url, results: [event.result], type: 'url' },
          ])
        }
        if (event.step === 'error') {
          setErrorMsg(event.message)
          setStep('error')
        }
      }
    } catch (e) {
      setErrorMsg(String(e))
      setStep('error')
    } finally {
      setLoading(false)
    }
  }

  async function handleFollowUp(e: React.SyntheticEvent) {
    e.preventDefault()
    if (!sessionId || !followUp.trim() || loading) return
    const msg = followUp.trim()
    setFollowUp('')
    setLoading(true)
    setStep('query_plan')
    setPuns([])

    try {
      const stream = streamPost<ChatSseEvent>('/chat', {
        session_id: sessionId,
        message: msg,
        recency: recency ?? undefined,
      })

      for await (const event of stream) {
        if (event.step === 'thinking') {
          setPuns([event.message])
        } else if (event.step === 'done') {
          setStep('done')
          if ('results' in event) {
            setGroups((prev) => [
              ...prev,
              { query: msg, results: event.results, type: 'filter' },
            ])
          } else if ('answer' in event) {
            setGroups((prev) => [
              ...prev,
              { query: msg, results: [], type: 'factual', answer: event.answer },
            ])
          }
        } else if (event.step === 'error') {
          setErrorMsg(event.message)
          setStep('error')
        }
      }
    } catch (e) {
      setErrorMsg(String(e))
      setStep('error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Input */}
      <ChatInput
        onSearch={handleSearch}
        onScoreUrl={handleScoreUrl}
        disabled={loading}
      />

      {/* Loading state */}
      {loading && <ProgressTimeline currentStep={step} puns={puns} />}

      {/* Error */}
      {errorMsg && !loading && (
        <div className="bg-rose-50 border border-rose-200 rounded-xl px-4 py-3 text-sm text-rose-700">
          {errorMsg}
        </div>
      )}

      {/* Result groups (oldest at top, newest at bottom) */}
      <div className="space-y-10">
        {groups.map((group, gi) => (
          <div key={gi} className="space-y-4">
            {/* Query label */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-stone-400 bg-stone-100 rounded-full px-3 py-1">
                {group.type === 'url' ? '🔗 Scored URL' :
                 group.type === 'filter' ? '🔍 Filtered' :
                 group.type === 'factual' ? '💬' : '🔍'}
              </span>
              <span className="text-sm font-medium text-stone-600 truncate">
                {group.query}
              </span>
            </div>

            {/* Factual answer */}
            {group.type === 'factual' && group.answer && (
              <div className="bg-amber-50 border border-amber-100 rounded-xl px-4 py-3 text-sm text-stone-700 leading-relaxed">
                {group.answer}
              </div>
            )}

            {/* Score cards */}
            {group.results.length > 0 && (
              <div className={`grid gap-4 ${group.results.length >= 2 ? 'md:grid-cols-2' : ''} ${group.results.length === 3 ? 'lg:grid-cols-3' : ''}`}>
                {group.results.map((scored, ri) => (
                  <ScoreCard
                    key={scored.recipe.url}
                    scored={scored}
                    rank={ri + 1}
                  />
                ))}
              </div>
            )}

            {group.type === 'filter' && group.results.length === 0 && (
              <p className="text-sm text-stone-500">
                No recipes pass that filter from the current results.
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Follow-up input (visible once there are results) */}
      {sessionId && groups.length > 0 && (
        <div className="bg-white rounded-2xl border border-amber-100 shadow-sm p-3 space-y-2">
          <div className="flex gap-2 items-end">
            <textarea
              ref={followUpResize.ref}
              rows={1}
              value={followUp}
              onChange={(e) => { setFollowUp(e.target.value); followUpResize.resize() }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleFollowUp(e)
                  if (followUpResize.ref.current) followUpResize.ref.current.style.height = 'auto'
                }
              }}
              placeholder='Follow up — e.g. "only oil-based" or "why does oil retain moisture?"'
              disabled={loading}
              className="flex-1 text-sm border border-stone-200 rounded-xl px-4 py-2 resize-none
                leading-relaxed overflow-hidden
                focus:outline-none focus:ring-2 focus:ring-amber-300
                placeholder:text-stone-300 disabled:bg-stone-50"
            />
            <button
              type="button"
              onClick={(e) => {
                handleFollowUp(e)
                if (followUpResize.ref.current) followUpResize.ref.current.style.height = 'auto'
              }}
              disabled={loading || !followUp.trim()}
              className="bg-stone-700 hover:bg-stone-800 disabled:opacity-40 shrink-0
                text-white text-sm font-medium px-4 py-2 rounded-xl transition-colors"
            >
              Ask
            </button>
          </div>
          <p className="text-[11px] text-stone-300">Enter to submit · Shift+Enter for new line</p>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
