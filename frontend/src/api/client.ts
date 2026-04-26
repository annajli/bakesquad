import type { RecipeBoxRow, ScoredRecipe, UserPrefs } from '../types/api'

const BASE = '/api'

// ---------------------------------------------------------------------------
// SSE streaming helper — works with POST bodies (EventSource only handles GET)
// ---------------------------------------------------------------------------

export async function* streamPost<T>(
  path: string,
  body: unknown,
): AsyncGenerator<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok || !res.body) {
    throw new Error(`API error ${res.status}: ${await res.text()}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE lines: "data: {...}\n\n"
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''

    for (const part of parts) {
      const line = part.trim()
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6)) as T
        } catch {
          // malformed — skip
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Recipe box CRUD
// ---------------------------------------------------------------------------

export async function saveRecipe(
  scored: ScoredRecipe,
  notes = '',
): Promise<void> {
  await fetch(`${BASE}/recipe-box`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url: scored.recipe.url,
      title: scored.recipe.title,
      category: scored.recipe.category,
      scored_recipe: scored,
      notes,
    }),
  })
}

export async function updateNotes(url: string, notes: string): Promise<void> {
  await fetch(`${BASE}/recipe-box`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, notes }),
  })
}

export async function deleteRecipe(url: string): Promise<void> {
  await fetch(`${BASE}/recipe-box`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
}

export async function getRecipeBox(params?: {
  sort?: string
  category?: string
  search?: string
}): Promise<RecipeBoxRow[]> {
  const qs = new URLSearchParams()
  if (params?.sort)     qs.set('sort', params.sort)
  if (params?.category) qs.set('category', params.category)
  if (params?.search)   qs.set('search', params.search)
  const res = await fetch(`${BASE}/recipe-box?${qs}`)
  const data = await res.json()
  return data.recipes as RecipeBoxRow[]
}

// ---------------------------------------------------------------------------
// Prefs
// ---------------------------------------------------------------------------

export async function getPrefs(): Promise<UserPrefs> {
  const res = await fetch(`${BASE}/prefs`)
  return res.json()
}

export async function updatePrefs(patch: Partial<UserPrefs>): Promise<void> {
  await fetch(`${BASE}/prefs`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}
