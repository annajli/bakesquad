// TypeScript mirrors of the Python Pydantic models returned by the backend.

export type Category =
  | 'cookie'
  | 'quick_bread'
  | 'cake'
  | 'yeasted_bread'
  | 'pastry'
  | 'brownie'
  | 'other'

export interface CriterionScore {
  name: string
  score: number
  weight: number
  details: string
}

export interface RatioResult {
  url: string
  category: Category
  flour_type: string
  modifiers: string[]
  liquid_to_flour: number | null
  fat_to_flour: number | null
  sugar_to_flour: number | null
  leavening_to_flour: number | null
  fat_type: 'oil' | 'butter' | 'mixed' | 'none' | null
  has_banana: boolean
  brown_to_white_sugar: number | null
  butter_to_flour: number | null
  has_extra_yolks: boolean
  leavening_type: 'soda' | 'powder' | 'both' | 'none' | null
  has_chocolate: boolean
  has_binding_agent: boolean
  flour_grams: number
  from_cache: boolean
}

export interface ParsedRecipe {
  title: string
  url: string
  category: Category
  flour_type: string
  modifiers: string[]
  yield_description: string
  instruction_count: number
  has_chocolate: boolean
  technique_signals: string[]
  technique_notes: string
}

export interface ScoredRecipe {
  recipe: ParsedRecipe
  ratios: RatioResult
  criteria: CriterionScore[]
  composite_score: number
  constraint_violations: string[]
  explanation: string
  rank: number
  technique_note_delta: number | null
  accessibility_score: number | null
}

// SSE event payloads from POST /api/search
export type SearchSseEvent =
  | { step: 'query_plan';     messages: string[] }
  | { step: 'query_plan_done'; data: { category: Category; constraints: string[]; preferences: string[] } }
  | { step: 'search';         messages: string[] }
  | { step: 'search_done';    data: { count: number } }
  | { step: 'fetch';          messages: string[] }
  | { step: 'fetch_done';     data: { fetched: number; attempted: number } }
  | { step: 'parse';          messages: string[] }
  | { step: 'parse_done';     data: { parsed: number; failed: number } }
  | { step: 'ratio';          messages: string[] }
  | { step: 'ratio_done';     data: { cache_hits: number } }
  | { step: 'score';          messages: string[] }
  | { step: 'done';           session_id: string; results: ScoredRecipe[]; similar_saved: SimilarRecipe[] }
  | { step: 'error';          message: string }
  | { step: 'thinking';       message: string }
  | { step: 're_search_start'; query: string }

// SSE event payloads from POST /api/score-url
export type ScoreUrlSseEvent =
  | { step: 'fetch';   message: string }
  | { step: 'parse';   message: string }
  | { step: 'ratio';   message: string }
  | { step: 'score';   message: string }
  | { step: 'done';    result: ScoredRecipe }
  | { step: 'error';   message: string }

// SSE event payloads from POST /api/chat
export type ChatSseEvent =
  | { step: 'thinking';       message: string }
  | { step: 'done'; type: 're_filter'; results: ScoredRecipe[] }
  | { step: 'done'; type: 'factual';   answer: string }
  | { step: 're_search_start'; query: string }
  | SearchSseEvent  // re_search tunnels through search stream

export interface SimilarRecipe {
  url: string
  title: string
  category: Category
  similarity: number
}

// Recipe box row from GET /api/recipe-box
export interface RecipeBoxRow {
  url: string
  title: string
  category: Category
  recipe_json: string   // JSON string of ScoredRecipe
  rating: number
  notes: string
  liked_at: string
  user_rating: number | null
  tried_date: string | null
}

// Prefs from GET /api/prefs
export interface UserPrefs {
  dietary_restrictions: string[]
  use_feedback_prefs: boolean
  prefer_accessibility: number
}

export type DietaryOption = 'gluten_free' | 'vegan' | 'dairy_free' | 'nut_free' | 'paleo' | 'keto'

export const DIETARY_LABELS: Record<DietaryOption, string> = {
  gluten_free:  'Gluten-Free',
  vegan:        'Vegan',
  dairy_free:   'Dairy-Free',
  nut_free:     'Nut-Free',
  paleo:        'Paleo',
  keto:         'Keto',
}

export const CATEGORY_LABELS: Record<Category, string> = {
  cookie:       'Cookie',
  quick_bread:  'Quick Bread',
  cake:         'Cake',
  yeasted_bread:'Yeasted Bread',
  pastry:       'Pastry',
  brownie:      'Brownie',
  other:        'Other',
}
