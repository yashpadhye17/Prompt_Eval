export interface Stat {
  n: number
  mean: number | null
  stdev: number | null
  ci95: number | null
  low: number | null
  high: number | null
}

export type Metrics = Record<string, Stat>

export interface Operational {
  generations: number
  errors: number
  truncated: number
  retries: number
  prompt_tokens: number
  completion_tokens: number
  reasoning_words: number
  cost_usd: number | null
  judge_cost_usd: number
  latency_ms: Stat
  ttft_ms: Stat
  words: Stat
}

export interface ModelEntry {
  model: string
  metrics: Metrics
  operational: Operational
  self_graded: boolean
  score: number | null
}

export interface TechniqueEntry {
  technique: string
  metrics: Metrics
  operational: Operational
  score: number | null
}

export interface Heatmap {
  techniques: string[]
  models: string[]
  grid: (number | null)[][]
}

export interface Cell {
  prompt_id: string
  model: string
  technique: string
  query_id: string
  repeats: number
  metrics: Metrics
  composite: Stat
  generation_ids: number[]
  consistency: { tfidf_cosine?: number; rouge_l?: number; composite_stdev?: number }
}

export interface FailureExample {
  raw: string
  classification: string
  evidence: string | null
}

export interface Failure {
  generation_id: number
  prompt_id: string
  technique: string
  model: string
  repeat_index: number
  severity: number
  contradictions: number | null
  unsupported: number | null
  word_count: number | null
  word_limit: number | null
  examples: FailureExample[]
  missing_sections: string[]
}

export interface RunSummary {
  totals: {
    generations: number
    ok: number
    failed: number
    judged: number
    self_graded: number
    models: string[]
    techniques: string[]
    prompts: string[]
  }
  overall: Metrics
  operational: Operational
  models: ModelEntry[]
  techniques: TechniqueEntry[]
  queries: { query_id: string; metrics: Metrics; score: number | null }[]
  heatmap: Heatmap
  cells: Cell[]
  consistency: {
    tfidf_cosine: Stat
    rouge_l: Stat
    by_model: { model: string; tfidf_cosine: Stat; rouge_l: Stat }[]
  }
  failures: Failure[]
  weights: Record<string, number>
  metric_labels: Record<string, string>
  run: {
    id: string
    status: string
    created_at: string
    repeats: number
    models: string[]
    prompt_ids: string[]
    total_tasks: number
    completed_tasks: number
    failed_tasks: number
    judge_model: string | null
    config: Record<string, unknown>
  }
  running: boolean
  has_report: boolean
}

export interface RunListItem {
  id: string
  status: string
  created_at: string
  models: string[]
  prompt_ids: string[]
  repeats: number
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  running: boolean
  has_report: boolean
}

export interface PromptInfo {
  prompt_id: string
  query_id: string
  technique: string
  variant: string
  word_limit: number | null
  required_sections: number
  required_tokens: string[]
}

export interface ConfigResponse {
  models: {
    id: string
    label: string
    family: string
    pricing: { input_per_mtok: number | null; output_per_mtok: number | null }
  }[]
  prompts: PromptInfo[]
  defaults: {
    repeats: number
    judge_enabled: boolean
    judge_model: string
    judge_repeats_per_cell: number
    temperature: number
    max_output_tokens: number
    concurrency: number
  }
  judge_options: string[]
  presets: RunPreset[]
  budget: BudgetSnapshot
  weights: Record<string, number>
  rubric: {
    scale: { min: number; max: number }
    dimensions: { id: string; name: string; weight: number; description: string }[]
  }
}

export interface RunPreset {
  id: string
  label: string
  description: string
  models?: string[]
  prompt_ids?: string[]
  repeats?: number
  judge_enabled?: boolean
  judge_model?: string
  judge_repeats_per_cell?: number
}

export interface BudgetModel {
  model: string
  limit: number | null
  used: number
  remaining: number | null
  usable?: number | null
  projected?: number
  fits?: boolean
  shortfall?: number
  calls: number
  fraction_used: number | null
  by_role: Record<string, { tokens: number; calls: number }>
}

export interface BudgetSnapshot {
  enabled: boolean
  day: string
  default_limit?: number | null
  models: BudgetModel[]
  fits?: boolean
  blocking_models?: string[]
}

export interface Estimate {
  generations: number
  judge_calls: number
  judge_repeats_per_cell?: number
  judge_sampled?: boolean
  judge_model?: string | null
  budget?: BudgetSnapshot
  per_model: {
    model: string
    generations: number
    prompt_tokens: number
    completion_tokens: number
    completion_estimate_source: string
    cost_usd: number | null
  }[]
  generation_cost_usd: number
  judge_cost_usd: number
  total_cost_usd: number
  unpriced_models: string[]
  note: string
}

export interface NumberItem {
  raw: string
  value: number
  dimension: string
  subject: string | null
  hi: number | null
  start: number
  end: number
  classification: string
  evidence: string | null
}

export interface GenerationDetail {
  id: number
  prompt_id: string
  technique: string
  model: string
  repeat_index: number
  status: string
  error: string | null
  output: string
  reasoning: string
  prompt_text: string
  word_count: number | null
  latency_ms: number | null
  ttft_ms: number | null
  prompt_tokens: number | null
  completion_tokens: number | null
  cost_usd: number | null
  truncated: number
  finish_reason: string | null
  fact_recall: number | null
  numeric_grounding: number | null
  contradiction_free: number | null
  structural_compliance: number | null
  length_compliance: number | null
  required_tokens: number | null
  format_clean: number | null
  judge_overall: number | null
  self_graded: number | null
  judge_error: string | null
  judge_scores: Record<string, number>
  judge_justifications: Record<string, string>
  details: {
    facts?: { found: string[]; missing: string[]; total: number }
    numbers?: {
      total: number
      supported: number
      derived: number
      unsupported: number
      contradicting: number
      items: NumberItem[]
    }
    structure?: { required: number; found: number; order_ok: boolean; missing: string[] }
    length?: { word_limit: number | null; word_count: number; ratio: number | null }
    required_tokens?: { required: string[]; present: string[]; missing: string[] }
    format?: { problems: string[] }
  }
  prompt_meta: { key_facts_raw?: string; required_sections?: { number: number; title: string }[] }
}

export interface ProgressEvent {
  type: string
  run_id?: string
  completed?: number
  total?: number
  status?: string
  model?: string
  prompt_id?: string
  technique?: string
  repeat_index?: number
  judge_overall?: number | null
  scores?: Record<string, number>
  word_count?: number
  latency_ms?: number
  error?: string
  phase?: string
  ts?: string
}
