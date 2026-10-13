export interface Effect {
  estimate: number
  se: number
  ci_low: number
  ci_high: number
  n: number
  method: string
  p_value: number | null
  variance_reduction?: number
  rho: number | null
  extra?: Record<string, unknown>
}

export interface LeaderRow {
  agent: string
  mean: number
  sd_across_tasks: number
  is_baseline: boolean
  ci_low: number
  ci_high: number
  se: number
  significant: boolean
  ci_note?: string
  meta_cost_per_task_tokens?: number
  meta_latency_s_per_task?: number
  meta_true_mean?: number
  meta_notes?: string
}

export interface VarianceBlock {
  n_tasks: number
  n_seeds: number
  n_judges: number
  n_cells: number
  n_tied_cells: number
  tie_rate: number
  naive_se: number
  unidentifiable: string[]
  mom: ComponentFit
  reml: ComponentFit
}

export interface ComponentFit {
  components: { task: number; seed: number; judge: number; eps: number }
  se: number
  se_terms: Record<string, number>
  dominant_term: string
  se_inflation_vs_naive: number
}

export interface Comparison {
  arm: string
  baseline: string
  arm_mean: number
  baseline_mean: number
  estimators: {
    naive: Effect
    cluster: Effect
    crossed: Effect
    bootstrap: Effect
  }
  variance: VarianceBlock
  se_inflation_naive_to_crossed: number
  ranking: { mean: number; crossed: Effect; significant_vs_baseline: boolean }
}

export interface JudgeCard {
  name: string
  n_judged: number
  observed_positive_rate: number
  bias_corrected_rate: number
  raw_overstates_by: number
  raw_agreement_with_truth: number
  sensitivity: number
  specificity: number
  bias: number
  reliability: number
  cohen_kappa_vs_truth: number
  gwet_ac1_vs_truth: number
  sensitivity_error: number
  specificity_error: number
  planted: {
    sensitivity: number
    specificity: number
    position_bias: number
    verbosity_bias: number
  }
  interval?: Record<string, number>
}

export interface Report {
  meta: {
    generated: string
    elapsed_s: number
    baseline: string
    n_agents: number
    n_tasks: number
    n_seeds: number
    n_judges: number
    n_cells: number
    target_delta: number
    spec: Record<string, unknown>
  }
  warnings: string[]
  leaderboard: LeaderRow[]
  pareto: { agent: string; mean: number; cost: number; latency: number; sd: number }[]
  comparisons: Comparison[]
  judges: {
    n_items: number
    n_judges: number
    latent_positive_rate: number
    observed_positive_rate: number
    true_positive_rate: number
    dawid_skene: Record<string, unknown>
    judges: Record<string, JudgeCard>
  }
  rank_stability: Record<string, unknown>
  replication_curve: {
    reps: number
    p_top1_unchanged: number
    expected_margin_over_runner_up: number
    top_agent: string
  }[]
  planner: {
    design: Record<string, number>
    cost: number
    costs: Record<string, number>
    unidentifiable: string[]
    min_detectable_delta: Record<string, number>
    table: {
      n_tasks: number
      n_seeds: number
      n_judges: number
      n_cells: number
      cost: number
      se: number
      ci_half_width: number
      power: number
    }[]
    recommended: {
      n_tasks: number
      n_seeds: number
      n_judges: number
      n_cells: number
      cost: number
      power: number
      ci_half_width: number
      cost_ratio_vs_current: number
    } | null
    current_power_at_target: number
    reference: { arm?: string; baseline?: string; observed_effect?: number; note: string }
  }
}
