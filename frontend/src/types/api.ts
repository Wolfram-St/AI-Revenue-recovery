export interface HealthResponse {
  status: string;
  service: string;
}

export interface ActionDistribution {
  action: string;
  count: number;
}

export interface DashboardResponse {
  total_cases: number;
  revenue_at_risk_inr: number;
  estimated_recoverable_value_inr: number;
  candidate_count: number;
  stop_count: number;
  noop_count: number;
  action_distribution: ActionDistribution[];
  demo_mode: boolean;
}

export interface CaseSummary {
  attempt_id: string;
  payment_id: string;
  customer_id: string;
  amount_inr: number;
  failure_category: string;
  scoring_recommendation: string;
  authorized_action: string;
  expected_recovery_value_inr: number;
  recovery_probability: number;
  is_stop: boolean;
}

export interface CaseListResponse {
  cases: CaseSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface EvaluatedRule {
  rule_id: string;
  matched: boolean;
}

export interface CaseDetail {
  attempt_id: string;
  payment_id: string;
  customer_id: string;
  amount_inr: number;
  failure_category: string;
  recovery_probability: number;
  expected_recovery_value_inr: number;
  scoring_recommendation: string;
  authorized_action: string;
  authorization_reason: string;
  matched_rule_id: string | null;
  matched_rule_name: string | null;
  is_stop: boolean;
  evaluated_rules: EvaluatedRule[];
  model_contract: string;
}

export interface CaseDetailResponse {
  case: CaseDetail;
  audit_history: Record<string, unknown>[];
}

export interface CandidateAction {
  arm: string;
  probability: number;
  expected_recovery_value_inr: number;
}

export interface PolicyInfo {
  decision: string;
  authorized_action: string;
  reason: string;
  matched_rule_id: string | null;
  matched_rule_name: string | null;
  is_stop: boolean;
}

export interface AnalysisResponse {
  attempt_id: string;
  amount_inr: number;
  failure_category: string;
  recovery_probability: number;
  scoring_recommendation: string;
  expected_recovery_value_inr: number;
  worth_intervening: boolean;
  candidate_actions: CandidateAction[];
  policy: PolicyInfo;
  audit_context: {
    evaluated_rules: EvaluatedRule[];
    model_contract: string;
    authorization_reason: string;
  };
}

export interface PortfolioRequest {
  budget_inr: number;
  human_review_capacity: number;
}

export interface PortfolioEntry {
  attempt_id: string;
  payment_id: string;
  optimizer_recommendation: string;
  authorized_action: string;
  authorization_reason: string;
  matched_rule_id: string | null;
  policy_overrode_recommendation: boolean;
  selected_net_incremental_value_inr: number | null;
  selected_action_cost_inr: number | null;
  no_intervention_reason: string | null;
}

export interface PortfolioSummary {
  total_rows: number;
  optimizer_allocated_count: number;
  no_intervention_count: number;
  budget_limit_inr: number | null;
  budget_allocated_inr: number;
  budget_remaining_inr: number | null;
  human_review_capacity_limit: number | null;
  human_review_allocated_count: number;
  post_policy_net_authorized_count: number;
  total_policy_overrides: number;
  optimizer_objective_value_inr: number;
  optimizer_status: string;
  action_recommendation_counts: Record<string, number>;
  action_authorized_counts: Record<string, number>;
}

export interface PortfolioResponse {
  solver: string;
  summary: PortfolioSummary;
  entries: PortfolioEntry[];
  metadata: Record<string, unknown>;
}

export interface AuditEntry {
  event_type: string;
  actor_type: string;
  recovery_case_id: string;
  action: string;
  decision_reason: string;
  event_payload: Record<string, unknown>;
}

export interface AuditResponse {
  entries: AuditEntry[];
  total: number;
  page: number;
  page_size: number;
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
  };
}

// ── Recovery Agent ─────────────────────────────────────────────────────

export interface RecoveryActionDist {
  action: string;
  count: number;
}

export interface RecoveryCategoryDist {
  category: string;
  count: number;
}

export interface RecoveryAuditSummary {
  total_entries: number;
  total_amount_inr: number;
  total_recovered_inr: number;
  recovery_rate_pct: number;
  success_rate_pct: number;
  action_distribution: RecoveryActionDist[];
  category_distribution: RecoveryCategoryDist[];
}

export interface RecoveryStatusResponse {
  mode: string;
  total_processed: number;
  total_amount_inr: number;
  recovered_count: number;
  recovered_inr: number;
  recovery_rate_pct: number;
  escalated_count: number;
  stopped_count: number;
  budget_used_inr: number;
  budget_remaining_inr: number;
  audit_summary: RecoveryAuditSummary;
}

export interface RecoveryAuditEntry {
  entry_id: string;
  payment_id: string;
  subscription_id: string;
  amount_paise: number;
  failure_category: string;
  classification_confidence: number;
  classification_reason: string;
  raw_error_code: string;
  raw_error_reason: string;
  action_type: string;
  strategy_reason: string;
  retry_count: number;
  estimated_cost_paise: number;
  execution_success: boolean;
  new_order_id: string;
  execution_message: string;
  execution_error: string;
  cost_paise: number;
  created_at: number;
  cycle_duration_ms: number;
}

export interface RecoverySimulateRequest {
  count?: number;
  budget_limit_inr?: number;
  max_retries?: number;
}
