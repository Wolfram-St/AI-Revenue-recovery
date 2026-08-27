"""Day 6 pre-registered optimizer justification gate + bounded recommender.

Plan Task 5 / decision D-E5. Authority discipline is UNCHANGED by this
module: AI recommends, policy authorizes. Everything here is a RECOMMENDATION
layer only -- the optimizer/recommender NEVER overrides policy and STOP
remains dominant when configured, because every candidate action still flows
through the deterministic ``decide_action`` engine (``recovery/policy.py``),
which stays the sole authorization boundary. A recommendation this module
emits describes what the revenue estimate prefers; the authorized action is
whatever the frozen business rules say, and when the two differ the
recommendation records ``policy_overrode_candidate=True``.

Honest limitation of the gate: the justification classification is ADVISORY
scaffolding. It runs in-process over a plain dict, so an in-process caller
could fabricate an evidence bundle that flips the verdict. Genuine Day 6
bundles carry a ``provenance_digest`` (sorted-json SHA256 emitted by
``ml/decision_evidence``); this module VALIDATES its presence/non-emptiness
and echoes it verbatim in every verdict, which makes non-canonical bundles
visible -- but it is NOT tamper-proof, and nothing here recomputes it.
Documentation, not code, identifies the canonical bundle.

Scope: every quantity consumed or emitted here is synthetic-world-only --
per-arm MODEL ESTIMATE probabilities of the simulated world and revenue
arithmetic over them -- supporting nothing causal about any production
system. The Day 2 baseline ``P(recovered | context)`` stays untouched.

Pre-registered thresholds (fixed before measurement, transparent
conventions -- NOT statistical tests): OPTIMIZER_JUSTIFIED iff ALL hold:
1. decision_match_rate >= 0.60;
2. relative_regret <= 0.15 (None fails with the literal reason "regret
   undefined");
3. policy_safety_probe_passed is True;
4. at least two pairwise-non-overlapping treated-arm bootstrap CI95s
   ([lo_a, hi_a] vs [lo_b, hi_b] disjoint; touching endpoints overlap).

Recommender conventions mirror Day 5 exactly: per treated arm ``a``, revenue
= (P_hat_a - P_hat_CONTROL) * amount - RETRY_INTERVENTION_COST_INR -
risk_penalty, with risk_penalty = UNKNOWN_CATEGORY_RISK_FRACTION * amount
iff failure_category == "unknown" (constants IMPORTED from
``recovery.scoring``, never restated); candidates are the FOUR TREATED ARMS
(CONTROL excluded; uniform retry-cost makes its revenue undefined/negative);
ties break to earlier ``ARM_ORDER`` precedence. Probabilities are queried
through ``predict_action_probability`` on a single-row frame COPY whose
``assigned_action`` column is OVERWRITTEN per queried arm -- the
counterfactual copy discipline of ``predict_pooled_probability``; the
caller's row is never mutated.

Documented probability-injection choice: ``decide_action`` consumes
``recovery_probability``. This module injects the TOP ARM's per-arm MODEL
ESTIMATE probability as that input (the probability of the action actually
being recommended). The alternative -- injecting the CONTROL-model baseline
probability -- is left as a future refinement; both are synthetic-world
quantities and neither weakens the policy boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    ActionModelBundle,
    predict_action_probability,
)
from ml.decision_evidence import TREATED_ARMS
from recovery.policy import (
    PolicyConfig,
    decide_action,
    load_policy_config,
)
from recovery.scoring import (
    RETRY_INTERVENTION_COST_INR,
    UNKNOWN_CATEGORY_RISK_FRACTION,
)

CLASSIFICATION_JUSTIFIED = "OPTIMIZER_JUSTIFIED"
CLASSIFICATION_NOT_YET_JUSTIFIED = "OPTIMIZER_NOT_YET_JUSTIFIED"

MATCH_RATE_THRESHOLD = 0.60
RELATIVE_REGRET_THRESHOLD = 0.15
MIN_NON_OVERLAPPING_CI_PAIRS = 2

LABEL_MODEL_ESTIMATE = "MODEL ESTIMATE"

REGRET_UNDEFINED_REASON = "regret undefined"

CRITERION_MATCH_RATE = "decision_match_rate_at_or_above_threshold"
CRITERION_RELATIVE_REGRET = "relative_regret_at_or_below_threshold"
CRITERION_POLICY_SAFETY_PROBE = "policy_safety_probe_passed"
CRITERION_CI_NON_OVERLAP = "minimum_non_overlapping_treated_arm_ci_pairs"

BOOTSTRAP_CI_KEY = "bootstrap_ci95_mean_model_revenue"

GATE_CONTEXT_COLUMNS = (
    "customer_opted_out",
    "fraud_risk",
    "failure_category",
    "attempt_number",
    "amount_inr",
)
"""Policy-gate columns NaN-guarded before authorization.

Mirrors ``simulation/treatment._reject_nan_context``: NaN/None compares
False inside rule conditions and would silently bypass STOP gating."""


class OptimizerNotJustifiedError(Exception):
    """Raised when a recommender is constructed without JUSTIFIED evidence.

    Carries the classifier's machine-readable ``reasons`` list (one string
    per failed D-E5 criterion) on ``.reasons``.
    """

    def __init__(self, reasons):
        self.reasons = [str(reason) for reason in reasons]
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class CandidateRecommendation:
    """One bounded recommendation: candidate plus policy authorization."""

    top_candidate_action: str
    incremental_revenue_estimate_inr: float
    model_label: str
    authorized_action: str
    authorization_reason: str
    policy_overrode_candidate: bool


def _finite_number(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"{label} must be a finite real number, got {value!r} of type "
            f"{type(value).__name__}"
        )
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite real number, got {value!r}")
    return number


def _validated_ci_bounds(arms: object) -> dict[str, tuple[float, float]]:
    """Validate the per-treated-arm bootstrap-CI blocks; name 'arms' loudly."""
    if not isinstance(arms, dict):
        raise ValueError(
            f"evidence field 'arms' must be a dict keyed by treated arm, got "
            f"{type(arms).__name__}"
        )
    absent = [arm for arm in TREATED_ARMS if arm not in arms]
    if absent:
        raise ValueError(
            f"evidence field 'arms' lacks treated-arm block(s) {absent}; "
            f"blocks for every arm in {list(TREATED_ARMS)} are required"
        )
    bounds: dict[str, tuple[float, float]] = {}
    for arm in TREATED_ARMS:
        entry = arms[arm]
        if not isinstance(entry, dict) or BOOTSTRAP_CI_KEY not in entry:
            raise ValueError(
                f"evidence field 'arms[{arm!r}]' must be a dict carrying the "
                f"'{BOOTSTRAP_CI_KEY}' block, got {entry!r}"
            )
        raw = entry[BOOTSTRAP_CI_KEY]
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(
                f"evidence field 'arms[{arm!r}].{BOOTSTRAP_CI_KEY}' must be a "
                f"[low, high] pair, got {raw!r}"
            )
        low = _finite_number(f"arms[{arm!r}].{BOOTSTRAP_CI_KEY} low bound", raw[0])
        high = _finite_number(f"arms[{arm!r}].{BOOTSTRAP_CI_KEY} high bound", raw[1])
        if low > high:
            raise ValueError(
                f"evidence field 'arms[{arm!r}].{BOOTSTRAP_CI_KEY}' bounds are "
                f"inverted: [{low}, {high}]"
            )
        bounds[arm] = (low, high)
    return bounds


def _non_overlapping_pair_count(
    bounds: dict[str, tuple[float, float]],
) -> int:
    """Count unordered treated-arm pairs whose CI95s are fully disjoint.

    Disjoint means one interval lies entirely above the other
    (hi_a < lo_b or hi_b < lo_a); touching endpoints count as OVERLAPPING,
    mirroring the overlap convention of ``ml/decision_evidence``.
    """
    names = list(TREATED_ARMS)
    count = 0
    for position, first in enumerate(names):
        lo_first, hi_first = bounds[first]
        for second in names[position + 1 :]:
            lo_second, hi_second = bounds[second]
            if hi_first < lo_second or hi_second < lo_first:
                count += 1
    return count


def classify_optimizer_justification(evidence: dict) -> dict:
    """Apply the pre-registered D-E5 rule to one evidence bundle.

    Validates the bundle structure (missing/malformed fields raise
    ``ValueError`` naming the offending key), then evaluates the FOUR
    criteria above and returns::

        {
            "classification": "OPTIMIZER_JUSTIFIED"
                              | "OPTIMIZER_NOT_YET_JUSTIFIED",
            "criteria": [{criterion, passed, evidence}, ...],
            "reasons": [<one string per failed criterion>],
            "provenance_digest": <echoed verbatim>,
        }

    Pure function over the dict: deterministic, draws no randomness, reads
    no wall clock, and never mutates its input. The digest echo is a
    VISIBILITY marker for non-canonical bundles, not tamper-proofing.
    """
    if not isinstance(evidence, dict):
        raise ValueError(
            f"evidence must be a dict produced by ml/decision_evidence, got "
            f"{type(evidence).__name__}"
        )
    required_keys = (
        "decision_match_rate",
        "relative_regret",
        "policy_safety_probe_passed",
        "arms",
        "provenance_digest",
    )
    missing = [key for key in required_keys if key not in evidence]
    if missing:
        raise ValueError(
            f"evidence is missing required key(s) {missing}; a Day 6 bundle "
            "must carry every one of them"
        )

    match_rate = _finite_number("decision_match_rate", evidence["decision_match_rate"])

    raw_regret = evidence["relative_regret"]
    if raw_regret is not None:
        _finite_number("relative_regret", raw_regret)

    probe_passed = evidence["policy_safety_probe_passed"]
    if type(probe_passed) is not bool:
        raise ValueError(
            f"policy_safety_probe_passed must be a bool, got {probe_passed!r} "
            f"of type {type(probe_passed).__name__}"
        )

    bounds = _validated_ci_bounds(evidence["arms"])

    digest = evidence["provenance_digest"]
    if not isinstance(digest, str) or not digest:
        raise ValueError(
            f"provenance_digest must be a non-empty string, got {digest!r}"
        )

    criteria: list[dict] = []
    reasons: list[str] = []

    match_ok = match_rate >= MATCH_RATE_THRESHOLD
    criteria.append(
        {
            "criterion": CRITERION_MATCH_RATE,
            "passed": match_ok,
            "evidence": {
                "observed": match_rate,
                "threshold": MATCH_RATE_THRESHOLD,
            },
        }
    )
    if not match_ok:
        reasons.append(
            f"decision_match_rate {match_rate:.4f} is below the "
            f"pre-registered D-E5 threshold {MATCH_RATE_THRESHOLD}"
        )

    if raw_regret is None:
        regret_ok = False
        regret_evidence = {
            "observed": None,
            "threshold": RELATIVE_REGRET_THRESHOLD,
        }
        reasons.append(REGRET_UNDEFINED_REASON)
    else:
        relative_regret = float(raw_regret)
        regret_ok = relative_regret <= RELATIVE_REGRET_THRESHOLD
        regret_evidence = {
            "observed": relative_regret,
            "threshold": RELATIVE_REGRET_THRESHOLD,
        }
        if not regret_ok:
            reasons.append(
                f"relative_regret {relative_regret:.4f} exceeds the "
                f"pre-registered D-E5 threshold {RELATIVE_REGRET_THRESHOLD}"
            )
    criteria.append(
        {
            "criterion": CRITERION_RELATIVE_REGRET,
            "passed": regret_ok,
            "evidence": regret_evidence,
        }
    )

    probe_ok = probe_passed is True
    criteria.append(
        {
            "criterion": CRITERION_POLICY_SAFETY_PROBE,
            "passed": probe_ok,
            "evidence": {"observed": probe_passed},
        }
    )
    if not probe_ok:
        reasons.append(
            "policy_safety_probe_passed is False: at least one crafted STOP "
            "context received a positive recommendation after policy gating"
        )

    pair_count = _non_overlapping_pair_count(bounds)
    overlap_ok = pair_count >= MIN_NON_OVERLAPPING_CI_PAIRS
    criteria.append(
        {
            "criterion": CRITERION_CI_NON_OVERLAP,
            "passed": overlap_ok,
            "evidence": {
                "non_overlapping_pairs": pair_count,
                "required": MIN_NON_OVERLAPPING_CI_PAIRS,
            },
        }
    )
    if not overlap_ok:
        reasons.append(
            f"only {pair_count} pairwise-non-overlapping treated-arm CI95 "
            f"pair(s) found; at least {MIN_NON_OVERLAPPING_CI_PAIRS} are "
            "required for distinctions the evidence can support"
        )

    return {
        "classification": (
            CLASSIFICATION_JUSTIFIED if not reasons
            else CLASSIFICATION_NOT_YET_JUSTIFIED
        ),
        "criteria": criteria,
        "reasons": reasons,
        "provenance_digest": digest,
    }


class BoundedRecommender:
    """Incremental-revenue candidate selector behind the D-E5 evidence gate.

    Constructible ONLY from evidence whose classification is
    OPTIMIZER_JUSTIFIED; otherwise raises ``OptimizerNotJustifiedError``
    carrying the classifier's reasons. Its output remains a CANDIDATE: the
    constructor-bound ``PolicyConfig`` (default: the frozen
    ``config/business_rules.yaml``) authorizes the final action through
    ``decide_action`` every single time -- AI recommends, policy authorizes.
    """

    def __init__(
        self,
        evidence: dict,
        calibrated_bundle: ActionModelBundle,
        policy_config: PolicyConfig | None = None,
    ):
        verdict = classify_optimizer_justification(evidence)
        if verdict["classification"] != CLASSIFICATION_JUSTIFIED:
            raise OptimizerNotJustifiedError(verdict["reasons"])
        if not isinstance(calibrated_bundle, ActionModelBundle):
            raise ValueError(
                "calibrated_bundle must be an ActionModelBundle, got "
                f"{type(calibrated_bundle).__name__}"
            )
        policy = (
            policy_config if policy_config is not None else load_policy_config()
        )
        if not isinstance(policy, PolicyConfig):
            raise ValueError(
                "policy_config must be a recovery.policy.PolicyConfig, got "
                f"{type(policy).__name__}"
            )
        self.classification = verdict
        self._bundle = calibrated_bundle
        self._policy = policy

    def recommend(self, context_row) -> CandidateRecommendation:
        """Select the top-revenue treated arm, then let policy authorize it.

        Accepts a ``pandas.Series`` or a plain dict decision-time row (it is
        copied immediately; the caller's structure is never mutated). Per-arm
        probabilities use the counterfactual overwrite pattern on a single-
        row COPY; revenue follows the Day 5 conventions documented in the
        module docstring; ties break to earlier ``ARM_ORDER`` precedence.
        The injected ``recovery_probability`` is the TOP ARM's model
        probability (documented choice; CONTROL-baseline injection is the
        documented future refinement). Missing ``amount_inr`` /
        ``failure_category`` raise ``ValueError`` naming the key, as do NaN
        values in any present policy-gate column and a top-arm probability
        outside [0, 1] (broken-bundle guard).
        """
        if isinstance(context_row, pd.Series):
            row = dict(context_row)
        elif isinstance(context_row, dict):
            row = dict(context_row)
        else:
            raise ValueError(
                f"context_row must be a pandas Series or dict, got "
                f"{type(context_row).__name__}"
            )
        for column in ("amount_inr", "failure_category"):
            if column not in row:
                raise ValueError(
                    f"context_row is missing required decision-time column "
                    f"'{column}'"
                )
        nan_offenders = [
            column
            for column in GATE_CONTEXT_COLUMNS
            if column in row and pd.isna(row[column])
        ]
        if nan_offenders:
            raise ValueError(
                f"context_row carries NaN/None in policy-gate column(s) "
                f"{nan_offenders}, which would silently compare False inside "
                "STOP-rule conditions and corrupt authorization"
            )
        amount = _finite_number("amount_inr", row["amount_inr"])
        category = row["failure_category"]
        if not isinstance(category, str) or not category:
            raise ValueError(
                f"failure_category must be a non-empty string, got {category!r}"
            )
        risk_penalty = (
            UNKNOWN_CATEGORY_RISK_FRACTION * amount
            if category == "unknown"
            else 0.0
        )

        query_frame = pd.DataFrame([row])
        probabilities = {}
        for arm in ARM_ORDER:
            counterfactual = query_frame.copy()
            counterfactual[ACTION_COLUMN] = arm
            probabilities[arm] = float(
                predict_action_probability(self._bundle, counterfactual, arm)[0]
            )

        revenues = {
            arm: (
                (probabilities[arm] - probabilities["CONTROL"]) * amount
                - RETRY_INTERVENTION_COST_INR
                - risk_penalty
            )
            for arm in TREATED_ARMS
        }
        top_arm = TREATED_ARMS[0]
        top_revenue = revenues[top_arm]
        for arm in TREATED_ARMS[1:]:
            if revenues[arm] > top_revenue:
                top_arm = arm
                top_revenue = revenues[arm]

        top_probability = probabilities[top_arm]
        if not 0.0 <= top_probability <= 1.0:
            raise ValueError(
                f"top-arm model probability for {top_arm} outside [0, 1]: "
                f"{top_probability!r}; the calibrated bundle appears broken "
                "or mis-stubbed"
            )

        policy_context = {**row, "recovery_probability": top_probability}
        decision = decide_action(policy_context, self._policy)
        return CandidateRecommendation(
            top_candidate_action=top_arm,
            incremental_revenue_estimate_inr=float(top_revenue),
            model_label=LABEL_MODEL_ESTIMATE,
            authorized_action=decision.authorized_action,
            authorization_reason=decision.reason,
            policy_overrode_candidate=decision.authorized_action != top_arm,
        )
