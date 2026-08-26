"""Day 6 decision-quality evidence and uncertainty quantification (D-E4).

This module quantifies the DECISION QUALITY and UNCERTAINTY of
incremental-revenue-based action selection inside the SYNTHETIC world
defined by the Day 4 simulator and the declarative treatment policy. Every
number here is either a MODEL ESTIMATE (from a fitted per-arm bundle) or the
noise-integrated SIMULATED GROUND TRUTH replayed via
``simulation.outcomes.ground_truth_propensity``; reports embed the labels
``OBSERVED SIMULATED OUTCOME`` and ``SIMULATED GROUND TRUTH`` verbatim so no
synthetic-world bookkeeping quantity can be mistaken for a real-world
claim, and nothing causal about any production system is supported anywhere
in this file. The emitted evidence bundle feeds the Day 6 evidence
classifier (``ml/decision_policy``, plan Task 5 / D-E5), which consumes its
metrics and validates its ``provenance_digest``.

Candidate set discipline (D-E4): decisions are taken over the FOUR TREATED
ARMS ONLY -- ``ARM_ORDER`` minus CONTROL. CONTROL is excluded because the
uniform retry-cost accounting makes CONTROL incremental revenue undefined
or negative by construction. For each randomized test row and each treated
arm ``a``:

    revenue_i(a) = (P_hat_a - P_hat_CONTROL) * amount_i
                   - RETRY_INTERVENTION_COST_INR
                   - risk_penalty_i

with ``risk_penalty_i = UNKNOWN_CATEGORY_RISK_FRACTION * amount_i`` iff
``failure_category == "unknown"`` (both constants IMPORTED from
``recovery.scoring``, never restated). The truth twin uses the identical
formula over noise-integrated ground-truth propensities. Because the cost
and risk terms are ARM-INDEPENDENT CONSTANTS per row, per-row argmax
decisions reduce to argmax incremental recovery x amount -- stated here so
cost sensitivity is not misread. Model argmax and truth argmax both break
ties deterministically toward the earlier ``ARM_ORDER`` precedence.

Metrics (all on the shared randomized row block):
1. ``decision_match_rate`` -- fraction of rows where the model argmax equals
   the truth argmax, with a binomial 95% CI via the normal approximation
   (unclipped p +- 1.96-sigma).
2. Relative regret -- (E[truth revenue at truth argmax] - E[truth revenue at
   model argmax]) divided by E[truth revenue at truth argmax]; DENOMINATOR
   GUARD: when that expectation is <= 0 the ratio is reported undefined with
   a reason string instead of dividing. Absolute regret in INR and PER-ROW
   regret quantiles p50/p90/p99 accompany it because mean relative regret
   hides heavy tails.
3. Per treated arm: seeded stratified bootstrap 95% CI (B=500 default,
   resampling within assigned-arm blocks at each block's TRUE filtered row
   positions, mirroring the Day-5 per-arm slice convention) around
   the mean MODEL incremental revenue, plus the pairwise CI-overlap lists
   (symmetric by construction, diagonal excluded).
4. ``uncertainty_inventory`` -- per-arm n, calibration status detected from
   bundle metadata ("calibration" key present means sigmoid-calibrated;
   anything else carries a loud warning because D-E4 evidence expects the
   calibrated shipped form), a propensity-range overlap note, and the
   seed-variance block: filled by the caller from multi-seed stability
   replicates (this module cannot refit bundles without train/validation
   frames) or computed when the caller supplies those replicate dicts.
5. ``provenance_digest`` -- sorted-json SHA256 of the report content,
   computed over the report EXCLUDING the digest field itself, so
   downstream consumers can detect non-canonical bundles visibly.

Native policy-safety probe (D-E5 criterion 3 producer): the report embeds
``policy_safety_probe(calibrated_bundle)`` output -- three canonical STOP
contexts (customer opted out / fraud risk / hard decline) replayed through
the bundle's counterfactual revenues and the frozen business rules via
``recovery.policy.decide_action``, passing only when EVERY context
authorizes STOP regardless of the revenue candidate. The Task 5 classifier
(``ml/decision_policy``) consumes this field, so genuine bundles satisfy the
classifier's structure by construction instead of relying on callers to
attach a hand-made probe result.

Determinism: the ONLY stochastic consumer is the stratified bootstrap, whose
generator is derived EXACTLY once from the named ``seed`` parameter via
``np.random.default_rng(seed)`` (default 20260826); two calls with identical
inputs produce byte-identical reports. Inputs are never mutated, no wall
clock is read, and every emitted value is a JSON-native primitive.

DEVIATION NOTES (reviewer-visible): stdlib roots ``hashlib``/``json`` are
REQUIRED by the provenance-digest contract above; the ``recovery`` root is
REQUIRED so the Day-2 scoring cost basis is imported rather than forked.
The plan sketch also mentioned reporting a pooled-bundle twin here; the
shipped contract evaluates ONE strict ``ActionModelBundle`` per call --
callers invoke the function once per family instead.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ml.action_evaluation import (
    LABEL_OBSERVED_SIMULATED_OUTCOME,
    TRUTH_LABEL_SIMULATED_GROUND_TRUTH,
    _detect_bundle_kind,
    _reject_missing_feature_columns,
    _reject_missing_observation_columns,
    _require_frame,
)
from ml.action_model import (
    ACTION_COLUMN,
    ARM_ORDER,
    STRATUM_COLUMN,
    STRATUM_RANDOMIZED,
    ActionModelBundle,
    predict_action_probability,
)
from recovery.policy import (
    PolicyConfig,
    decide_action,
    load_policy_config,
)
from recovery.scoring import (
    RETRY_INTERVENTION_COST_INR,
    UNKNOWN_CATEGORY_RISK_FRACTION,
)
from simulation.config import TreatmentPolicy
from simulation.outcomes import ground_truth_propensity

LABEL_OBSERVED = LABEL_OBSERVED_SIMULATED_OUTCOME
TRUTH_LABEL = TRUTH_LABEL_SIMULATED_GROUND_TRUTH

TREATED_ARMS = tuple(arm for arm in ARM_ORDER if arm != "CONTROL")

DEFAULT_BOOTSTRAP_REPLICATIONS = 500
MATCH_RATE_CI_Z_95 = 1.959963984540054
MATCH_RATE_CI_NOTE = (
    "binomial 95% CI via the normal (Wald) approximation "
    "p +- 1.96 * sqrt(p(1-p)/n); bounds are clamped into [0, 1] because "
    "the unclipped interval can overshoot at extreme rates"
)

SCOPE_NOTE = (
    "Synthetic-world-only quantification of decision quality for "
    "incremental-revenue-based action selection against the simulator's "
    "known ground truth; feeds the Day 6 evidence classifier; supports "
    "nothing causal about production."
)

CANDIDATE_SET_NOTE = (
    "Candidate set = the 4 treated arms (ARM_ORDER minus CONTROL); CONTROL "
    "is excluded because uniform retry-cost accounting leaves its "
    "incremental revenue undefined/negative by construction."
)

DECISION_RULE_NOTE = (
    "Per-row argmax of modeled vs noise-integrated truth incremental "
    "revenue over the candidate set; ties break deterministically to the "
    "earlier ARM_ORDER precedence on BOTH sides; cost/risk terms are "
    "arm-independent constants per row, so argmax decisions reduce to "
    "argmax incremental recovery x amount."
)

COST_SIMPLIFICATION_NOTE = (
    "A single retry-cost constant is applied uniformly to all treated arms "
    "including REQUEST_UPDATE and HUMAN_REVIEW, whose true economics differ."
)

BOOTSTRAP_SCHEME = (
    "stratified_within_assigned_arm_row_resampling_true_per_arm_positions"
)

PROPENSITY_OVERLAP_NOTE = (
    "Per treated arm: whether the observed [min, max] range of the MODEL "
    "estimate overlaps the [min, max] range of the noise-integrated "
    "SIMULATED GROUND TRUTH propensity on the identical rows; a coarse "
    "positivity-style sanity flag, never a gate."
)

DENOMINATOR_GUARD_REASON = (
    "relative_regret undefined: expected best-case SIMULATED GROUND TRUTH "
    "revenue is <= 0 (denominator guard), so no meaningful positive "
    "normalization exists on this synthetic slice"
)

SEED_VARIANCE_NOT_COMPUTED_NOTE = (
    "filled by caller: supply per-seed stability replicates (D-E1 runs) to "
    "compute spread across seeds"
)

POLICY_SAFETY_PROBE_NOTE = (
    "three canonical STOP contexts replayed through the calibrated bundle's "
    "counterfactual revenues and the frozen business rules; the gate passes "
    "only when EVERY context authorizes STOP regardless of the revenue "
    "candidate"
)

_PROBE_BASE_CONTEXT = {
    "amount_inr": 2500.0,
    "attempt_number": 1,
    "customer_tenure_days": 365,
    "successful_payment_count": 2,
    "failed_payment_count": 1,
    "historical_recovery_count": 1,
    "customer_opted_out": False,
    "fraud_risk": False,
    "payment_method": "card",
    "failure_code": "generic_decline",
    "failure_category": "temporary_decline",
    "issuer_response": "do_not_honor",
    "device_type": "mobile",
    "country": "US",
    # Label column read by the shared feature builder; ignored for queries.
    "recovered": 0,
}

_PROBE_CONTEXT_OVERRIDES = (
    ("customer_opted_out", {"customer_opted_out": True}),
    ("fraud_risk", {"fraud_risk": True}),
    ("hard_decline", {"failure_category": "hard_decline"}),
)


def policy_safety_probe(
    calibrated_bundle: ActionModelBundle,
    policy_config: PolicyConfig | None = None,
    seed: int = 20260826,
) -> dict:
    """Replay three canonical STOP contexts through bundle + business rules.

    For each crafted context -- customer opted out, fraud risk, hard decline
    -- per-arm MODEL ESTIMATE probabilities are queried exactly like the
    ``decision_evidence`` conventions (single-row frame COPY whose
    ``assigned_action`` column is OVERWRITTEN per queried arm), converted to
    incremental revenue with the imported Day-2 cost basis, and argmaxed
    over the four treated arms with ``ARM_ORDER`` tie-breaking. The TOP
    ARM's model probability is injected as ``recovery_probability``
    (documented choice mirroring ``ml/decision_policy``) and the frozen
    business rules authorize the final action through
    ``recovery.policy.decide_action``.

    Emits exactly::

        {
            "policy_safety_probe_passed": bool,   # EVERY context -> STOP
            "probe_details": [{context, candidate, authorized, overrode}],
            "label": "OBSERVED SIMULATED OUTCOME",
        }

    STOP dominance holds regardless of candidate revenue: an adversarial
    bundle can move the candidate but never the authorized verdict.
    Deterministic: consumes NO randomness (``seed`` exists for interface
    stability only); identical inputs reproduce byte-identical output.
    Synthetic-world-only; supports nothing causal about production.
    """
    if not isinstance(calibrated_bundle, ActionModelBundle):
        raise ValueError(
            "calibrated_bundle must be an ActionModelBundle, got "
            f"{type(calibrated_bundle).__name__}"
        )
    unknown_arms = sorted(set(calibrated_bundle.arms) - set(ARM_ORDER))
    if unknown_arms:
        raise ValueError(
            f"bundle carries non-canonical arms: {unknown_arms}"
        )
    if policy_config is None:
        policy = load_policy_config()
    else:
        if not isinstance(policy_config, PolicyConfig):
            raise ValueError(
                "policy_config must be a recovery.policy.PolicyConfig, got "
                f"{type(policy_config).__name__}"
            )
        policy = policy_config

    details: list[dict] = []
    for context_name, overrides in _PROBE_CONTEXT_OVERRIDES:
        row = {**_PROBE_BASE_CONTEXT, **overrides}
        query_frame = pd.DataFrame([row])
        probabilities = {}
        for arm in ARM_ORDER:
            counterfactual = query_frame.copy()
            counterfactual[ACTION_COLUMN] = arm
            probabilities[arm] = float(
                predict_action_probability(calibrated_bundle, counterfactual, arm)[0]
            )

        amount = float(row["amount_inr"])
        category = row["failure_category"]
        risk_penalty = (
            UNKNOWN_CATEGORY_RISK_FRACTION * amount
            if category == "unknown"
            else 0.0
        )
        revenues = {
            arm: (
                (probabilities[arm] - probabilities["CONTROL"]) * amount
                - RETRY_INTERVENTION_COST_INR
                - risk_penalty
            )
            for arm in TREATED_ARMS
        }
        candidate = TREATED_ARMS[0]
        for arm in TREATED_ARMS[1:]:
            if revenues[arm] > revenues[candidate]:
                candidate = arm

        decision = decide_action(
            {**row, "recovery_probability": probabilities[candidate]}, policy
        )
        details.append(
            {
                "context": context_name,
                "candidate": candidate,
                "authorized": decision.authorized_action,
                "overrode": decision.authorized_action != candidate,
            }
        )

    passed = all(entry["authorized"] == "STOP" for entry in details)
    return {
        "policy_safety_probe_passed": bool(passed),
        "probe_details": details,
        "label": LABEL_OBSERVED,
    }


def _require_treated_arm_columns(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in ARM_ORDER if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing probability columns {missing}; one column "
            f"per canonical arm {list(ARM_ORDER)} is required"
        )


def _revenue_by_arm(
    probability_frame: pd.DataFrame,
    amounts: np.ndarray,
    unknown_mask: np.ndarray,
    intervention_cost: float,
    unknown_risk_fraction: float,
) -> dict[str, np.ndarray]:
    """Incremental revenue per treated arm on the shared rows."""
    control = probability_frame["CONTROL"].to_numpy(dtype=float)
    return {
        arm: (
            (probability_frame[arm].to_numpy(dtype=float) - control) * amounts
            - intervention_cost
            - unknown_mask * unknown_risk_fraction * amounts
        )
        for arm in TREATED_ARMS
    }


def _argmax_names(revenue_by_arm: dict[str, np.ndarray], n_rows: int) -> np.ndarray:
    """Row-wise argmax arm names; ties resolve to earlier ARM_ORDER."""
    stacked = np.vstack([revenue_by_arm[arm] for arm in TREATED_ARMS])
    indices = np.argmax(stacked, axis=0)
    if indices.shape != (n_rows,):
        raise ValueError(
            f"argmax index shape {indices.shape} does not match the "
            f"{n_rows}-row decision problem; revenue arrays and probability "
            "frames must be positionally aligned one-to-one"
        )
    return np.array(TREATED_ARMS, dtype=object)[indices]


def _binomial_ci95(matches: int, n_rows: int, rate: float) -> list[float]:
    """Normal-approximation binomial CI95 for a rate, clamped to [0, 1].

    The unclipped Wald interval can overshoot the unit interval at extreme
    rates (e.g. rate 1.0 on few rows); bounds are clamped into [0, 1] and
    the emitted ``decision_match_rate_ci95_note`` records this.
    """
    standard_error = float(np.sqrt(rate * (1.0 - rate) / n_rows))
    half_width = MATCH_RATE_CI_Z_95 * standard_error
    low = min(max(rate - half_width, 0.0), 1.0)
    high = min(max(rate + half_width, 0.0), 1.0)
    return [float(low), float(high)]


def _decision_core(
    model_probability_frame: pd.DataFrame,
    truth_probability_frame: pd.DataFrame,
    amounts: np.ndarray,
    categories: np.ndarray,
    intervention_cost: float = RETRY_INTERVENTION_COST_INR,
    unknown_risk_fraction: float = UNKNOWN_CATEGORY_RISK_FRACTION,
) -> dict:
    """Pure decision-quality kernel over aligned probability frames.

    Both frames carry one probability COLUMN per canonical arm (CONTROL
    included, positionally row-aligned); ``amounts`` and ``categories`` are
    length-n arrays. Computes both sides' revenues with the documented
    formula, per-row argmaxes with ARM_ORDER tie-breaking, the decision-
    match rate with its normal-approximation binomial CI95, absolute/
    relative regret with the <=0 denominator guard, per-row regret
    quantiles (numpy linear-interpolation percentiles), and per-arm mean
    model/truth revenues. Deterministic: draws no randomness.
    """
    for name, frame in (
        ("model_probability_frame", model_probability_frame),
        ("truth_probability_frame", truth_probability_frame),
    ):
        _require_frame(frame, name)
        _require_treated_arm_columns(frame, name)
        if len(frame) != len(amounts) or len(frame) != len(categories):
            raise ValueError(
                f"{name} holds {len(frame)} rows but amounts/categories "
                f"carry {len(amounts)}/{len(categories)} entries; the "
                "decision kernel requires positionally aligned inputs"
            )
    n_rows = int(len(model_probability_frame))
    if len(truth_probability_frame) != n_rows:
        raise ValueError(
            "model and truth probability frames must be row-aligned, got "
            f"{n_rows} vs {len(truth_probability_frame)}"
        )
    if n_rows == 0:
        raise ValueError("the decision kernel requires at least one row")

    amounts_array = np.asarray(amounts, dtype=float)
    unknown_mask = np.asarray(categories) == "unknown"

    model_revenue = _revenue_by_arm(
        model_probability_frame,
        amounts_array,
        unknown_mask,
        intervention_cost,
        unknown_risk_fraction,
    )
    truth_revenue = _revenue_by_arm(
        truth_probability_frame,
        amounts_array,
        unknown_mask,
        intervention_cost,
        unknown_risk_fraction,
    )
    model_names = _argmax_names(model_revenue, n_rows)
    truth_names = _argmax_names(truth_revenue, n_rows)

    matches = int((model_names == truth_names).sum())
    match_rate = float(matches) / float(n_rows)

    truth_stacked = np.vstack([truth_revenue[arm] for arm in TREATED_ARMS])
    name_to_index = {arm: index for index, arm in enumerate(TREATED_ARMS)}
    truth_best_per_row = truth_stacked[
        np.fromiter((name_to_index[name] for name in truth_names), dtype=int, count=n_rows),
        np.arange(n_rows),
    ]
    model_chosen_truth = truth_stacked[
        np.fromiter((name_to_index[name] for name in model_names), dtype=int, count=n_rows),
        np.arange(n_rows),
    ]
    per_row_regret = truth_best_per_row - model_chosen_truth
    expected_best = float(np.mean(truth_best_per_row))
    absolute_regret = float(np.mean(per_row_regret))

    relative_regret = None
    reason = None
    if expected_best > 0.0:
        relative_regret = absolute_regret / expected_best
    else:
        reason = DENOMINATOR_GUARD_REASON

    quantile_levels = (50.0, 90.0, 99.0)
    bounds = np.percentile(per_row_regret, quantile_levels)
    return {
        "n": n_rows,
        "match_count": matches,
        "match_rate": match_rate,
        "match_ci95": _binomial_ci95(matches, n_rows, match_rate),
        "model_argmax_names": model_names,
        "truth_argmax_names": truth_names,
        "model_revenue_by_arm": model_revenue,
        "truth_revenue_by_arm": truth_revenue,
        "mean_model_revenue": {
            arm: float(np.mean(model_revenue[arm])) for arm in TREATED_ARMS
        },
        "mean_truth_revenue": {
            arm: float(np.mean(truth_revenue[arm])) for arm in TREATED_ARMS
        },
        "absolute_regret_inr": absolute_regret,
        "relative_regret": relative_regret,
        "relative_regret_reason": reason,
        "expected_best_truth_revenue_inr": expected_best,
        "regret_quantiles": {
            key: float(value)
            for key, value in zip(("p50", "p90", "p99"), bounds)
        },
    }


def _provenance_digest(metrics: dict) -> str:
    """Sorted-json SHA256 over the metrics payload (digest field excluded).

    Compact separators keep the canonical form stable; ``allow_nan=False``
    makes any NaN/Infinity payload fail loudly here instead of emitting
    non-standard JSON literals -- this module reports undefined quantities
    as ``None`` so payloads stay clean finite JSON.
    """
    payload = json.dumps(
        metrics, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assigned_arm_strata_blocks(randomized: pd.DataFrame) -> list[np.ndarray]:
    """TRUE per-arm position blocks over the randomized row pool.

    Mirrors the Day-5 convention (action_evaluation per-arm slices): block i
    holds EXACTLY the row positions whose assigned_action equals
    ARM_ORDER[i] -- ``np.flatnonzero`` over the arm mask -- so resampling
    within a block draws only from that arm's rows even under fully
    interleaved assignment. Unrecognized assigned arms leave a coverage gap
    and raise loudly rather than silently dropping rows from the resample.
    """
    blocks: list[np.ndarray] = []
    for arm in ARM_ORDER:
        blocks.append(
            np.flatnonzero((randomized[ACTION_COLUMN] == arm).to_numpy())
        )
    covered = int(sum(block.size for block in blocks))
    if covered != int(len(randomized)):
        raise ValueError(
            f"randomized rows carry unrecognized assigned_action values; "
            f"canonical arm blocks cover {covered} of {len(randomized)} rows"
        )
    return blocks


def _bootstrap_mean_ci(
    values: np.ndarray,
    strata_blocks: list[np.ndarray],
    rng: np.random.Generator,
    replications: int,
) -> list[float]:
    """Percentile bootstrap CI95 for a mean under stratified resampling."""
    statistics = np.empty(replications, dtype=float)
    for replication in range(replications):
        resampled = np.concatenate(
            [
                block[rng.integers(0, block.size, block.size)]
                for block in strata_blocks
            ]
        )
        statistics[replication] = float(np.mean(values[resampled]))
    low, high = np.percentile(statistics, [2.5, 97.5])
    return [float(low), float(high)]


def _calibration_status(bundle_kind: str) -> str:
    if bundle_kind == "calibrated":
        return "calibrated"
    return (
        "WARNING: raw bundle (metadata lacks 'calibration' record); D-E4 "
        "evidence expects the sigmoid-calibrated shipped form produced by "
        "calibrate_action_models"
    )


def _population_sd(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if len(finite) < 2:
        return None
    return float(np.std(np.asarray(finite, dtype=float), ddof=0))


def _seed_variance_block(stability_runs: list[dict] | tuple[dict, ...] | None) -> dict:
    """Seed-variance inventory entry from caller-supplied stability runs.

    Each run dict must carry ``decision_match_rate``, ``relative_regret``
    (float or None for guarded slices), and an ``arms`` mapping with
    ``mean_model_revenue`` per treated arm. Population sd (ddof=0) across
    runs is reported for each quantity; None-valued regrets are skipped and
    yield None when fewer than two finite values survive.
    """
    if stability_runs is None:
        return {
            "status": "not_computed",
            "note": SEED_VARIANCE_NOT_COMPUTED_NOTE,
        }
    runs = list(stability_runs)
    if len(runs) < 2:
        raise ValueError(
            "computing seed variance requires at least TWO caller-supplied "
            f"stability runs, got {len(runs)}"
        )
    match_rates: list[float] = []
    regrets: list[float] = []
    revenue_values: dict[str, list[float]] = {arm: [] for arm in TREATED_ARMS}
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(
                f"stability run #{index} must be a dict, got "
                f"{type(run).__name__}"
            )
        missing = [
            key
            for key in ("decision_match_rate", "relative_regret", "arms")
            if key not in run
        ]
        if missing:
            raise ValueError(
                f"stability run #{index} is missing required keys {missing}"
            )
        arms_block = run["arms"]
        absent_arms = [arm for arm in TREATED_ARMS if arm not in arms_block]
        if absent_arms:
            raise ValueError(
                f"stability run #{index} lacks treated arms {absent_arms}; "
                f"every arm in {list(TREATED_ARMS)} is required"
            )
        incomplete_entries = [
            arm
            for arm in TREATED_ARMS
            if not isinstance(arms_block[arm], dict)
            or "mean_model_revenue" not in arms_block[arm]
        ]
        if incomplete_entries:
            raise ValueError(
                f"stability run #{index} arm entries {incomplete_entries} "
                f"lack the 'mean_model_revenue' key; every treated-arm "
                "entry must carry its mean model incremental revenue"
            )
        match_rates.append(float(run["decision_match_rate"]))
        if run["relative_regret"] is not None:
            regrets.append(float(run["relative_regret"]))
        for arm in TREATED_ARMS:
            revenue_values[arm].append(
                float(arms_block[arm]["mean_model_revenue"])
            )
    return {
        "status": "computed_from_caller_supplied_stability_runs",
        "n_runs": len(runs),
        "decision_match_rate_sd": _population_sd(match_rates),
        "relative_regret_sd": _population_sd(regrets),
        "mean_model_revenue_sd_by_arm": {
            arm: _population_sd(revenue_values[arm]) for arm in TREATED_ARMS
        },
        "note": (
            "population sd across caller-supplied per-seed stability "
            "replicates (D-E1/D-E4); this module does not refit models"
        ),
    }


def decision_evidence(
    calibrated_bundle: ActionModelBundle,
    test_frame: pd.DataFrame,
    policy: TreatmentPolicy,
    seed: int = 20260826,
    *,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    stability_runs: list[dict] | tuple[dict, ...] | None = None,
) -> dict:
    """Quantify decision quality + uncertainty for one calibrated bundle.

    Scores EVERY randomized test row under EVERY treated arm (plus CONTROL
    as the contrast baseline) through ``predict_action_probability`` and,
    identically shaped, through the noise-integrated
    ``ground_truth_propensity`` replay; converts both sides to incremental
    revenue with the imported Day-2 cost basis; measures the decision-match
    rate (binomial CI95), absolute/relative regret (denominator-guarded)
    with per-row regret quantiles p50/p90/p99, per-arm stratified-bootstrap
    CIs around mean model revenue with pairwise overlap lists, and an
    uncertainty inventory whose calibration status warns loudly on raw
    bundles. It also runs the native ``policy_safety_probe`` over three
    canonical STOP contexts and embeds its verdict as the top-level
    ``policy_safety_probe_passed`` / ``policy_safety_probe_details`` keys,
    INSIDE the provenance content: the report self-hashes into
    ``provenance_digest`` (sorted-json SHA256 excluding the digest field
    itself). ``stability_runs``, when supplied, must hold >= 2 per-seed
    replicate dicts and yields population sds of match rate / relative
    regret / per-arm mean model revenue.

    Nothing mutates its inputs; the ONLY randomness is the single named-
    seed bootstrap stream; identical inputs reproduce byte-identical
    reports. All quantities describe the synthetic world only and support
    nothing causal about any production system.
    """
    _require_frame(test_frame, "test_frame")
    _reject_missing_observation_columns(test_frame, "test_frame")
    _reject_missing_feature_columns(test_frame, "test_frame")
    if not isinstance(calibrated_bundle, ActionModelBundle):
        raise ValueError(
            "calibrated_bundle must be an ActionModelBundle, got "
            f"{type(calibrated_bundle).__name__}"
        )
    unknown_arms = sorted(set(calibrated_bundle.arms) - set(ARM_ORDER))
    if unknown_arms:
        raise ValueError(
            f"bundle carries non-canonical arms: {unknown_arms}"
        )
    if not isinstance(policy, TreatmentPolicy):
        raise ValueError(
            f"policy must be a TreatmentPolicy, got {type(policy).__name__}"
        )
    if bootstrap_replications < 1:
        raise ValueError(
            "bootstrap_replications must be >= 1, got "
            f"{bootstrap_replications}"
        )

    randomized = test_frame.loc[
        test_frame[STRATUM_COLUMN] == STRATUM_RANDOMIZED
    ]
    n_rows = int(len(randomized))
    if n_rows == 0:
        raise ValueError(
            "test_frame holds zero 'randomized' stratum rows; decision "
            "evidence consumes only randomized observations"
        )

    # The ONE sanctioned randomness derivation: named-seed bootstrap stream.
    rng = np.random.default_rng(seed)

    model_probabilities = pd.DataFrame(
        {
            arm: np.asarray(
                predict_action_probability(calibrated_bundle, randomized, arm),
                dtype=float,
            )
            for arm in ARM_ORDER
        },
        columns=list(ARM_ORDER),
    )
    truth_probabilities = pd.DataFrame(
        {
            arm: np.asarray(ground_truth_propensity(randomized, policy, arm))
            for arm in ARM_ORDER
        },
        columns=list(ARM_ORDER),
    )
    core = _decision_core(
        model_probabilities,
        truth_probabilities,
        randomized["amount_inr"].to_numpy(dtype=float),
        randomized["failure_category"].to_numpy(),
    )

    strata_blocks = _assigned_arm_strata_blocks(randomized)
    arms_block = {}
    ci_bounds = {}
    for arm in TREATED_ARMS:
        values = core["model_revenue_by_arm"][arm]
        bounds = _bootstrap_mean_ci(
            values, strata_blocks, rng, bootstrap_replications
        )
        ci_bounds[arm] = bounds
        arms_block[arm] = {
            "n": n_rows,
            "mean_model_revenue": core["mean_model_revenue"][arm],
            "mean_truth_revenue": core["mean_truth_revenue"][arm],
            "bootstrap_ci95_mean_model_revenue": bounds,
        }
    for first_index, first_arm in enumerate(TREATED_ARMS):
        overlaps: list[str] = []
        lo_first, hi_first = ci_bounds[first_arm]
        for second_arm in TREATED_ARMS:
            if second_arm == first_arm:
                continue
            lo_second, hi_second = ci_bounds[second_arm]
            if lo_first <= hi_second and lo_second <= hi_first:
                overlaps.append(second_arm)
        arms_block[first_arm]["ci_overlap_with"] = overlaps

    propensity_overlap = {}
    for arm in TREATED_ARMS:
        model_range = (
            float(np.min(model_probabilities[arm])),
            float(np.max(model_probabilities[arm])),
        )
        truth_range = (
            float(np.min(truth_probabilities[arm])),
            float(np.max(truth_probabilities[arm])),
        )
        propensity_overlap[arm] = bool(
            model_range[0] <= truth_range[1]
            and truth_range[0] <= model_range[1]
        )

    bundle_kind = _detect_bundle_kind(calibrated_bundle)
    inventory = {
        "per_arm_n": {arm: n_rows for arm in TREATED_ARMS},
        "calibration_status": _calibration_status(bundle_kind),
        "propensity_overlap_note": PROPENSITY_OVERLAP_NOTE,
        "propensity_range_overlap_by_arm": propensity_overlap,
        "seed_variance": _seed_variance_block(stability_runs),
    }

    probe = policy_safety_probe(calibrated_bundle)

    report = {
        "label": LABEL_OBSERVED,
        "truth_label": TRUTH_LABEL,
        "scope_note": SCOPE_NOTE,
        "candidate_arms": TREATED_ARMS,
        "candidate_set_note": CANDIDATE_SET_NOTE,
        "decision_rule_note": DECISION_RULE_NOTE,
        "cost_simplification_note": COST_SIMPLIFICATION_NOTE,
        "seed": int(seed),
        "bundle_kind": bundle_kind,
        "bootstrap": {
            "replications": int(bootstrap_replications),
            "confidence_level": 0.95,
            "scheme": BOOTSTRAP_SCHEME,
        },
        "n_randomized_test_rows": n_rows,
        "decision_match_rate": core["match_rate"],
        "decision_match_count": core["match_count"],
        "decision_match_rate_ci95": core["match_ci95"],
        "decision_match_rate_ci95_note": MATCH_RATE_CI_NOTE,
        "relative_regret": core["relative_regret"],
        "relative_regret_reason": core["relative_regret_reason"],
        "absolute_regret_inr": core["absolute_regret_inr"],
        "expected_best_truth_revenue_inr": core[
            "expected_best_truth_revenue_inr"
        ],
        "regret_quantiles": core["regret_quantiles"],
        "arms": arms_block,
        "uncertainty_inventory": inventory,
        "policy_safety_probe_passed": probe["policy_safety_probe_passed"],
        "policy_safety_probe_details": probe["probe_details"],
    }
    report["provenance_digest"] = _provenance_digest(
        {key: value for key, value in report.items() if key != "provenance_digest"}
    )
    return report
