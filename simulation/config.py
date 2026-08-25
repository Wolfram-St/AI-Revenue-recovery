"""Declarative synthetic-world configuration for Day 4 treatment outcomes.

This module loads ``config/treatment_policy.yaml`` into typed frozen
dataclasses using ``yaml.safe_load`` only. Nothing executable ships in the
YAML: the file is pure declarative data, the loader enforces a closed key
vocabulary, and every number is an ILLUSTRATIVE SYNTHETIC choice describing
the simulated world -- none are estimated from any real provider data.

Validation discipline mirrors ``recovery/policy.py``: every violation raises
a loud ``ValueError`` naming the offending item instead of silently coercing
or defaulting. Loading is deterministic -- two loads produce equal objects
because all mappings are rebuilt in a fixed documented order regardless of
YAML key order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_POLICY_PATH = "config/treatment_policy.yaml"

SEED_STREAM_ASSIGNMENT = 0
"""Child index 0 of ``default_rng(master_seed).spawn(k)``: stage-2 assignment
multinomial draws (plan decision D1b, fixed spawn order)."""

SEED_STREAM_OUTCOMES = 1
"""Child index 1 of ``default_rng(master_seed).spawn(k)``: outcome Bernoulli
draws plus logit noise (plan decision D1b, fixed spawn order)."""

SEED_STREAM_TEMPORAL = 2
"""Child index 2 of ``default_rng(master_seed).spawn(k)``: temporal
resolution-window draws (plan decision D1b, fixed spawn order)."""

CANONICAL_ARMS: frozenset[str] = frozenset(
    {"CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW"}
)

TREATED_ARMS: frozenset[str] = CANONICAL_ARMS - {"CONTROL"}

CANONICAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "temporary_decline",
        "payment_method_issue",
        "authentication_required",
        "unknown",
        "hard_decline",
    }
)

INTERACTION_COLUMNS: frozenset[str] = frozenset(
    {"failure_category", "attempt_number"}
)

_ARM_ORDER = ("CONTROL", "RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW")
_CATEGORY_ORDER = (
    "temporary_decline",
    "payment_method_issue",
    "authentication_required",
    "unknown",
    "hard_decline",
)

_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "master_seed",
        "arm_probabilities",
        "main_effects_logit",
        "interactions_logit",
        "noise_sigma_logit",
        "treatment_delay_hours",
        "resolution_window_hours",
        "base_propensity_terms",
    }
)

_ALLOWED_INTERACTION_KEYS = frozenset(
    {"action", "column", "equals_value", "min_threshold", "effect_logit"}
)
_REQUIRED_INTERACTION_KEYS = ("action", "column", "effect_logit")

_BASE_TERM_FIELDS = (
    "intercept",
    "successful_payment_count_log1p",
    "historical_recovery_count_min5",
    "attempt_number_prior_offset",
    "fraud_risk",
    "amount_log1p_per_k",
    "method_upi",
    "device_android",
)

_PROBABILITY_TOLERANCE = 1e-9
_EFFECT_BOUND = 3.0
_MAX_NOISE_SIGMA = 5.0
_MAX_DELAY_HOURS = 168.0
_MAX_RESOLUTION_HOURS = 720.0


@dataclass(frozen=True)
class InteractionRule:
    """One bounded interaction effect; exactly one shape field may be set."""

    action: str
    column: str
    equals_value: str | None
    min_threshold: int | None
    effect_logit: float

    def __post_init__(self) -> None:
        provided = sum(
            value is not None for value in (self.equals_value, self.min_threshold)
        )
        if provided != 1:
            raise ValueError(
                f"interaction rule {self.action!r}/{self.column!r} must set "
                "exactly one of equals_value or min_threshold"
            )


@dataclass(frozen=True)
class BasePropensityTerms:
    """Fixed base-recovery-propensity coefficients (synthetic world truth)."""

    intercept: float
    category_effects: dict[str, float]
    successful_payment_count_log1p: float
    historical_recovery_count_min5: float
    attempt_number_prior_offset: float
    fraud_risk: float
    amount_log1p_per_k: float
    method_upi: float
    device_android: float


@dataclass(frozen=True)
class TreatmentPolicy:
    """Frozen synthetic treatment/outcome policy loaded from YAML."""

    version: str
    master_seed: int
    arm_probabilities: dict[str, float]
    main_effects_logit: dict[str, float]
    interactions: tuple[InteractionRule, ...]
    noise_sigma_logit: float
    treatment_delay_hours: dict[str, float]
    resolution_window_hours: tuple[float, float]
    base_propensity_terms: BasePropensityTerms


def _require_mapping(raw: object, label: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must parse to a mapping, got {type(raw).__name__}")
    return raw


def _required_section(raw: dict, name: str) -> object:
    if name not in raw:
        raise ValueError(f"missing required configuration section: {name}")
    return raw[name]


def _check_exact_keys(mapping: dict, expected: frozenset | set, label: str) -> None:
    keys = set(mapping)
    extra = sorted(keys - set(expected))
    missing = sorted(set(expected) - keys)
    if extra or missing:
        raise ValueError(
            f"{label}: unexpected keys {extra}, missing keys {missing}"
        )


def _as_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _load_interactions(raw: object) -> tuple[InteractionRule, ...]:
    if not isinstance(raw, list):
        raise ValueError("interactions_logit must be a list of mappings")
    rules: list[InteractionRule] = []
    for index, entry in enumerate(raw):
        label = f"interactions_logit[{index}]"
        entry_map = _require_mapping(entry, label)
        unknown = sorted(set(entry_map) - _ALLOWED_INTERACTION_KEYS)
        if unknown:
            raise ValueError(f"{label}: unknown keys {unknown}")
        for required in _REQUIRED_INTERACTION_KEYS:
            if required not in entry_map:
                raise ValueError(f"{label}: missing required key {required!r}")
        action = entry_map["action"]
        if not isinstance(action, str) or action not in TREATED_ARMS:
            raise ValueError(
                f"{label}: action {action!r} is not a treated canonical arm "
                f"{sorted(TREATED_ARMS)}"
            )
        label = f"{label} action {action!r}"
        column = entry_map["column"]
        if not isinstance(column, str) or column not in INTERACTION_COLUMNS:
            raise ValueError(
                f"{label}: column {column!r} is outside the closed enum "
                f"{sorted(INTERACTION_COLUMNS)}"
            )
        effect = _as_finite_float(entry_map["effect_logit"], f"{label}.effect_logit")
        if abs(effect) > _EFFECT_BOUND:
            raise ValueError(
                f"{label}.effect_logit {effect!r} outside [{-_EFFECT_BOUND}, "
                f"{_EFFECT_BOUND}]"
            )
        equals_value = entry_map.get("equals_value")
        min_threshold = entry_map.get("min_threshold")
        if column == "failure_category":
            if "min_threshold" in entry_map:
                raise ValueError(
                    f"{label}: min_threshold is forbidden for column "
                    "'failure_category'"
                )
            if not isinstance(equals_value, str) or not equals_value.strip():
                raise ValueError(
                    f"{label}: equals_value must be a non-empty string for "
                    "column 'failure_category'"
                )
            if equals_value not in CANONICAL_CATEGORIES:
                raise ValueError(
                    f"{label}: unknown failure_category for interaction "
                    f"equals_value {equals_value!r}; expected one of "
                    f"{sorted(CANONICAL_CATEGORIES)}"
                )
        elif column == "attempt_number":
            if "equals_value" in entry_map:
                raise ValueError(
                    f"{label}: equals_value is forbidden for column "
                    "'attempt_number'; use min_threshold"
                )
            if isinstance(min_threshold, bool) or not isinstance(min_threshold, int):
                raise ValueError(
                    f"{label}: min_threshold must be an integer for column "
                    f"'attempt_number', got {min_threshold!r}"
                )
            if min_threshold < 1:
                raise ValueError(
                    f"{label}: min_threshold must be >= 1, got {min_threshold}"
                )
        rules.append(
            InteractionRule(
                action=action,
                column=column,
                equals_value=equals_value if column == "failure_category" else None,
                min_threshold=min_threshold if column == "attempt_number" else None,
                effect_logit=effect,
            )
        )
    return tuple(rules)


def _load_arm_probabilities(raw: object) -> dict[str, float]:
    raw_map = _require_mapping(raw, "arm_probabilities")
    _check_exact_keys(raw_map, CANONICAL_ARMS, "arm_probabilities")
    probabilities: dict[str, float] = {}
    for arm in _ARM_ORDER:
        value = _as_finite_float(raw_map[arm], f"arm_probabilities.{arm}")
        if arm == "CONTROL":
            if not 0.0 <= value < 1.0:
                raise ValueError(
                    f"arm_probabilities.CONTROL must lie in [0, 1), got {value!r}"
                )
        elif not 0.0 < value <= 1.0:
            raise ValueError(
                f"arm_probabilities.{arm} must lie in (0, 1], got {value!r}"
            )
        probabilities[arm] = value
    total = math.fsum(probabilities.values())
    if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
        raise ValueError(
            f"arm_probabilities must sum to 1.0 within {_PROBABILITY_TOLERANCE}; "
            f"got sum={total!r} from {probabilities}"
        )
    return probabilities


def _load_main_effects(raw: object) -> dict[str, float]:
    raw_map = _require_mapping(raw, "main_effects_logit")
    _check_exact_keys(raw_map, CANONICAL_ARMS, "main_effects_logit")
    effects: dict[str, float] = {}
    for arm in _ARM_ORDER:
        value = _as_finite_float(raw_map[arm], f"main_effects_logit.{arm}")
        if abs(value) > _EFFECT_BOUND:
            raise ValueError(
                f"main_effects_logit.{arm} {value!r} outside "
                f"[{-_EFFECT_BOUND}, {_EFFECT_BOUND}]"
            )
        if arm == "CONTROL" and value != 0.0:
            raise ValueError(
                f"main_effects_logit.CONTROL must be exactly 0.0, got {value!r}"
            )
        effects[arm] = value
    return effects


def _load_treatment_delays(raw: object) -> dict[str, float]:
    raw_map = _require_mapping(raw, "treatment_delay_hours")
    _check_exact_keys(raw_map, TREATED_ARMS, "treatment_delay_hours")
    delays: dict[str, float] = {}
    for arm in _ARM_ORDER:
        if arm == "CONTROL":
            continue
        value = _as_finite_float(raw_map[arm], f"treatment_delay_hours.{arm}")
        if not 0.0 < value <= _MAX_DELAY_HOURS:
            raise ValueError(
                f"treatment_delay_hours.{arm} must lie in (0, {_MAX_DELAY_HOURS}], "
                f"got {value!r}"
            )
        delays[arm] = value
    return delays


def _load_resolution_window(raw: object) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            "resolution_window_hours must be a two-item sequence [low, high]"
        )
    low = _as_finite_float(raw[0], "resolution_window_hours[0]")
    high = _as_finite_float(raw[1], "resolution_window_hours[1]")
    if not 0.0 < low <= high <= _MAX_RESOLUTION_HOURS:
        raise ValueError(
            f"resolution_window_hours must satisfy 0 < low <= high <= "
            f"{_MAX_RESOLUTION_HOURS}, got [{low!r}, {high!r}]"
        )
    return (low, high)


def _load_base_terms(raw: object) -> BasePropensityTerms:
    raw_map = _require_mapping(raw, "base_propensity_terms")
    expected = set(_BASE_TERM_FIELDS) | {"category_effects"}
    _check_exact_keys(raw_map, expected, "base_propensity_terms")
    category_raw = _require_mapping(
        raw_map["category_effects"], "base_propensity_terms.category_effects"
    )
    _check_exact_keys(
        category_raw, CANONICAL_CATEGORIES, "base_propensity_terms.category_effects"
    )
    category_effects = {
        category: _as_finite_float(
            category_raw[category],
            f"base_propensity_terms.category_effects.{category}",
        )
        for category in _CATEGORY_ORDER
    }
    scalars = {
        field: _as_finite_float(raw_map[field], f"base_propensity_terms.{field}")
        for field in _BASE_TERM_FIELDS
    }
    return BasePropensityTerms(category_effects=category_effects, **scalars)


def load_treatment_policy(path: str | Path = DEFAULT_POLICY_PATH) -> TreatmentPolicy:
    """Load and validate the frozen synthetic treatment policy.

    An empty interactions list is valid (a no-interaction synthetic world);
    the shipped configuration ships exactly the two documented terms.
    """
    raw_document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = _require_mapping(raw_document, f"treatment policy document {str(path)!r}")

    unknown_top_level = sorted(set(raw) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ValueError(f"unknown top-level keys {unknown_top_level}")

    version = _required_section(raw, "version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"version must be a non-empty string, got {version!r}")

    master_seed = _required_section(raw, "master_seed")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError(
            f"master_seed must be a positive integer, got {master_seed!r}"
        )
    if master_seed <= 0:
        raise ValueError(f"master_seed must be positive, got {master_seed}")

    noise_sigma = _as_finite_float(
        _required_section(raw, "noise_sigma_logit"), "noise_sigma_logit"
    )
    if not 0.0 < noise_sigma <= _MAX_NOISE_SIGMA:
        raise ValueError(
            f"noise_sigma_logit must lie in (0, {_MAX_NOISE_SIGMA}], got "
            f"{noise_sigma!r}"
        )

    return TreatmentPolicy(
        version=version,
        master_seed=master_seed,
        arm_probabilities=_load_arm_probabilities(
            _required_section(raw, "arm_probabilities")
        ),
        main_effects_logit=_load_main_effects(
            _required_section(raw, "main_effects_logit")
        ),
        interactions=_load_interactions(_required_section(raw, "interactions_logit")),
        noise_sigma_logit=noise_sigma,
        treatment_delay_hours=_load_treatment_delays(
            _required_section(raw, "treatment_delay_hours")
        ),
        resolution_window_hours=_load_resolution_window(
            _required_section(raw, "resolution_window_hours")
        ),
        base_propensity_terms=_load_base_terms(
            _required_section(raw, "base_propensity_terms")
        ),
    )
