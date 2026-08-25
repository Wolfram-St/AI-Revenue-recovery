"""Policy configuration loading and safe condition evaluation.

Conditions come from the frozen ``config/business_rules.yaml`` artifact and
are never executed dynamically. ``parse_condition`` builds a syntax tree with
``ast.parse`` in eval mode (which runs nothing) and then walks it against a
strict whitelist. The accepted grammar: the root is a single comparison or
comparisons joined by ``and``; each comparison pits one column name against a
literal or another column name using == != >= <= > <; lowercase ``true`` and
``false`` are reserved boolean literals and cannot be column names. Anything
else -- ``or``, calls, attributes, arithmetic, lambdas, comprehensions,
imports, subscripts, chained comparisons -- raises ValueError naming the
rejected construct rather than evaluating to False. Column references resolve
via ``context[name]`` at evaluation time, so a missing column raises KeyError.

Decision layer
--------------
``decide_action`` implements authorization, not recommendation: it consumes
only decision-time context facts and the calibrated probability
``P(recovered | context)``; it never sees or overrides any external AI
recommendation because none exists at this layer. Whatever action it returns
is the authorized action, resolved by deterministic precedence over the
frozen rules and explained by a reason string on every decision.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import yaml

CANONICAL_ACTIONS: frozenset[str] = frozenset(
    {"RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW", "STOP"}
)

RESIDUAL_DEFAULT_ACTION = "RETRY_LATER"
"""Documented placeholder authorized when no rule matches.

This is the residual default until a future ERV-based NOW-vs-LATER chooser
exists to decide between ``RETRY_NOW`` and ``RETRY_LATER`` on expected
recovery value; it keeps the action vocabulary closed in the meantime.
"""

DEFAULT_CONFIG_PATH = "config/business_rules.yaml"
MAX_CONDITION_LENGTH = 256

_BOOLEAN_LITERALS = {"true": True, "false": False}
_ALLOWED_COMPARE_OPS = (ast.Eq, ast.NotEq, ast.GtE, ast.LtE, ast.Gt, ast.Lt)
_ALLOWED_CONSTANT_TYPES = (int, float, str)
_CLAUSE_ERROR = "condition must be a comparison or comparisons joined by and"


@dataclass(frozen=True)
class PolicyRule:
    id: str
    name: str
    priority: int
    action: str
    reason: str
    condition_ast: ast.AST
    condition_text: str
    enabled: bool = True


@dataclass(frozen=True)
class PolicyDecision:
    authorized_action: str
    matched_rule_id: str | None
    matched_rule_name: str | None
    priority: int | None
    reason: str
    is_stop: bool
    evaluated_rules: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class PolicyConfig:
    version: str
    stop_precedence: bool
    rules: tuple[PolicyRule, ...]
    canonical_actions: frozenset[str]


def _reject(node: ast.AST) -> ValueError:
    return ValueError(f"unsupported construct in condition: {type(node).__name__}")


def _validate_comparison_operand(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.Constant):
        if type(node.value) not in _ALLOWED_CONSTANT_TYPES:
            raise ValueError(
                f"literal {node.value!r} of type {type(node.value).__name__} is not "
                "allowed; use int, float, quoted string, or lowercase true/false"
            )
        return
    raise _reject(node)


def _validate_clause(node: ast.AST) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And):
            raise ValueError(
                f"boolean operator {type(node.op).__name__} is not allowed; only 'and' is supported"
            )
        for operand in node.values:
            _validate_clause(operand)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ValueError("chained comparisons are not allowed; use a single comparison")
        if not isinstance(node.ops[0], _ALLOWED_COMPARE_OPS):
            raise ValueError(
                f"comparison operator {type(node.ops[0]).__name__} is not allowed"
            )
        _validate_comparison_operand(node.left)
        _validate_comparison_operand(node.comparators[0])
    else:
        raise ValueError(_CLAUSE_ERROR)


def parse_condition(text: str) -> object:
    """Parse a policy condition into a validated AST.

    Raises ValueError for anything outside the supported grammar; parsing
    never executes the input and rejection is loud, never a silent False.
    """
    if not isinstance(text, str):
        raise ValueError("condition must be a string")
    stripped = text.strip()
    if not stripped:
        raise ValueError("condition must not be empty")
    if len(text) > MAX_CONDITION_LENGTH:
        raise ValueError(
            f"condition exceeds the {MAX_CONDITION_LENGTH}-character limit"
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"condition does not match the grammar: {exc.msg}") from exc
    if not isinstance(tree, ast.Expression):
        raise ValueError(_CLAUSE_ERROR)
    _validate_clause(tree.body)
    return tree


def _resolve_operand(node: ast.AST, context: dict) -> object:
    if isinstance(node, ast.Name):
        if node.id in _BOOLEAN_LITERALS:
            return _BOOLEAN_LITERALS[node.id]
        return context[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise _reject(node)


def _evaluate_node(node: ast.AST, context: dict) -> bool:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, context)
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.And):
            raise ValueError("only and is supported")
        results = [_evaluate_node(operand, context) for operand in node.values]
        return bool(all(results))
    if isinstance(node, ast.Compare):
        left = _resolve_operand(node.left, context)
        right = _resolve_operand(node.comparators[0], context)
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            result = left == right
        elif isinstance(op, ast.NotEq):
            result = left != right
        elif isinstance(op, ast.GtE):
            result = left >= right
        elif isinstance(op, ast.LtE):
            result = left <= right
        elif isinstance(op, ast.Gt):
            result = left > right
        elif isinstance(op, ast.Lt):
            result = left < right
        else:
            raise ValueError(
                f"comparison operator {type(op).__name__} is not allowed"
            )
        return bool(result)
    raise _reject(node)


def evaluate_condition(condition_ast: object, context: dict) -> bool:
    """Evaluate a parsed condition against a decision-time context."""
    if not isinstance(condition_ast, ast.AST):
        raise ValueError("condition must come from parse_condition")
    return _evaluate_node(condition_ast, context)


def load_policy_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PolicyConfig:
    """Load and validate the frozen business-rules configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    stop_precedence = raw["rule_resolution"]["stop_precedence"]
    if type(stop_precedence) is not bool:
        raise ValueError("rule_resolution.stop_precedence must be a boolean")

    vocabulary = frozenset(raw["action_vocabulary"])
    if vocabulary != CANONICAL_ACTIONS:
        raise ValueError(
            "action_vocabulary must equal the canonical set "
            f"{sorted(CANONICAL_ACTIONS)}"
        )

    seen_ids: set[str] = set()
    rules: list[PolicyRule] = []
    for entry in raw["rules"]:
        rule_id = entry["id"]
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError(f"rule id must be a non-empty string, got {rule_id!r}")
        if rule_id in seen_ids:
            raise ValueError(f"duplicate rule id: {rule_id}")
        seen_ids.add(rule_id)
        priority = entry["priority"]
        if type(priority) is not int:
            raise ValueError(f"rule {rule_id}: priority must be an integer")
        action = entry["action"]
        if action not in CANONICAL_ACTIONS:
            raise ValueError(
                f"rule {rule_id}: action {action!r} is outside the canonical vocabulary"
            )
        condition_text = entry["condition"]
        enabled = entry.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError(f"rule {rule_id}: enabled must be a boolean")
        rules.append(
            PolicyRule(
                id=rule_id,
                name=entry["name"],
                priority=priority,
                action=action,
                reason=entry["reason"],
                condition_ast=parse_condition(condition_text),
                condition_text=condition_text,
                enabled=enabled,
            )
        )

    return PolicyConfig(
        version=raw["version"],
        stop_precedence=stop_precedence,
        rules=tuple(rules),
        canonical_actions=vocabulary,
    )


def _select_by_precedence(candidates: list[PolicyRule]) -> PolicyRule:
    return min(candidates, key=lambda rule: (-rule.priority, rule.id))


def decide_action(context: dict, policy: PolicyConfig) -> PolicyDecision:
    """Authorize exactly one action for a decision-time context.

    Deterministic precedence: every enabled rule is evaluated in config
    order (a missing context column raises KeyError, never a silent False);
    among matched rules the highest priority wins and equal priorities break
    to the lowest rule id. When ``stop_precedence`` is set and any matched
    rule authorizes STOP, a matched STOP rule wins even against a strictly
    higher positive priority. If nothing matches, the residual default
    ``RESIDUAL_DEFAULT_ACTION`` is authorized. Disabled rules are never
    evaluated but still appear in ``evaluated_rules`` as ``(id, False)``.
    """
    evaluated_rules: list[tuple[str, bool]] = []
    matched: list[PolicyRule] = []
    for rule in policy.rules:
        if not rule.enabled:
            evaluated_rules.append((rule.id, False))
            continue
        did_match = evaluate_condition(rule.condition_ast, context)
        evaluated_rules.append((rule.id, did_match))
        if did_match:
            matched.append(rule)

    candidates = matched
    if policy.stop_precedence:
        stop_matches = [rule for rule in matched if rule.action == "STOP"]
        if stop_matches:
            candidates = stop_matches

    if candidates:
        winner = _select_by_precedence(candidates)
        decision = PolicyDecision(
            authorized_action=winner.action,
            matched_rule_id=winner.id,
            matched_rule_name=winner.name,
            priority=winner.priority,
            reason=winner.reason,
            is_stop=winner.action == "STOP",
            evaluated_rules=tuple(evaluated_rules),
        )
    else:
        decision = PolicyDecision(
            authorized_action=RESIDUAL_DEFAULT_ACTION,
            matched_rule_id=None,
            matched_rule_name=None,
            priority=None,
            reason=(
                f"Residual default {RESIDUAL_DEFAULT_ACTION} applied because "
                "no policy rule matched this case."
            ),
            is_stop=False,
            evaluated_rules=tuple(evaluated_rules),
        )

    if decision.authorized_action not in CANONICAL_ACTIONS:
        raise ValueError(
            f"authorized action {decision.authorized_action!r} is outside the "
            f"canonical vocabulary {sorted(CANONICAL_ACTIONS)}"
        )
    return decision
