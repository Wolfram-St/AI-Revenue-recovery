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
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import yaml

CANONICAL_ACTIONS: frozenset[str] = frozenset(
    {"RETRY_NOW", "RETRY_LATER", "REQUEST_UPDATE", "HUMAN_REVIEW", "STOP"}
)

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
        rules.append(
            PolicyRule(
                id=rule_id,
                name=entry["name"],
                priority=priority,
                action=action,
                reason=entry["reason"],
                condition_ast=parse_condition(condition_text),
                condition_text=condition_text,
            )
        )

    return PolicyConfig(
        version=raw["version"],
        stop_precedence=stop_precedence,
        rules=tuple(rules),
        canonical_actions=vocabulary,
    )
