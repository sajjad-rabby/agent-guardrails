"""Policy engine: declarative allow/deny rules evaluated against tool calls."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_OPERATORS = ("contains", "equals", "matches", "starts_with")


class PolicyError(ValueError):
    """Raised when a policy definition is malformed."""


@dataclass
class Condition:
    """A single predicate over the arguments of a tool call."""

    arg: str
    operator: str
    value: str

    def check(self, args: Dict[str, Any]) -> bool:
        """Return True when the condition holds for the given arguments."""
        if self.operator not in _OPERATORS:
            raise PolicyError(
                "unknown operator {!r}; expected one of {}".format(
                    self.operator, ", ".join(_OPERATORS)
                )
            )
        actual = args.get(self.arg)
        if not isinstance(actual, str):
            return False
        if self.operator == "contains":
            return self.value in actual
        if self.operator == "equals":
            return actual == self.value
        if self.operator == "starts_with":
            return actual.startswith(self.value)
        try:
            pattern = re.compile(self.value)
        except re.error as exc:
            raise PolicyError("invalid regex {!r}: {}".format(self.value, exc))
        return pattern.search(actual) is not None

    @classmethod
    def from_json(cls, data: Any) -> "Condition":
        """Build a Condition from a JSON-compatible mapping."""
        if not isinstance(data, dict):
            raise PolicyError("condition must be an object, got {}".format(type(data).__name__))
        arg = data.get("arg")
        operator = data.get("operator")
        value = data.get("value")
        if not isinstance(arg, str) or not arg:
            raise PolicyError("condition.arg must be a non-empty string")
        if not isinstance(operator, str) or operator not in _OPERATORS:
            raise PolicyError(
                "condition.operator must be one of {}".format(", ".join(_OPERATORS))
            )
        if not isinstance(value, str):
            raise PolicyError("condition.value must be a string")
        condition = cls(arg=arg, operator=operator, value=value)
        if operator == "matches":
            try:
                re.compile(value)
            except re.error as exc:
                raise PolicyError("invalid regex in condition.value: {}".format(exc))
        return condition


@dataclass
class Rule:
    """An allow or deny rule scoped to one tool (or every tool via '*')."""

    name: str
    tool: str
    effect: str
    conditions: Tuple[Condition, ...]
    reason: str = ""

    def matches(self, tool: str, args: Dict[str, Any]) -> bool:
        """Return True when this rule applies to the given tool call."""
        if self.tool != "*" and self.tool != tool:
            return False
        return all(condition.check(args) for condition in self.conditions)

    @classmethod
    def from_json(cls, data: Any, default_name: str) -> "Rule":
        """Build a Rule from a JSON-compatible mapping."""
        if not isinstance(data, dict):
            raise PolicyError("rule must be an object, got {}".format(type(data).__name__))
        tool = data.get("tool")
        effect = data.get("effect")
        name = data.get("name", default_name)
        reason = data.get("reason", "")
        conditions_raw = data.get("conditions", [])
        if not isinstance(tool, str) or not tool:
            raise PolicyError("rule.tool must be a non-empty string")
        if effect not in ("allow", "deny"):
            raise PolicyError("rule.effect must be 'allow' or 'deny'")
        if not isinstance(name, str) or not name:
            raise PolicyError("rule.name must be a non-empty string")
        if not isinstance(reason, str):
            raise PolicyError("rule.reason must be a string")
        if not isinstance(conditions_raw, list):
            raise PolicyError("rule.conditions must be a list")
        conditions = tuple(Condition.from_json(item) for item in conditions_raw)
        return cls(name=name, tool=tool, effect=effect, conditions=conditions, reason=reason)


@dataclass
class Decision:
    """The outcome of evaluating a policy against a tool call."""

    allowed: bool
    reason: str
    matched_rule: Optional[str]


class Policy:
    """An ordered list of rules; the first matching rule decides the call."""

    def __init__(self, rules: Tuple[Rule, ...]):
        self._rules = rules

    @property
    def rules(self) -> Tuple[Rule, ...]:
        """The rules in evaluation order."""
        return self._rules

    def decide(self, tool: str, args: Dict[str, Any]) -> Decision:
        """Evaluate the policy for a tool call; unmatched calls are denied."""
        for rule in self._rules:
            if rule.matches(tool, args):
                allowed = rule.effect == "allow"
                reason = rule.reason or "{} by rule '{}'".format(rule.effect, rule.name)
                return Decision(allowed=allowed, reason=reason, matched_rule=rule.name)
        return Decision(
            allowed=False,
            reason="no matching policy rule (default deny)",
            matched_rule=None,
        )

    @classmethod
    def from_json(cls, data: Any) -> "Policy":
        """Build a Policy from a JSON-compatible mapping."""
        if not isinstance(data, dict):
            raise PolicyError("policy must be an object, got {}".format(type(data).__name__))
        rules_raw = data.get("rules")
        if not isinstance(rules_raw, list):
            raise PolicyError("policy.rules must be a list")
        rules = []
        for index, item in enumerate(rules_raw):
            rules.append(Rule.from_json(item, "rule-{}".format(index + 1)))
        return cls(tuple(rules))
