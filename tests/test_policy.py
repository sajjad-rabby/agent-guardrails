"""Tests for policy rule parsing and evaluation."""
from __future__ import annotations

import pytest

from agent_guardrails.policy import Condition, Policy, PolicyError, Rule


def test_condition_equality():
    condition = Condition.from_json({"arg": "path", "operator": "equals", "value": "/tmp/x"})
    assert condition.check({"path": "/tmp/x"}) is True
    assert condition.check({"path": "/tmp/y"}) is False


def test_condition_contains_and_starts_with():
    assert Condition.from_json(
        {"arg": "cmd", "operator": "contains", "value": "rm"}
    ).check({"cmd": "rm -rf /"})
    assert Condition.from_json(
        {"arg": "cmd", "operator": "starts_with", "value": "git"}
    ).check({"cmd": "git status"})


def test_condition_matches_regex():
    condition = Condition.from_json(
        {"arg": "expr", "operator": "matches", "value": r"^[0-9+\-*/() ]+$"}
    )
    assert condition.check({"expr": "1+2"}) is True
    assert condition.check({"expr": "rm -rf"}) is False


def test_condition_missing_arg_is_false():
    condition = Condition.from_json({"arg": "nope", "operator": "equals", "value": "1"})
    assert condition.check({"other": 1}) is False


def test_condition_from_json_rejects_unknown_operator():
    with pytest.raises(PolicyError):
        Condition.from_json({"arg": "a", "operator": "wat", "value": "1"})


def test_condition_from_json_requires_fields():
    with pytest.raises(PolicyError):
        Condition.from_json({"operator": "equals", "value": "1"})
    with pytest.raises(PolicyError):
        Condition.from_json({"arg": "a", "operator": "equals"})


def test_rule_matches_only_named_tool():
    rule = Rule.from_json({"tool": "calculator", "effect": "deny", "reason": "no"}, "r1")
    assert rule.matches("calculator", {}) is True
    assert rule.matches("write_file", {}) is False


def test_rule_requires_all_conditions():
    rule = Rule.from_json(
        {
            "tool": "calculator",
            "effect": "deny",
            "conditions": [
                {"arg": "expression", "operator": "contains", "value": "__"}
            ],
        },
        "r1",
    )
    assert rule.matches("calculator", {"expression": "__import__"}) is True
    assert rule.matches("calculator", {"expression": "1+1"}) is False


def test_rule_from_json_rejects_bad_effect():
    with pytest.raises(PolicyError):
        Rule.from_json({"tool": "calculator", "effect": "maybe"}, "r1")


def test_rule_from_json_rejects_missing_tool():
    with pytest.raises(PolicyError):
        Rule.from_json({"effect": "allow"}, "r1")


def test_policy_is_default_deny():
    """An unmatched call is denied: configuring a policy means opting into
    default-deny, so a forgotten rule fails closed rather than open."""
    decision = Policy(()).decide("calculator", {"expression": "1+1"})
    assert decision.allowed is False
    assert decision.matched_rule is None
    assert "default deny" in decision.reason


def test_deny_rule_blocks_matching_call():
    policy = Policy.from_json(
        {"rules": [{"tool": "write_file", "effect": "deny", "reason": "writes are blocked"}]}
    )
    decision = policy.decide("write_file", {"path": "a.txt", "content": "x"})
    assert decision.allowed is False
    assert decision.reason == "writes are blocked"
    assert decision.matched_rule is not None


def test_allow_rule_takes_precedence_when_matched():
    policy = Policy.from_json(
        {
            "rules": [
                {
                    "tool": "calculator",
                    "effect": "allow",
                    "reason": "safe arithmetic",
                    "conditions": [
                        {"arg": "expression", "operator": "matches", "value": "^[0-9]+$"}
                    ],
                },
                {"tool": "calculator", "effect": "deny", "reason": "catch-all"},
            ]
        }
    )
    allowed = policy.decide("calculator", {"expression": "42"})
    assert allowed.allowed is True
    assert allowed.reason == "safe arithmetic"

    denied = policy.decide("calculator", {"expression": "2+2"})
    assert denied.allowed is False
    assert denied.reason == "catch-all"


def test_first_matching_rule_wins():
    policy = Policy.from_json(
        {
            "rules": [
                {"tool": "calculator", "effect": "deny", "reason": "first"},
                {"tool": "calculator", "effect": "allow", "reason": "second"},
            ]
        }
    )
    assert policy.decide("calculator", {}).reason == "first"


def test_policy_from_json_rejects_non_object():
    with pytest.raises(PolicyError):
        Policy.from_json(["not", "an", "object"])


def test_policy_from_json_rejects_non_list_rules():
    with pytest.raises(PolicyError):
        Policy.from_json({"rules": {"tool": "x"}})


def test_policy_exposes_rules_tuple():
    policy = Policy.from_json({"rules": [{"tool": "calculator", "effect": "allow"}]})
    assert isinstance(policy.rules, tuple)
    assert policy.rules[0].tool == "calculator"
