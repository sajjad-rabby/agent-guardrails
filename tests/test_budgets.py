"""Tests for budget enforcement."""
from __future__ import annotations

import time

import pytest

from agent_guardrails.budgets import Budget, BudgetExceeded, BudgetTracker


def test_defaults_are_permissive_enough_to_be_useful():
    budget = Budget()
    assert budget.max_tool_calls > 0
    assert budget.max_cost_usd > 0
    assert budget.timeout_seconds > 0


def test_usage_starts_at_zero():
    tracker = BudgetTracker(Budget())
    usage = tracker.usage()
    assert usage.tool_calls == 0
    assert usage.cost_usd == 0.0
    assert usage.elapsed_seconds >= 0.0


def test_record_tool_call_accumulates_count_and_cost():
    tracker = BudgetTracker(Budget(max_tool_calls=100, max_cost_usd=100.0))
    tracker.record_tool_call(cost=0.25)
    tracker.record_tool_call(cost=0.75)
    usage = tracker.usage()
    assert usage.tool_calls == 2
    assert usage.cost_usd == 1.0


def test_check_next_call_allows_up_to_the_ceiling():
    tracker = BudgetTracker(Budget(max_tool_calls=2, max_cost_usd=100.0))
    tracker.check_next_call()
    tracker.record_tool_call()
    tracker.check_next_call()
    tracker.record_tool_call()
    with pytest.raises(BudgetExceeded):
        tracker.check_next_call()


def test_budget_exceeded_carries_the_field_name():
    tracker = BudgetTracker(Budget(max_tool_calls=0, max_cost_usd=100.0))
    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check_next_call()
    assert excinfo.value.field == "max_tool_calls"
    assert "max_tool_calls" in str(excinfo.value)


def test_cost_ceiling_is_enforced_with_tolerance():
    tracker = BudgetTracker(Budget(max_tool_calls=100, max_cost_usd=1.0))
    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check_next_call(cost=5.0)
    assert excinfo.value.field == "max_cost_usd"


def test_floating_point_noise_is_tolerated():
    """Three 0.1 charges must not trip a 0.3 ceiling through representation error."""
    tracker = BudgetTracker(Budget(max_tool_calls=100, max_cost_usd=0.3))
    for _ in range(3):
        tracker.check_next_call(cost=0.1)
        tracker.record_tool_call(cost=0.1)
    assert tracker.usage().tool_calls == 3


def test_timeout_is_detected():
    tracker = BudgetTracker(Budget(timeout_seconds=0.05))
    time.sleep(0.08)
    assert tracker.timed_out() is True
    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check_next_call()
    assert excinfo.value.field == "timeout_seconds"


def test_not_timed_out_immediately():
    assert BudgetTracker(Budget(timeout_seconds=300.0)).timed_out() is False


def test_tracker_exposes_configured_budget():
    budget = Budget(max_tool_calls=7)
    assert BudgetTracker(budget).budget is budget


def test_exceeded_flag_is_recorded():
    tracker = BudgetTracker(Budget(max_tool_calls=0, max_cost_usd=100.0))
    with pytest.raises(BudgetExceeded):
        tracker.check_next_call()
    assert tracker.exceeded == "max_tool_calls"


def test_usage_is_rounded_for_stable_reporting():
    tracker = BudgetTracker(Budget())
    tracker.record_tool_call(cost=1 / 3)
    usage = tracker.usage()
    assert usage.cost_usd == round(1 / 3, 6)
