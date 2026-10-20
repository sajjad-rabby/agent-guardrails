"""Cost and call-count budgets for agent runs."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

_COST_TOLERANCE = 1e-9


class BudgetExceeded(Exception):
    """Raised when a run would exceed its configured budget."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


@dataclass
class Budget:
    """Limits applied to a single agent run."""

    max_tool_calls: int = 25
    max_cost_usd: float = 1.0
    timeout_seconds: float = 300.0


@dataclass
class BudgetUsage:
    """Snapshot of consumption at a point in time."""

    tool_calls: int
    cost_usd: float
    elapsed_seconds: float


class BudgetTracker:
    """Tracks consumption against a Budget and refuses overrun calls."""

    def __init__(self, budget: Budget, start_time: Optional[float] = None):
        self._budget = budget
        self._start_time = start_time if start_time is not None else time.time()
        self._tool_calls = 0
        self._cost_usd = 0.0
        self.exceeded: Optional[str] = None

    @property
    def budget(self) -> Budget:
        """The configured budget."""
        return self._budget

    def _elapsed(self) -> float:
        return time.time() - self._start_time

    def timed_out(self) -> bool:
        """Return True when the run has exceeded its timeout."""
        return self._elapsed() > self._budget.timeout_seconds

    def check_next_call(self, cost: float = 0.0) -> None:
        """Raise BudgetExceeded if one more call (at the given cost) overruns."""
        if self._tool_calls + 1 > self._budget.max_tool_calls:
            message = "budget max_tool_calls={} reached".format(self._budget.max_tool_calls)
            self.exceeded = "max_tool_calls"
            raise BudgetExceeded("max_tool_calls", message)
        if self._cost_usd + cost > self._budget.max_cost_usd + _COST_TOLERANCE:
            message = "budget max_cost_usd={} would be exceeded".format(self._budget.max_cost_usd)
            self.exceeded = "max_cost_usd"
            raise BudgetExceeded("max_cost_usd", message)
        if self.timed_out():
            message = "budget timeout_seconds={} exceeded".format(self._budget.timeout_seconds)
            self.exceeded = "timeout_seconds"
            raise BudgetExceeded("timeout_seconds", message)

    def record_tool_call(self, cost: float = 0.0) -> None:
        """Record one completed tool call and its cost."""
        self._tool_calls += 1
        self._cost_usd += cost

    def usage(self) -> BudgetUsage:
        """Return the current consumption snapshot."""
        return BudgetUsage(
            tool_calls=self._tool_calls,
            cost_usd=round(self._cost_usd, 6),
            elapsed_seconds=round(self._elapsed(), 3),
        )
