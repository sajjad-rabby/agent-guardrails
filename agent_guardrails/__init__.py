"""agent-guardrails - a stdlib-first safety and governance layer for LLM agents.

The package layers safety controls around tool-calling LLM agents:
- schema: a minimal JSON-schema validator for tool arguments
- tools: tool definitions and a built-in toolset (calculator, scoped file I/O, HTTP GET, time)
- policy: allow/deny rules evaluated before any tool runs
- approvals: human-in-the-loop confirmation for sensitive tools
- audit: hash-chained, tamper-evident JSONL audit log
- budgets: per-run caps on tool calls, spend and wall-clock time
- llm: OpenAI-compatible chat-completions client and offline test double
- agent: the guarded execution loop that ties everything together
- cli: the ``guardrails`` command line entry point

Everything runs on the Python standard library only (Python 3.9+). pytest is
optional and used only for the test suite.
"""

from agent_guardrails.agent import Agent, AgentConfig, RunResult, ToolCallRecord
from agent_guardrails.approvals import (
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResult,
    AutoApproveHandler,
    CliApprovalHandler,
)
from agent_guardrails.audit import AuditLog, json_default
from agent_guardrails.budgets import Budget, BudgetExceeded, BudgetTracker, BudgetUsage
from agent_guardrails.llm import LLMClient, LLMError, OpenAICompatLLM, ScriptedLLM
from agent_guardrails.policy import Decision, Policy, PolicyError, Rule
from agent_guardrails.schema import validate
from agent_guardrails.tools import Tool, ToolRegistry, default_tools

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "RunResult",
    "ToolCallRecord",
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalResult",
    "AutoApproveHandler",
    "CliApprovalHandler",
    "AuditLog",
    "json_default",
    "Budget",
    "BudgetExceeded",
    "BudgetTracker",
    "BudgetUsage",
    "LLMClient",
    "LLMError",
    "OpenAICompatLLM",
    "ScriptedLLM",
    "Decision",
    "Policy",
    "PolicyError",
    "Rule",
    "Tool",
    "ToolRegistry",
    "default_tools",
    "validate",
    "__version__",
]
