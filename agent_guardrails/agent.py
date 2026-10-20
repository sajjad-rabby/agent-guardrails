"""Autonomous agent loop with policy enforcement, HITL approvals, and budgets."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent_guardrails.approvals import ApprovalHandler, ApprovalRequest, AutoApproveHandler
from agent_guardrails.audit import AuditLog
from agent_guardrails.budgets import Budget, BudgetExceeded, BudgetTracker
from agent_guardrails.llm import LLMClient, LLMError
from agent_guardrails.policy import Decision, Policy
from agent_guardrails.schema import validate
from agent_guardrails.tools import (
    ToolError,
    ToolInputError,
    ToolRegistry,
    default_tools,
    validate_args,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous agent that may use tools to complete a task.\n"
    "You MUST respond with EXACTLY ONE JSON object and nothing else:\n"
    'To call a tool: {"tool": "<tool-name>", "args": {<arguments>}}\n'
    'When the task is complete: {"final": "<your final answer>"}\n'
    "Do not wrap the JSON in markdown fences. Do not add commentary."
)

DEFAULT_FINAL_ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["final"],
    "properties": {"final": {"type": "string"}},
}

_STOP_REASONS = {
    "final_answer",
    "max_steps",
    "timeout",
    "max_retries",
    "budget_exceeded",
    "llm_error",
}


@dataclass
class ToolCallRecord:
    """Record of one attempted tool execution inside a run."""

    tool: str
    args: Dict[str, Any]
    decision: Optional[Decision]
    approved: bool
    duration_ms: float
    ok: bool
    result_summary: str


@dataclass
class RunResult:
    """Outcome of a completed agent run."""

    answer: Optional[str]
    tool_calls: List[ToolCallRecord]
    audit_path: str
    budget_usage: Dict[str, Any]
    stop_reason: str


@dataclass
class AgentConfig:

    """Configuration for an Agent run.

    Attributes:
        policy: Optional policy; when None every tool call is allowed.
        audit_path: Filesystem path for the hash-chained audit log.
        workspace_root: Root directory the built-in file tools may access.
        budgets: Budget limits enforced during the run.
        approval_handler: Human-in-the-loop handler consulted before execution.
        final_answer_schema: JSON schema applied to the assistant's whole JSON
            object. It must describe the envelope: require the ``final`` key and
            constrain its value; the value of ``final`` then becomes
            ``RunResult.answer``. See ``DEFAULT_FINAL_ANSWER_SCHEMA``.
        max_steps: Maximum number of assistant turns in the loop.
        max_retries: Maximum number of counted retries after bad LLM output.
        system_prompt: Overrides DEFAULT_SYSTEM_PROMPT when non-empty.
    """

    policy: Optional[Policy] = None
    audit_path: str = "audit.jsonl"
    workspace_root: str = "."
    budgets: Budget = field(default_factory=Budget)
    approval_handler: ApprovalHandler = field(default_factory=AutoApproveHandler)
    final_answer_schema: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_FINAL_ANSWER_SCHEMA)
    )
    max_steps: int = 100
    max_retries: int = 3
    system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT


class Agent:
    """Runs an LLM-driven tool loop with guardrails around every step."""

    def __init__(self, llm: LLMClient, config: Optional[AgentConfig] = None) -> None:
        self.llm = llm
        self.config = config if config is not None else AgentConfig()
        self.tools = ToolRegistry()
        for tool in default_tools(roots=[self.config.workspace_root]).list():
            self.tools.register(tool)
        self.audit = AuditLog(self.config.audit_path)

    def run(self, task: str) -> RunResult:
        """Execute the task, auditing start, every step, and finish."""
        tracker = BudgetTracker(self.config.budgets)
        self.audit.append("run_started", {"task": task})
        self._tool_calls: List[ToolCallRecord] = []
        self._retries = 0
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": task},
        ]
        stop_reason, answer = self._loop(messages, tracker)
        usage = tracker.usage()
        self.audit.append("run_finished", {"stop_reason": stop_reason, "usage": usage})
        return RunResult(
            answer=answer,
            tool_calls=self._tool_calls,
            audit_path=self.audit.path,
            budget_usage=usage,
            stop_reason=stop_reason,
        )

    def _loop(
        self, messages: List[Dict[str, str]], tracker: BudgetTracker
    ) -> Tuple[str, Optional[str]]:
        stop_reason = "final_answer"
        answer: Optional[str] = None
        try:
            for _ in range(self.config.max_steps):
                if tracker.timed_out():
                    stop_reason = "timeout"
                    break
                if len(self._tool_calls) >= self.config.max_steps:
                    stop_reason = "max_steps"
                    break
                try:
                    raw = self.llm.complete(messages)
                except LLMError as exc:
                    self.audit.append("llm_error", {"error": str(exc)})
                    stop_reason = "llm_error"
                    break
                parsed = self._extract_json(raw)
                if parsed is None:
                    feedback, exhausted = self._count_retry(
                        "Output was not a valid JSON object. Respond with exactly one JSON object."
                    )
                    if exhausted:
                        stop_reason = "max_retries"
                        break
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": feedback})
                    continue
                if "final" in parsed:
                    final = parsed.get("final")
                    errors = validate(parsed, self.config.final_answer_schema)
                    if errors:
                        feedback, exhausted = self._count_retry(
                            "The final answer does not satisfy the required schema: "
                            + "; ".join(errors)
                        )
                        if exhausted:
                            stop_reason = "max_retries"
                            break
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({"role": "user", "content": feedback})
                        continue
                    answer = final
                    self.audit.append("final_answer", {"answer": answer})
                    break
                tool_name = parsed.get("tool")
                tool_args = parsed.get("args", {})
                if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
                    feedback, exhausted = self._count_retry(
                        'Each tool call must be {"tool": "<name>", "args": {<object>}}.'
                    )
                    if exhausted:
                        stop_reason = "max_retries"
                        break
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": feedback})
                    continue
                record, feedback = self._execute_tool(tool_name, tool_args, tracker)
                if record is not None:
                    self._tool_calls.append(record)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": feedback})
        except BudgetExceeded as exc:
            self.audit.append(
                "budget_exceeded", {"field": exc.field, "message": str(exc)}
            )
            stop_reason = "budget_exceeded"
        return stop_reason, answer

    def _count_retry(self, feedback: str) -> Tuple[str, bool]:
        """Increment the retry counter and report whether retries are exhausted."""
        self._retries += 1
        if self._retries > self.config.max_retries:
            return feedback, True
        return feedback, False

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
        """Parse one JSON object out of raw LLM text, tolerating fences."""
        text = raw.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _execute_tool(
        self, tool_name: str, tool_args: Dict[str, Any], tracker: BudgetTracker
    ) -> Tuple[Optional[ToolCallRecord], str]:
        """Validate, approve, and execute one tool call under the guardrails."""
        spec = self.tools.get(tool_name)
        # Charge the budget for the attempt, whatever happens next. A call that
        # is denied by policy or approval still costs an LLM round-trip, and an
        # agent that keeps re-requesting a denied tool must stay bounded.
        cost = spec.cost_per_call if spec is not None else 0.0
        tracker.check_next_call(cost=cost)
        tracker.record_tool_call(cost=cost)
        if spec is None:
            self.audit.append("unknown_tool", {"tool": tool_name, "args": tool_args})
            return None, "Unknown tool: {0}. Available tools: {1}.".format(
                tool_name, ", ".join(self.tools.names())
            )
        errors = validate_args(spec, tool_args)
        if errors:
            self.audit.append(
                "invalid_args", {"tool": tool_name, "args": tool_args, "errors": errors}
            )
            return (
                None,
                "Arguments for {0} are invalid: {1}.".format(tool_name, "; ".join(errors)),
            )
        if self.config.policy is not None:
            decision = self.config.policy.decide(tool_name, tool_args)
        else:
            decision = Decision(allowed=True, reason="no policy configured", matched_rule=None)
        if not decision.allowed:
            self.audit.append(
                "policy_denied",
                {"tool": tool_name, "args": tool_args, "decision": decision.reason},
            )
            return (
                None,
                "Policy denied the tool call {0}: {1}.".format(tool_name, decision.reason),
            )
        approval = self.config.approval_handler.handle(
            ApprovalRequest(tool=tool_name, args=tool_args, reason=decision.reason)
        )
        if not approval.approved:
            record = ToolCallRecord(
                tool=tool_name,
                args=tool_args,
                decision=decision,
                approved=False,
                duration_ms=0.0,
                ok=False,
                result_summary=approval.note,
            )
            self.audit.append(
                "tool_call",
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "decision": decision.reason,
                    "approved": False,
                    "ok": False,
                    "duration_ms": 0.0,
                    "result_summary": approval.note,
                },
            )
            return record, "Approval was denied for {0}: {1}.".format(
                tool_name, approval.note
            )
        started = time.time()
        ok = True
        result_summary: str
        try:
            result = spec.handler(**tool_args)
            result_summary = self._summarize(result)
        except ToolError as exc:
            ok = False
            result_summary = str(exc)
        duration_ms = (time.time() - started) * 1000.0
        record = ToolCallRecord(
            tool=tool_name,
            args=tool_args,
            decision=decision,
            approved=True,
            duration_ms=duration_ms,
            ok=ok,
            result_summary=result_summary,
        )
        self.audit.append(
            "tool_call",
            {
                "tool": tool_name,
                "args": tool_args,
                "decision": decision.reason,
                "approved": True,
                "ok": ok,
                "duration_ms": duration_ms,
                "result_summary": result_summary,
            },
        )
        if ok:
            feedback = "Tool {0} executed, result: {1}".format(tool_name, result_summary)
        else:
            feedback = "Tool {0} failed: {1}".format(tool_name, result_summary)
        return record, feedback

    @staticmethod
    def _summarize(result: Any, limit: int = 500) -> str:
        """Render a tool result as a bounded JSON string for the LLM."""
        try:
            summary = json.dumps(result, sort_keys=True, default=str)
        except (TypeError, ValueError):
            summary = str(result)
        if len(summary) > limit:
            summary = summary[:limit] + "... [truncated]"
        return summary
