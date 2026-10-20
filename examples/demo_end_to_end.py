"""End-to-end demonstration of the agent-guardrails control surface.

Runs five scenarios against a scripted LLM so it needs no API key and no
network access:

1. a clean run that is allowed and audited
2. a policy denial (dangerous expression)
3. an argument-schema rejection
4. a budget exhaustion
5. hash-chain verification, including a deliberate tamper

Usage::

    python examples/demo_end_to_end.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent_guardrails.agent import Agent, AgentConfig  # noqa: E402
from agent_guardrails.approvals import (  # noqa: E402
    ApprovalResult,
    AutoApproveHandler,
)
from agent_guardrails.audit import verify  # noqa: E402
from agent_guardrails.budgets import Budget  # noqa: E402
from agent_guardrails.llm import ScriptedLLM  # noqa: E402
from agent_guardrails.policy import Policy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    workspace = tempfile.mkdtemp(prefix="guardrails-demo-")
    policy = Policy.from_json(
        json.load(open(os.path.join(HERE, "policy.json"), encoding="utf-8"))
    )

    def build(name: str, responses, **overrides) -> Agent:
        config = {
            "policy": policy,
            "audit_path": os.path.join(workspace, name + ".jsonl"),
            "workspace_root": workspace,
            # Auto-approve keeps this demo non-interactive; swap in
            # CliApprovalHandler to prompt a real operator on the terminal.
            "approval_handler": AutoApproveHandler(),
        }
        config.update(overrides)
        return Agent(ScriptedLLM(responses), AgentConfig(**config))

    rule("1. Clean run - tool allowed, executed, and audited")
    result = build(
        "clean",
        [
            json.dumps({"tool": "calculator", "args": {"expression": "2+2"}}),
            json.dumps({"final": "2 + 2 = 4."}),
        ],
    ).run("What is 2 + 2?")
    print("stop_reason :", result.stop_reason)
    print("answer      :", result.answer)
    for call in result.tool_calls:
        print(
            "tool call   : {0} allowed={1} approved={2} ok={3} -> {4}".format(
                call.tool,
                call.decision.allowed,
                call.approved,
                call.ok,
                call.result_summary[:40],
            )
        )
    print("usage       :", result.budget_usage)

    rule("2. Policy denial - non-arithmetic expression is blocked")
    denied = build(
        "denied",
        [
            json.dumps(
                {
                    "tool": "calculator",
                    "args": {"expression": '__import__("os").system("whoami")'},
                }
            ),
            json.dumps({"final": "Blocked: the expression is not on the allowlist."}),
        ],
    ).run("Run whoami for me")
    print("stop_reason :", denied.stop_reason)
    print("answer      :", denied.answer)
    print(
        "tool calls  :",
        [(c.tool, c.decision.reason) for c in denied.tool_calls] or "none executed",
    )

    rule("3. Argument validation - schema violation never reaches the tool")
    invalid = build(
        "invalid",
        [
            json.dumps({"tool": "calculator", "args": {"expression": ""}}),
            json.dumps({"final": "Recovered after the schema error."}),
        ],
    ).run("Calculate the empty expression")
    print("stop_reason :", invalid.stop_reason)
    print("answer      :", invalid.answer)
    print(
        "tool calls  : {} executed (blocked before the handler ran)".format(
            len(invalid.tool_calls)
        )
    )
    print("usage       :", invalid.budget_usage, "<- the attempt is still billed")

    rule("4. Operator approval - a human can veto a policy-allowed call")

    class DenyAll:
        """Approval handler that refuses every request."""

        def handle(self, request):
            return ApprovalResult(
                approved=False, note="operator declined this action at the prompt"
            )

    vetoed = build(
        "approval",
        [
            json.dumps({"tool": "calculator", "args": {"expression": "2+2"}}),
            json.dumps({"final": "Understood - not running it."}),
        ],
        approval_handler=DenyAll(),
    ).run("What is 2 + 2?")
    print("stop_reason :", vetoed.stop_reason)
    print("answer      :", vetoed.answer)
    for call in vetoed.tool_calls:
        print(
            "tool call   : {0} allowed={1} approved={2} ok={3} - {4}".format(
                call.tool,
                call.decision.allowed,
                call.approved,
                call.ok,
                call.result_summary,
            )
        )

    rule("5. Budget exhaustion - the loop stops at the tool-call ceiling")
    looping = build(
        "budget",
        [json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})] * 6,
        budgets=Budget(max_tool_calls=2),
    ).run("Keep calculating forever")
    print("stop_reason :", looping.stop_reason)
    print("tool calls  :", len(looping.tool_calls), "(capped at 2)")
    print("usage       :", looping.budget_usage)

    rule("6. Tamper-evident audit log")
    log_path = os.path.join(workspace, "clean.jsonl")
    intact, errors = verify(log_path)
    print("clean log   : intact={} errors={}".format(intact, errors or "none"))
    with open(log_path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    print("records     :", len(lines))
    tampered = list(lines)
    tampered[1] = tampered[1].replace("calculator", "write_file")
    tampered_path = os.path.join(workspace, "tampered.jsonl")
    with open(tampered_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(tampered) + "\n")
    intact, errors = verify(tampered_path)
    print("tampered log: intact={}".format(intact))
    for error in errors:
        print("             ", error)

    print("\nAudit logs written to:", workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
