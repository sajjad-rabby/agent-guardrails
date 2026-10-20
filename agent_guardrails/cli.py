"""Command-line interface for agent-guardrails.

Subcommands:
- ``policy-check`` evaluate a tool call against a policy
- ``validate``     validate a JSON instance against a schema
- ``verify``       re-derive an audit log's hash chain
- ``tools``        list the built-in tools
- ``run``          run a task through the guarded agent loop

``run`` needs an LLM. ``--llm scripted`` replays a JSON file of canned responses so
the guardrails can be exercised without a network call or
an API key; ``--llm openai`` talks to any OpenAI-compatible endpoint.

Subcommands
-----------
``policy-check``  Evaluate one tool call against a policy file.
``validate``      Validate a JSON instance against a JSON schema.
``verify``        Re-derive an audit log's hash chain.
``tools``         List the built-in tools and their schemas.
``run``           Run a task through a guarded agent loop.

The ``run`` subcommand needs an LLM. ``--llm scripted`` replays a JSON file of
canned responses so the guardrails can be exercised without a network call or
an API key; ``--llm openai`` talks to any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from agent_guardrails.audit import AuditLog, json_default, verify
from agent_guardrails.budgets import Budget
from agent_guardrails.policy import Policy
from agent_guardrails.schema import validate
from agent_guardrails.tools import default_tools

EXIT_OK = 0
EXIT_DENIED = 1
EXIT_INVALID = 2
EXIT_ERROR = 3


def _load_json(path: str) -> Any:
    """Read and parse a JSON file, raising ``SystemExit`` with a clear message."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit("error: file not found: {}".format(path))
    except json.JSONDecodeError as exc:
        raise SystemExit("error: {} is not valid JSON: {}".format(path, exc))


def _emit(payload: Any) -> None:
    """Print a payload as JSON, encoding dataclasses and enums as plain data."""
    print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))


def _cmd_policy_check(args: argparse.Namespace) -> int:
    policy = Policy.from_json(_load_json(args.policy))
    call_args = _load_json(args.args) if args.args else {}
    if not isinstance(call_args, dict):
        raise SystemExit("error: --args must contain a JSON object")
    decision = policy.decide(args.tool, call_args)
    _emit(
        {
            "tool": args.tool,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "matched_rule": decision.matched_rule,
        }
    )
    return EXIT_OK if decision.allowed else EXIT_DENIED


def _cmd_validate(args: argparse.Namespace) -> int:
    schema = _load_json(args.schema)
    instance = _load_json(args.instance)
    errors = validate(instance, schema)
    _emit({"valid": not errors, "errors": errors})
    return EXIT_OK if not errors else EXIT_INVALID


def _cmd_verify(args: argparse.Namespace) -> int:
    intact, errors = verify(args.path)
    _emit(
        {
            "path": args.path,
            "intact": intact,
            "record_count": sum(1 for _ in open(args.path, encoding="utf-8")) if os.path.exists(args.path) else 0,
            "errors": errors,
        }
    )
    return EXIT_OK if intact else EXIT_INVALID


def _cmd_tools(args: argparse.Namespace) -> int:
    roots = [args.root] if args.root else ["."]
    domains = [args.domain] if args.domain else []
    registry = default_tools(roots=roots, allowed_domains=domains)
    _emit({"tools": [tool.tool_spec() for tool in registry.list()]})
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one guarded task end to end and report the run record."""
    # Imported lazily so the other subcommands work without an LLM available.
    from agent_guardrails.agent import Agent, AgentConfig
    from agent_guardrails.llm import OpenAICompatLLM, ScriptedLLM

    if args.llm == "openai":
        if not args.base_url or not args.model:
            raise SystemExit("error: --llm openai requires --base-url and --model")
        llm: Any = OpenAICompatLLM(
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get("GUARDRAILS_API_KEY"),
        )
    else:
        if not args.responses:
            raise SystemExit("error: --llm scripted requires --responses")
        responses = _load_json(args.responses)
        if not isinstance(responses, list) or not all(
            isinstance(item, str) for item in responses
        ):
            raise SystemExit("error: --responses must be a JSON array of strings")
        llm = ScriptedLLM(list(responses))

    policy = Policy.from_json(_load_json(args.policy)) if args.policy else None
    budget = Budget(
        max_tool_calls=args.max_tool_calls,
        max_cost_usd=args.max_cost_usd,
        timeout_seconds=args.timeout,
    )
    agent = Agent(
        llm,
        AgentConfig(
            policy=policy,
            audit_path=args.audit,
            workspace_root=args.root or ".",
            budgets=budget,
        ),
    )
    result = agent.run(args.task)
    _emit(
        {
            "stop_reason": result.stop_reason,
            "answer": result.answer,
            "audit_path": result.audit_path,
            "budget_usage": result.budget_usage,
            "tool_calls": [
                {
                    "tool": call.tool,
                    "args": call.args,
                    "allowed": call.decision.allowed,
                    "approved": call.approved,
                    "ok": call.ok,
                    "result_summary": call.result_summary,
                }
                for call in result.tool_calls
            ],
        }
    )
    return EXIT_OK if result.stop_reason == "final_answer" else EXIT_DENIED


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``guardrails`` command."""
    parser = argparse.ArgumentParser(
        prog="guardrails",
        description="Policy, approval, budget and audit guardrails for LLM agents.",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("policy-check", help="evaluate a tool call against a policy")
    p.add_argument("--policy", required=True, help="policy JSON file")
    p.add_argument("--tool", required=True, help="tool name")
    p.add_argument("--args", help="JSON file holding the arguments object")
    p.set_defaults(func=_cmd_policy_check)

    p = sub.add_parser("validate", help="validate a JSON instance against a schema")
    p.add_argument("--schema", required=True, help="schema JSON file")
    p.add_argument("--instance", required=True, help="instance JSON file")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("verify", help="verify an audit log hash chain")
    p.add_argument("--path", required=True, help="audit JSONL file")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("tools", help="list built-in tools")
    p.add_argument("--root", help="allowed workspace root (default: .)")
    p.add_argument("--domain", action="append", help="allowed domain (repeatable)")
    p.set_defaults(func=_cmd_tools)

    p = sub.add_parser("run", help="run a task through the guarded agent loop")
    p.add_argument("--task", required=True, help="natural-language task")
    p.add_argument("--llm", choices=["scripted", "openai"], default="scripted")
    p.add_argument("--responses", help="JSON array of canned responses (scripted mode)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (openai mode)")
    p.add_argument("--model", help="model name (openai mode)")
    p.add_argument("--policy", help="policy JSON file")
    p.add_argument("--audit", default="audit.jsonl", help="audit log path")
    p.add_argument("--root", help="workspace root for file tools")
    p.add_argument("--max-tool-calls", type=int, default=25)
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=_cmd_run)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        return int(args.func(args))
    except SystemExit:
        raise
    except Exception as exc:  # surfaced as a clean CLI error, not a traceback
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
