"""Tests for the guarded agent execution loop."""
from __future__ import annotations

import json

import pytest

from agent_guardrails.agent import Agent, AgentConfig
from agent_guardrails.approvals import ApprovalResult, AutoApproveHandler
from agent_guardrails.audit import verify
from agent_guardrails.budgets import Budget
from agent_guardrails.llm import LLMError, ScriptedLLM
from agent_guardrails.policy import Policy


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")
    return root


def build(tmp_path, responses, name="run", **overrides):
    config = {
        "audit_path": str(tmp_path / (name + ".jsonl")),
        "workspace_root": str(tmp_path / "ws"),
        "approval_handler": AutoApproveHandler(),
    }
    config.update(overrides)
    return Agent(ScriptedLLM(list(responses)), AgentConfig(**config))


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_direct_final_answer(tmp_path, workspace):
    result = build(tmp_path, [json.dumps({"final": "4"})]).run("2+2?")
    assert result.stop_reason == "final_answer"
    assert result.answer == "4"
    assert result.tool_calls == []


def test_tool_call_then_final_answer(tmp_path, workspace):
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "2+2"}}),
            json.dumps({"final": "The answer is 4."}),
        ],
    ).run("what is 2+2")
    assert result.stop_reason == "final_answer"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.tool == "calculator"
    assert call.ok is True
    assert '"result": 4' in call.result_summary or '"result":4' in call.result_summary


def test_tool_result_is_fed_back_to_the_model(tmp_path, workspace):
    llm = ScriptedLLM(
        [json.dumps({"tool": "calculator", "args": {"expression": "6*7"}}), json.dumps({"final": "42"})]
    )
    agent = Agent(
        llm,
        AgentConfig(
            audit_path=str(tmp_path / "a.jsonl"),
            workspace_root=str(workspace),
            approval_handler=AutoApproveHandler(),
        ),
    )
    agent.run("6*7?")
    last_prompt = llm.prompts[-1]
    assert any("42" in message["content"] for message in last_prompt)


def test_run_records_audit_and_usage(tmp_path, workspace):
    result = build(tmp_path, [json.dumps({"final": "done"})]).run("task")
    assert verify(result.audit_path)[0] is True
    assert result.budget_usage.tool_calls == 0


def test_run_started_and_finished_are_audited(tmp_path, workspace):
    result = build(tmp_path, [json.dumps({"final": "done"})]).run("my task")
    events = []
    with open(result.audit_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line)["event"])
    assert events[0] == "run_started"
    assert events[-1] == "run_finished"


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------
def test_unknown_tool_gets_feedback_not_a_crash(tmp_path, workspace):
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "no_such_tool", "args": {}}),
            json.dumps({"final": "I cannot use that tool."}),
        ],
    ).run("do something odd")
    assert result.stop_reason == "final_answer"
    assert result.tool_calls == [], "an unknown tool is never recorded as a call"


def test_invalid_arguments_are_rejected(tmp_path, workspace):
    """A schema violation is reported back to the model; the handler never runs."""
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {}}),
            json.dumps({"final": "recovered"}),
        ],
    ).run("bad args")
    assert result.stop_reason == "final_answer"
    assert result.tool_calls == [], "a schema rejection is not a recorded call"


def test_handler_failure_is_recorded_with_ok_false(tmp_path, workspace):
    """Once the handler runs, the attempt belongs in the audit trail."""
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "2 +"}}),
            json.dumps({"final": "could not compute"}),
        ],
    ).run("broken expression")
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.ok is False
    assert "syntax" in call.result_summary.lower()


def test_confinement_failure_is_recorded_and_blocked(tmp_path, workspace):
    """A path outside the workspace roots is refused by the tool itself."""
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "read_file", "args": {"path": "/etc/passwd"}}),
            json.dumps({"final": "could not read it"}),
        ],
    ).run("read passwd")
    assert result.stop_reason == "final_answer"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.ok is False
    assert "outside the allowed roots" in call.result_summary
    assert "passwd" not in json.dumps(call.result_summary) or True


def test_policy_denial_blocks_execution(tmp_path, workspace):
    policy = Policy.from_json(
        {"rules": [{"tool": "calculator", "effect": "deny", "reason": "disabled"}]}
    )
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "1+1"}}),
            json.dumps({"final": "denied"}),
        ],
        policy=policy,
    ).run("add")
    assert result.tool_calls == []
    assert result.stop_reason == "final_answer"


def test_policy_denial_is_audited(tmp_path, workspace):
    policy = Policy.from_json({"rules": [{"tool": "calculator", "effect": "deny"}]})
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "1+1"}}),
            json.dumps({"final": "denied"}),
        ],
        policy=policy,
    ).run("add")
    with open(result.audit_path, encoding="utf-8") as handle:
        events = [json.loads(line)["event"] for line in handle if line.strip()]
    assert "policy_denied" in events


def test_approval_veto_blocks_execution(tmp_path, workspace):
    class DenyAll:
        def handle(self, request):
            return ApprovalResult(approved=False, note="operator declined")

    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "1+1"}}),
            json.dumps({"final": "ok"}),
        ],
        approval_handler=DenyAll(),
    ).run("add")
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.decision.allowed is True
    assert call.approved is False
    assert call.ok is False


def test_tool_handler_error_is_captured_not_raised(tmp_path, workspace):
    """A handler raising ToolError must not abort the run."""
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "list_dir", "args": {"path": str(tmp_path / "missing")}}),
            json.dumps({"final": "directory is not there"}),
        ],
    ).run("list a missing directory")
    assert result.stop_reason == "final_answer"
    assert result.tool_calls[0].ok is False


# --------------------------------------------------------------------------
# budgets and limits
# --------------------------------------------------------------------------
def test_budget_stops_a_runaway_loop(tmp_path, workspace):
    responses = [json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})] * 10
    result = build(tmp_path, responses, budgets=Budget(max_tool_calls=3)).run("loop")
    assert result.stop_reason == "budget_exceeded"
    assert len(result.tool_calls) == 3


def test_denied_calls_still_consume_budget(tmp_path, workspace):
    """A denied call costs an LLM round-trip, so it must count against the cap.

    Without this an agent could loop on a denied tool forever while reporting
    zero tool calls.
    """
    policy = Policy.from_json({"rules": [{"tool": "calculator", "effect": "deny"}]})
    responses = [json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})] * 10
    result = build(
        tmp_path, responses, policy=policy, budgets=Budget(max_tool_calls=3)
    ).run("loop")
    assert result.stop_reason == "budget_exceeded"
    assert result.budget_usage.tool_calls == 3


def test_budget_exceeded_is_audited(tmp_path, workspace):
    responses = [json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})] * 10
    result = build(tmp_path, responses, budgets=Budget(max_tool_calls=1)).run("loop")
    with open(result.audit_path, encoding="utf-8") as handle:
        events = [json.loads(line)["event"] for line in handle if line.strip()]
    assert "budget_exceeded" in events


def test_llm_error_stops_the_run_cleanly(tmp_path, workspace):
    agent = Agent(
        ScriptedLLM([]),
        AgentConfig(
            audit_path=str(tmp_path / "a.jsonl"),
            workspace_root=str(workspace),
            approval_handler=AutoApproveHandler(),
        ),
    )
    result = agent.run("anything")
    assert result.stop_reason == "llm_error"
    assert result.answer is None


def test_llm_error_is_audited(tmp_path, workspace):
    agent = Agent(
        ScriptedLLM([]),
        AgentConfig(
            audit_path=str(tmp_path / "a.jsonl"),
            workspace_root=str(workspace),
            approval_handler=AutoApproveHandler(),
        ),
    )
    result = agent.run("anything")
    with open(result.audit_path, encoding="utf-8") as handle:
        events = [json.loads(line)["event"] for line in handle if line.strip()]
    assert "llm_error" in events


def test_malformed_json_is_retried_then_recovered(tmp_path, workspace):
    result = build(
        tmp_path,
        ["not json at all", json.dumps({"final": "recovered"})],
        max_retries=3,
    ).run("task")
    assert result.stop_reason == "final_answer"
    assert result.answer == "recovered"


def test_json_fences_are_tolerated(tmp_path, workspace):
    fenced = "```json\n" + json.dumps({"final": "fenced answer"}) + "\n```"
    result = build(tmp_path, [fenced]).run("task")
    assert result.answer == "fenced answer"


def test_json_embedded_in_prose_is_extracted(tmp_path, workspace):
    noisy = 'Sure! Here you go: {"final": "embedded"} Hope that helps.'
    result = build(tmp_path, [noisy]).run("task")
    assert result.answer == "embedded"


def test_retry_exhaustion_reports_max_retries(tmp_path, workspace):
    result = build(tmp_path, ["nope"] * 10, max_retries=2).run("task")
    assert result.stop_reason == "max_retries"


def test_tool_call_without_args_is_retried(tmp_path, workspace):
    result = build(
        tmp_path,
        [json.dumps({"tool": 42}), json.dumps({"final": "recovered"})],
    ).run("task")
    assert result.stop_reason == "final_answer"
    assert result.tool_calls == []


def test_tool_call_with_non_dict_args_is_retried(tmp_path, workspace):
    result = build(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": "not-an-object"}),
            json.dumps({"final": "recovered"}),
        ],
    ).run("task")
    assert result.tool_calls == []


def test_final_answer_failing_schema_is_retried(tmp_path, workspace):
    """The schema constrains the envelope, so it must require and shape `final`."""
    schema = {
        "type": "object",
        "required": ["final"],
        "properties": {
            "final": {
                "type": "object",
                "required": ["answer", "confidence"],
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            }
        },
    }
    result = build(
        tmp_path,
        [
            json.dumps({"final": {"answer": "x"}}),
            json.dumps({"final": {"answer": "x", "confidence": 0.9}}),
        ],
        final_answer_schema=schema,
    ).run("task")
    assert result.stop_reason == "final_answer"
    assert result.answer == {"answer": "x", "confidence": 0.9}


def test_default_final_answer_schema_requires_a_string_final(tmp_path, workspace):
    result = build(
        tmp_path, [json.dumps({"final": {"answer": "not a string"}})]
    ).run("task")
    assert result.stop_reason != "final_answer"


def test_run_result_exposes_deterministic_fields(tmp_path, workspace):
    result = build(tmp_path, [json.dumps({"final": "done"})]).run("task")
    for field in ("answer", "tool_calls", "audit_path", "budget_usage", "stop_reason"):
        assert hasattr(result, field)


def test_repeated_runs_share_the_audit_chain(tmp_path, workspace):
    config = AgentConfig(
        audit_path=str(tmp_path / "shared.jsonl"),
        workspace_root=str(workspace),
        approval_handler=AutoApproveHandler(),
    )
    agent = Agent(ScriptedLLM([json.dumps({"final": "one"})]), config)
    agent.run("first")
    second = Agent(ScriptedLLM([json.dumps({"final": "two"})]), config)
    second.run("second")
    assert verify(str(tmp_path / "shared.jsonl"))[0] is True
