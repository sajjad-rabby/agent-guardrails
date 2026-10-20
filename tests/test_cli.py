"""Tests for the guardrails command-line interface."""
from __future__ import annotations

import json
import os

import pytest

from agent_guardrails import cli


@pytest.fixture()
def policy_file(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "tool": "calculator",
                        "effect": "allow",
                        "reason": "safe arithmetic",
                        "conditions": [
                            {"arg": "expression", "operator": "matches", "value": r"^[0-9+\-*/() ]+$"}
                        ],
                    },
                    {"tool": "calculator", "effect": "deny", "reason": "not allowed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture()
def audit_file(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    from agent_guardrails.audit import AuditLog

    log = AuditLog(path)
    log.append("run_started", {"task": "x"})
    log.append("run_finished", {"stop_reason": "final_answer"})
    return path


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def test_no_command_prints_help(capsys):
    assert cli.main([]) == cli.EXIT_ERROR
    assert "usage" in capsys.readouterr().out.lower()


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0


# --------------------------------------------------------------------------
# policy-check
# --------------------------------------------------------------------------
@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _args(tmp_path, payload):
    path = tmp_path / "args.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_policy_check_allows(policy_file, tmp_path, capsys):
    code = cli.main(
        [
            "policy-check",
            "--policy",
            policy_file,
            "--tool",
            "calculator",
            "--args",
            _args(tmp_path, {"expression": "2+2"}),
        ]
    )
    assert code == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["allowed"] is True


def test_policy_check_denies_unmatched_args_by_default(policy_file, capsys):
    """No args means the condition cannot match, so default-deny applies."""
    code = cli.main(["policy-check", "--policy", policy_file, "--tool", "calculator"])
    assert code == cli.EXIT_DENIED
    assert json.loads(capsys.readouterr().out)["allowed"] is False


def test_policy_check_denies_with_exit_one(policy_file, tmp_path, capsys):
    code = cli.main(
        [
            "policy-check",
            "--policy",
            policy_file,
            "--tool",
            "calculator",
            "--args",
            _args(tmp_path, {"expression": "__import__('os')"}),
        ]
    )
    assert code == cli.EXIT_DENIED
    output = json.loads(capsys.readouterr().out)
    assert output["allowed"] is False
    assert output["reason"] == "not allowed"


def test_policy_check_missing_file_exits(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["policy-check", "--policy", "nope.json", "--tool", "x"])
    assert "file not found" in str(excinfo.value)


def test_policy_check_invalid_json_exits(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["policy-check", "--policy", str(bad), "--tool", "x"])
    assert "not valid JSON" in str(excinfo.value)


def test_policy_check_rejects_non_object_args(tmp_path, policy_file):
    args = tmp_path / "args.json"
    args.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            ["policy-check", "--policy", policy_file, "--tool", "calculator", "--args", str(args)]
        )
    assert "must contain a JSON object" in str(excinfo.value)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def _schema(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps({"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}),
        encoding="utf-8",
    )
    return str(path)


def test_validate_accepts(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert cli.main(["validate", "--schema", _schema(tmp_path), "--instance", str(good)]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_validate_rejects_with_exit_two(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"b": 2}), encoding="utf-8")
    code = cli.main(["validate", "--schema", _schema(tmp_path), "--instance", str(bad)])
    assert code == cli.EXIT_INVALID
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is False
    assert output["errors"]


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def test_verify_intact(audit_file, capsys):
    assert cli.main(["verify", "--path", audit_file]) == cli.EXIT_OK
    output = json.loads(capsys.readouterr().out)
    assert output["intact"] is True
    assert output["record_count"] == 2


def test_verify_detects_tampering(audit_file, tmp_path, capsys):
    lines = open(audit_file, encoding="utf-8").read().splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["task"] = "tampered"
    lines[0] = json.dumps(tampered, separators=(",", ":"))
    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert cli.main(["verify", "--path", str(broken)]) == cli.EXIT_INVALID
    assert json.loads(capsys.readouterr().out)["intact"] is False


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
def test_tools_lists_builtin_tools(capsys):
    assert cli.main(["tools"]) == cli.EXIT_OK
    names = [tool["name"] for tool in json.loads(capsys.readouterr().out)["tools"]]
    assert "calculator" in names


def test_tools_accepts_repeatable_domains(capsys):
    assert cli.main(["tools", "--domain", "example.com", "--domain", "example.org"]) == cli.EXIT_OK
    assert "tools" in json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def _responses(tmp_path, items):
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return str(path)


def test_run_completes(tmp_path, workspace, policy_file, capsys):
    responses = _responses(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "2+2"}}),
            json.dumps({"final": "2 + 2 = 4."}),
        ],
    )
    code = cli.main(
        [
            "run",
            "--task",
            "what is 2+2",
            "--responses",
            responses,
            "--policy",
            policy_file,
            "--audit",
            str(tmp_path / "run.jsonl"),
            "--root",
            str(workspace),
        ]
    )
    assert code == cli.EXIT_OK
    output = json.loads(capsys.readouterr().out)
    assert output["stop_reason"] == "final_answer"
    assert output["answer"] == "2 + 2 = 4."
    assert output["tool_calls"][0]["ok"] is True


def test_run_denied_returns_nonzero(tmp_path, workspace, policy_file, capsys):
    responses = _responses(
        tmp_path,
        [
            json.dumps({"tool": "calculator", "args": {"expression": "__import__('os')"}}),
            json.dumps({"final": "blocked"}),
        ],
    )
    code = cli.main(
        [
            "run",
            "--task",
            "escape",
            "--responses",
            responses,
            "--policy",
            policy_file,
            "--audit",
            str(tmp_path / "run2.jsonl"),
            "--root",
            str(workspace),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["tool_calls"] == []
    assert code == cli.EXIT_OK


def test_run_rejects_bad_responses_file(tmp_path, workspace):
    responses = _responses(tmp_path, {"not": "a list"})
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--task", "x", "--responses", responses, "--root", str(workspace)])
    assert "JSON array of strings" in str(excinfo.value)


def test_run_openai_mode_requires_endpoint(tmp_path, workspace):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--task", "x", "--llm", "openai", "--root", str(workspace)])
    assert "requires --base-url and --model" in str(excinfo.value)


def test_run_writes_auditable_log(tmp_path, workspace):
    responses = _responses(tmp_path, [json.dumps({"final": "done"})])
    audit = str(tmp_path / "cli-audit.jsonl")
    cli.main(
        ["run", "--task", "x", "--responses", responses, "--audit", audit, "--root", str(workspace)]
    )
    from agent_guardrails.audit import verify

    assert os.path.exists(audit)
    assert verify(audit)[0] is True
