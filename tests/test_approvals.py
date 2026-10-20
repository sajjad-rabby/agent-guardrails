"""Tests for approval handlers."""
from __future__ import annotations

import pytest

from agent_guardrails.approvals import (
    ApprovalRequest,
    ApprovalResult,
    AutoApproveHandler,
    CliApprovalHandler,
)


def test_auto_approve_approves_everything():
    handler = AutoApproveHandler()
    result = handler.handle(ApprovalRequest("calculator", {"expression": "1+1"}, "safe"))
    assert result.approved is True


def test_auto_approve_approves_dangerous_looking_requests_too():
    """AutoApprove is a test/demo convenience; policy is the real gate."""
    handler = AutoApproveHandler()
    assert handler.handle(ApprovalRequest("write_file", {"path": "/etc/x"}, "")).approved


def _request() -> ApprovalRequest:
    return ApprovalRequest("calculator", {"expression": "1+1"}, "safe arithmetic")


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", " yes "])
def test_cli_handler_accepts_affirmative_answers(monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    assert CliApprovalHandler().handle(_request()).approved is True


@pytest.mark.parametrize("answer", ["n", "N", "no", "", "maybe"])
def test_cli_handler_treats_anything_else_as_decline(monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda *_: answer)
    result = CliApprovalHandler().handle(_request())
    assert result.approved is False
    assert result.note == "user declined"


def test_cli_handler_denies_on_eof(monkeypatch):
    def raise_eof(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    result = CliApprovalHandler().handle(_request())
    assert result.approved is False
    assert result.note == "interrupted"


def test_cli_handler_denies_on_keyboard_interrupt(monkeypatch):
    def raise_interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)
    result = CliApprovalHandler().handle(_request())
    assert result.approved is False
    assert result.note == "interrupted"


def test_cli_handler_survives_non_serialisable_args(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    class Weird:
        pass

    request = ApprovalRequest("x", {"obj": Weird()}, "reason")
    assert CliApprovalHandler().handle(request).approved is True


def test_custom_handler_can_veto():
    class DenyAll:
        def handle(self, request):
            return ApprovalResult(approved=False, note="operator said no")

    assert DenyAll().handle(_request()).approved is False


def test_custom_handler_sees_tool_args_and_reason():
    seen = {}

    class Recorder:
        def handle(self, request):
            seen["tool"] = request.tool
            seen["args"] = dict(request.args)
            seen["reason"] = request.reason
            return ApprovalResult(approved=True, note="ok")

    Recorder().handle(_request())
    assert seen["tool"] == "calculator"
    assert seen["args"] == {"expression": "1+1"}
    assert seen["reason"] == "safe arithmetic"
