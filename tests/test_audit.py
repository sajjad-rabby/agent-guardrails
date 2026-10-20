"""Tests for the hash-chained audit log."""
from __future__ import annotations

import dataclasses
import json
import os

import pytest

from agent_guardrails.audit import AuditLog, verify


def test_append_creates_file_with_first_record(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    record = AuditLog(path).append("run_started", {"task": "demo"})
    assert os.path.exists(path)
    assert record["seq"] == 1
    assert record["prev_hash"] == "0" * 64
    assert len(record["hash"]) == 64


def test_records_chain_together(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    first = log.append("a", {"i": 1})
    second = log.append("b", {"i": 2})
    assert second["seq"] == 2
    assert second["prev_hash"] == first["hash"]


def test_appends_across_instances(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    AuditLog(path).append("a")
    record = AuditLog(path).append("b")
    assert record["seq"] == 2, "a new instance must resume the existing chain"


def test_empty_log_verifies_as_intact(tmp_path):
    path = str(tmp_path / "empty.jsonl")
    open(path, "w", encoding="utf-8").close()
    intact, errors = verify(path)
    assert intact is True
    assert errors == []


def test_missing_file_is_reported(tmp_path):
    intact, errors = verify(str(tmp_path / "nope.jsonl"))
    assert intact is False
    assert "not found" in errors[0]


def test_payload_change_is_detected(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    log.append("tool_call", {"tool": "calculator"})
    log.append("final_answer", {"answer": "4"})

    lines = open(path, encoding="utf-8").read().splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["tool"] = "write_file"
    lines[0] = json.dumps(tampered, separators=(",", ":"))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    intact, errors = verify(path)
    assert intact is False
    assert any("hash mismatch" in e for e in errors)


def test_deleted_record_is_detected(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    for i in range(4):
        log.append("event", {"i": i})

    lines = open(path, encoding="utf-8").read().splitlines()
    del lines[1]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    intact, errors = verify(path)
    assert intact is False


def test_reordering_records_is_detected(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    for i in range(3):
        log.append("event", {"i": i})

    lines = open(path, encoding="utf-8").read().splitlines()
    lines[0], lines[2] = lines[2], lines[0]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    intact, _ = verify(path)
    assert intact is False


def test_garbage_line_is_reported_not_raised(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    AuditLog(path).append("ok")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    intact, errors = verify(path)
    assert intact is False
    assert any("invalid json" in e for e in errors)


def test_unicode_survives_round_trip(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    AuditLog(path).append("note", {"text": "héllo → 世界"})
    intact, _ = verify(path)
    assert intact is True
    record = json.loads(open(path, encoding="utf-8").read().splitlines()[0])
    assert record["payload"]["text"] == "héllo → 世界"


def test_non_serializable_payload_is_encoded_not_raised(tmp_path):
    """An audit log must never crash the agent it is recording."""

    @dataclasses.dataclass
    class Usage:
        tool_calls: int = 3
        cost_usd: float = 0.25

    path = str(tmp_path / "audit.jsonl")
    record = AuditLog(path).append("run_finished", {"usage": Usage()})
    assert record["payload"]["usage"] == {"tool_calls": 3, "cost_usd": 0.25}
    assert verify(path)[0] is True


def test_non_serializable_payload_hashes_consistently(tmp_path):
    @dataclasses.dataclass
    class Usage:
        tool_calls: int = 3

    first = AuditLog(str(tmp_path / "a.jsonl")).append("e", {"u": Usage()})
    second = AuditLog(str(tmp_path / "b.jsonl")).append("e", {"u": Usage()})
    assert first["hash"] == second["hash"]


def test_enum_and_set_payloads(tmp_path):
    import enum

    class Colour(enum.Enum):
        RED = "red"

    path = str(tmp_path / "audit.jsonl")
    record = AuditLog(path).append("e", {"c": Colour.RED, "s": {"b", "a"}})
    assert record["payload"]["c"] == "red"
    assert record["payload"]["s"] == ["a", "b"]
    assert verify(path)[0] is True


def test_path_property_is_exposed(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    assert log.path == path
    assert log.append("e")["event"] == "e"


def test_verify_preserves_sequence_numbers(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path)
    for i in range(5):
        log.append("e", {"i": i})
    lines = open(path, encoding="utf-8").read().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2, 3, 4, 5]
    assert verify(path)[0] is True
