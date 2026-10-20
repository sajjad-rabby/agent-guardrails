"""Tamper-evident audit log: hash-chained JSONL records."""
from __future__ import annotations

import dataclasses
import datetime as _datetime
import enum
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_GENESIS_HASH = "0" * 64
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def json_default(value: Any) -> Any:
    """Deterministically encode values JSON does not handle natively.

An audit log must never crash the agent it is recording, so dataclasses,
enums, sets, dates and paths are normalised rather than raising. The
encoding is deterministic, which is what keeps the hash chain stable.

Args:
    value: The object to encode.

Returns:
    A JSON-compatible representation.

    An audit log must never crash the agent it is recording, so dataclasses,
    enums, sets, dates and paths are normalised rather than raising. The
    encoding is deterministic, which is what keeps the hash chain stable.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    )


def _record_hash(record: Dict[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "hash"}
    return hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only JSONL log where each record commits to the previous one."""

    def __init__(self, path: str):
        self._path = path

    @property
    def path(self) -> str:
        """Filesystem location of the log file."""
        return self._path

    def _read_tail(self) -> Tuple[int, str]:
        """Return (seq, hash) of the last non-blank record, or (0, genesis)."""
        if not os.path.exists(self._path):
            return 0, _GENESIS_HASH
        with open(self._path, "r", encoding="utf-8") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
        if not lines:
            return 0, _GENESIS_HASH
        record = json.loads(lines[-1])
        return int(record["seq"]), str(record["hash"])

    def append(self, event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write one chained record and return it.

The payload is normalised through the same encoder used for hashing, so the
returned record is byte-for-byte what was persisted. Dataclasses, enums and
sets therefore come back as plain JSON values rather than the
original Python objects.

Args:
    event: Event name.
    payload: Optional event data.

Returns:
    The record that was written.

        The payload is normalised through the same encoder used for hashing, so
        the returned record is byte-for-byte what was persisted. Dataclasses,
        enums and sets therefore come back as plain JSON values rather than the
        original Python objects.
        """
        seq, prev_hash = self._read_tail()
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = {
            "seq": seq + 1,
            "ts": datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT),
            "event": event,
            "payload": json.loads(_canonical(payload or {})),
            "prev_hash": prev_hash,
        }
        record["hash"] = _record_hash(record)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
        return record


def verify(path: str) -> Tuple[bool, List[str]]:
    """Re-derive the hash chain; return (intact, list of error strings)."""
    errors: List[str] = []
    if not os.path.exists(path):
        return False, ["file not found: {}".format(path)]
    expected_seq = 1
    prev_hash = _GENESIS_HASH
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append("line {}: invalid json".format(line_number))
                continue
            if int(record.get("seq", -1)) != expected_seq:
                errors.append(
                    "line {}: seq mismatch, expected {} got {}".format(
                        line_number, expected_seq, record.get("seq")
                    )
                )
            if record.get("prev_hash") != prev_hash:
                errors.append(
                    "line {}: prev_hash mismatch, expected {} got {}".format(
                        line_number, prev_hash, record.get("prev_hash")
                    )
                )
            stored_hash = record.get("hash")
            if stored_hash is None:
                errors.append("line {}: missing hash".format(line_number))
            elif _record_hash(record) != stored_hash:
                errors.append("line {}: hash mismatch (record was modified)".format(line_number))
            if isinstance(stored_hash, str):
                prev_hash = stored_hash
            expected_seq += 1
    return (len(errors) == 0), errors
