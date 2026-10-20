"""Tests for the minimal JSON-schema validator."""
from __future__ import annotations

import pytest

from agent_guardrails.schema import validate


def test_empty_schema_accepts_anything():
    assert validate({"a": 1}, {}) == []
    assert validate("text", {}) == []
    assert validate(None, {}) == []


def test_type_matching():
    schema = {"type": "string"}
    assert validate("hello", schema) == []
    assert validate(5, schema) != []


def test_number_accepts_integer_but_not_bool():
    schema = {"type": "number"}
    assert validate(3, schema) == []
    assert validate(3.5, schema) == []
    assert validate(True, schema) != [], "booleans must not satisfy number"


def test_integer_rejects_float():
    assert validate(3, {"type": "integer"}) == []
    assert validate(3.5, {"type": "integer"}) != []


def test_boolean_is_not_integer():
    assert validate(True, {"type": "boolean"}) == []
    assert validate(True, {"type": "integer"}) != []


def test_required_property_missing():
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    errors = validate({"other": 1}, schema)
    assert len(errors) == 1
    assert "missing required property" in errors[0]


def test_nested_object_validation():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer", "minimum": 1}},
            }
        },
    }
    assert validate({"user": {"id": 3}}, schema) == []
    errors = validate({"user": {"id": 0}}, schema)
    assert any("less than minimum" in e for e in errors)
    errors = validate({"user": {}}, schema)
    assert any("missing required property" in e for e in errors)


def test_string_length_bounds():
    schema = {"type": "string", "minLength": 2, "maxLength": 4}
    assert validate("ab", schema) == []
    assert validate("abcd", schema) == []
    assert validate("a", schema) != []
    assert validate("abcde", schema) != []


def test_pattern_constraint():
    schema = {"type": "string", "pattern": r"^[a-z]+$"}
    assert validate("abc", schema) == []
    assert validate("ABC", schema) != []


def test_invalid_pattern_is_reported_not_raised():
    schema = {"type": "string", "pattern": "([unclosed"}
    errors = validate("abc", schema)
    assert any("invalid pattern" in e for e in errors)


def test_number_bounds():
    schema = {"type": "number", "minimum": 0, "maximum": 10}
    assert validate(5, schema) == []
    assert validate(-1, schema) != []
    assert validate(11, schema) != []


def test_enum_constraint():
    schema = {"enum": ["allow", "deny"]}
    assert validate("allow", schema) == []
    assert validate("maybe", schema) != []


def test_array_items_validated():
    schema = {"type": "array", "items": {"type": "integer"}}
    assert validate([1, 2, 3], schema) == []
    errors = validate([1, "two", 3], schema)
    assert any("[1]" in e for e in errors)


def test_error_paths_use_dot_and_index_notation():
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string", "minLength": 2}},
        },
    }
    errors = validate({"tags": ["ok", "x"]}, schema)
    assert len(errors) == 1
    assert errors[0].startswith("$.tags[1]") or "tags[1]" in errors[0]


def test_unknown_properties_are_not_errors():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert validate({"a": "x", "b": 99}, schema) == []


def test_type_mismatch_short_circuits_nested_checks():
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "integer", "minimum": 5}},
    }
    errors = validate("not an object", schema)
    assert len(errors) == 1
    assert "expected type" in errors[0]


def test_null_type():
    assert validate(None, {"type": "null"}) == []
    assert validate(0, {"type": "null"}) != []


def test_validation_collects_all_errors():
    schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    }
    errors = validate({}, schema)
    assert len(errors) == 2
