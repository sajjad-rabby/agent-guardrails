"""Tests for the built-in toolset, argument validation and path confinement."""
from __future__ import annotations

import json
import os

import pytest

from agent_guardrails.tools import (
    Tool,
    ToolError,
    ToolInputError,
    ToolRegistry,
    default_tools,
    validate_args,
)


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "note.txt").write_text("hello world", encoding="utf-8")
    (root / "sub" / "nested.txt").write_text("nested data", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_registry_register_get_and_list():
    registry = ToolRegistry()
    tool = Tool(
        name="echo",
        description="echo input",
        params_schema={"type": "object"},
        handler=lambda **kw: {"echo": kw.get("text", "")},
    )
    registry.register(tool)
    assert registry.get("echo") is tool
    assert registry.names() == ["echo"]
    assert [t.name for t in registry.list()] == ["echo"]


def test_registry_get_unknown_returns_none():
    assert ToolRegistry().get("nope") is None


def test_default_tools_exposes_expected_names():
    names = default_tools(roots=["."]).names()
    for expected in ("calculator", "read_file", "write_file", "list_dir"):
        assert expected in names


def test_tool_spec_shape():
    registry = default_tools(roots=["."])
    spec = registry.get("calculator").tool_spec()
    assert spec["name"] == "calculator"
    assert spec["description"]
    assert spec["params"]["required"] == ["expression"]


def test_tool_spec_is_json_serializable():
    for tool in default_tools(roots=["."]).list():
        assert json.loads(json.dumps(tool.tool_spec()))["name"] == tool.name


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------
def test_validate_args_accepts_valid_payload():
    tool = default_tools(roots=["."]).get("calculator")
    assert validate_args(tool, {"expression": "1+1"}) == []


def test_validate_args_reports_schema_errors():
    tool = default_tools(roots=["."]).get("calculator")
    errors = validate_args(tool, {})
    assert errors and "expression" in errors[0]


# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------
@pytest.fixture()
def calc():
    return default_tools(roots=["."]).get("calculator")


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2", 4),
        ("10 - 3", 7),
        ("2 * 3 + 1", 7),
        ("(2 + 3) * 4", 20),
        ("7 / 2", 3.5),
        ("-5 + 2", -3),
        ("2 ** 8", 256),
        ("max(1, 9)", 9),
        ("min(4, 2)", 2),
        ("round(3.14159, 2)", 3.14),
        ("abs(-7)", 7),
    ],
)
def test_calculator_arithmetic(calc, expression, expected):
    assert calc.handler(expression=expression)["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd')",
        "().__class__",
        "lambda: 1",
        "[1,2,3]",
        "{'a': 1}",
        "x + 1",
        "1 if True else 2",
    ],
)
def test_calculator_rejects_unsafe_constructs(calc, expression):
    with pytest.raises(ToolInputError):
        calc.handler(expression=expression)


def test_calculator_rejects_string_constants(calc):
    with pytest.raises(ToolInputError):
        calc.handler(expression="'abc' + 1")


def test_calculator_rejects_syntax_errors(calc):
    with pytest.raises(ToolError):
        calc.handler(expression="2 +")


def test_calculator_rejects_keyword_arguments(calc):
    with pytest.raises(ToolInputError):
        calc.handler(expression="round(3.14159, ndigits=2)")


# --------------------------------------------------------------------------
# file tools and root confinement
# --------------------------------------------------------------------------
@pytest.fixture()
def files(workspace):
    return default_tools(roots=[str(workspace)])


def test_read_file_inside_root(files, workspace):
    result = files.get("read_file").handler(path=str(workspace / "note.txt"))
    assert result["content"] == "hello world"


def test_read_file_rejects_path_traversal(files, workspace):
    with pytest.raises(ToolInputError):
        files.get("read_file").handler(path=str(workspace / ".." / "secret.txt"))


def test_read_file_rejects_absolute_path_outside_root(files, workspace, tmp_path):
    with pytest.raises(ToolInputError):
        files.get("read_file").handler(path=str(tmp_path / "secret.txt"))


def test_write_then_read_round_trip(files, workspace):
    files.get("write_file").handler(path=str(workspace / "new.txt"), content="written")
    result = files.get("read_file").handler(path=str(workspace / "new.txt"))
    assert result["content"] == "written"


def test_write_file_rejects_traversal(files, workspace):
    with pytest.raises(ToolInputError):
        files.get("write_file").handler(
            path=str(workspace / ".." / "escape.txt"), content="nope"
        )


def test_list_dir_lists_entries(files, workspace):
    result = files.get("list_dir").handler(path=str(workspace))
    names = result["entries"] if isinstance(result.get("entries"), list) else list(result)
    assert "note.txt" in json.dumps(names)


def test_list_dir_rejects_traversal(files, workspace):
    with pytest.raises(ToolInputError):
        files.get("list_dir").handler(path=str(workspace / ".." / "."))


def test_path_escape_is_blocked(tmp_path):
    """A path pointing outside the root must not grant access."""
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("classified", encoding="utf-8")
    tools = default_tools(roots=[str(root)])
    with pytest.raises(ToolInputError):
        tools.get("read_file").handler(path=str(secret))
