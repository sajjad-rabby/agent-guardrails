"""Tool definitions, argument validation and the built-in toolset.

The calculator evaluates expressions by walking the AST against an allowlist of
nodes, operators and functions rather than using ``eval``. File tools
are constrained to allowlisted roots and domains.

- a name and human-readable description,
- a JSON-schema (subset, see :mod:`agent_guardrails.schema`) describing its
  arguments,
- a callable ``handler(**args) -> dict`` that performs the work.

The registry plus the schema validator give the agent layer a single choke
point where tool arguments are checked before any code runs. The built-in
toolset includes a strictly-whitelisted calculator and file/HTTP tools that
are constrained to allowlisted roots and domains.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .schema import validate

__all__ = [
    "Tool",
    "ToolError",
    "ToolInputError",
    "ToolRegistry",
    "default_tools",
]


class ToolError(Exception):
    """Base class for tool execution failures."""


class ToolInputError(ToolError):
    """Raised when tool arguments do not satisfy the tool's schema."""


@dataclass
class Tool:
    """A single tool available to the agent."""

    name: str
    description: str
    params_schema: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    require_approval: bool = False
    cost_per_call: float = 0.0
    allowed_roots: Optional[List[str]] = None
    allowed_domains: Optional[List[str]] = None

    def tool_spec(self) -> Dict[str, Any]:
        """Return a serializable spec describing this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params_schema,
        }


def validate_args(tool: Tool, args: Dict[str, Any]) -> List[str]:
    """Validate arguments against a tool's parameter schema.

Args:
    tool: The tool whose schema applies.
    args: Candidate arguments.

Returns:
    A list of human-readable errors; an empty list means the
arguments are acceptable.

    Returns a list of human-readable error strings; an empty list means the
    arguments are acceptable.
    """
    return validate(args, tool.params_schema)


class ToolRegistry:
    """A name-keyed collection of tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add ``tool`` to the registry, replacing any tool of the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool registered under ``name``, or ``None``."""
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        """Return all registered tools sorted by name."""
        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> List[str]:
        """Return the names of all registered tools sorted alphabetically."""
        return sorted(self._tools)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _calc_node(node: ast.AST) -> Any:
    """Evaluate a whitelisted AST node. Everything else is rejected."""
    if isinstance(node, ast.Expression):
        return _calc_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolInputError("calculator only accepts numeric constants")
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolInputError("unknown name: %s" % node.id)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolInputError("unsupported operator")
        left = _calc_node(node.left)
        right = _calc_node(node.right)
        result = op(left, right)
        if isinstance(result, complex):
            raise ToolInputError("operation produced a complex number")
        return result
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolInputError("unsupported unary operator")
        return op(_calc_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolInputError("only whitelisted function calls are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ToolInputError("unknown function: %s" % node.func.id)
        if node.keywords:
            raise ToolInputError("keyword arguments are not supported")
        args = [_calc_node(arg) for arg in node.args]
        return func(*args)
    raise ToolInputError("expression contains a disallowed construct")


def _calculator_handler(expression: str) -> Dict[str, Any]:
    """Safely evaluate an arithmetic expression (no eval/exec)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError("invalid expression: %s" % exc.msg) from exc
    try:
        result = _calc_node(tree)
    except ZeroDivisionError as exc:
        raise ToolInputError("division by zero") from exc
    except OverflowError as exc:
        raise ToolInputError("operation overflowed") from exc
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc
    if isinstance(result, float) and abs(result) < 1e15 and result == int(result):
        result = int(result)
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


def _path_within_roots(path: str, roots: Sequence[str]) -> str:
    """Resolve ``path`` and ensure it lies inside one of ``roots``."""
    resolved = os.path.realpath(path)
    for root in roots:
        root_real = os.path.realpath(root)
        if resolved == root_real or resolved.startswith(root_real + os.sep):
            return resolved
    raise ToolInputError(
        "path is outside the allowed roots: %s" % path
    )


def _make_read_file(roots: Sequence[str], max_bytes: int) -> Callable[..., Dict[str, Any]]:
    def handler(path: str, max_bytes_override: Optional[int] = None) -> Dict[str, Any]:
        """Read a file within the allowed roots.

        Args:
            path: File to read.
            max_bytes_override: Optional size cap replacing the tool default.
        """
        limit = max_bytes_override if max_bytes_override is not None else max_bytes
        resolved = _path_within_roots(path, roots)
        size = os.path.getsize(resolved)
        if size > limit:
            raise ToolInputError(
                "file is larger than the %d byte limit" % limit
            )
        with open(resolved, "r", encoding="utf-8") as handle:
            content = handle.read(limit)
        return {"path": resolved, "size": size, "content": content}

    return handler


def _make_write_file(roots: Sequence[str]) -> Callable[..., Dict[str, Any]]:
    def handler(path: str, content: str) -> Dict[str, Any]:
        """Write text to a file within the allowed roots.

        Args:
            path: Destination file.
            content: Text to write.
        """
        resolved = _path_within_roots(path, roots)
        parent = os.path.dirname(resolved)
        if parent and not os.path.isdir(parent):
            raise ToolInputError("parent directory does not exist: %s" % parent)
        with open(resolved, "w", encoding="utf-8") as handle:
            handle.write(content)
        return {"path": resolved, "bytes_written": len(content.encode("utf-8"))}

    return handler


def _make_list_dir(roots: Sequence[str]) -> Callable[..., Dict[str, Any]]:
    def handler(path: str) -> Dict[str, Any]:
        """List a directory within the allowed roots.

        Args:
            path: Directory to list.
        """
        resolved = _path_within_roots(path, roots)
        if not os.path.isdir(resolved):
            raise ToolInputError("not a directory: %s" % resolved)
        entries: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(resolved)):
            full = os.path.join(resolved, name)
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir", "size": 0})
            else:
                entries.append(
                    {"name": name, "type": "file", "size": os.path.getsize(full)}
                )
        return {"path": resolved, "entries": entries}

    return handler


# ---------------------------------------------------------------------------
# HTTP tool
# ---------------------------------------------------------------------------

_USER_AGENT = "agent-guardrails/0.1"
_HTTP_TIMEOUT_SECONDS = 30.0


def _hostname_allowed(hostname: str, allowed_domains: Sequence[str]) -> bool:
    """True when ``hostname`` equals an allowed domain or is its subdomain."""
    hostname = hostname.lower().rstrip(".")
    for domain in allowed_domains:
        domain = domain.lower().rstrip(".")
        if hostname == domain or hostname.endswith("." + domain):
            return True
    return False


def _make_http_get(allowed_domains: Sequence[str]) -> Callable[..., Dict[str, Any]]:
    def handler(url: str, max_bytes: int = 65536) -> Dict[str, Any]:
        """Fetch a URL within the allowed domains.

        Args:
            url: Absolute http or https URL.
            max_bytes: Response size cap.

        Returns:
            Mapping with the status code, headers and decoded body.
        """
        if not url.startswith("http://") and not url.startswith("https://"):
            raise ToolInputError("only http and https URLs are allowed")
        hostname = urllib.request.urlsplit(url).hostname or ""
        if not hostname:
            raise ToolInputError("URL has no hostname")
        if not _hostname_allowed(hostname, allowed_domains):
            raise ToolInputError(
                "hostname is not in the allowlist: %s" % hostname
            )
        request = urllib.request.Request(
            url, headers={"User-Agent": _USER_AGENT}, method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
                status = resp.status
                data = resp.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise ToolError("HTTP error %d for %s" % (exc.code, url)) from exc
        except urllib.error.URLError as exc:
            raise ToolError("connection failed for %s: %s" % (url, exc.reason)) from exc
        if len(data) > max_bytes:
            raise ToolInputError(
                "response exceeds the %d byte limit" % max_bytes
            )
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolInputError("response is not valid UTF-8") from exc
        return {"status": status, "url": url, "content": content}

    return handler


# ---------------------------------------------------------------------------
# Default toolset
# ---------------------------------------------------------------------------

_READ_FILE_MAX_BYTES = 65536


def default_tools(
    roots: Optional[Sequence[str]] = None,
    allowed_domains: Optional[Sequence[str]] = None,
) -> ToolRegistry:
    """Build the default tool registry.

Args:
    roots: Directories the file tools may touch. Defaults to the current directory.
    allowed_domains: Hosts the HTTP tool may reach. Defaults to none, which means the
        HTTP tool refuses every request until a domain allowlist is configured.

Returns:
    A populated :class:`ToolRegistry`.

    ``roots`` scopes the file tools (default: the current working directory).
    ``allowed_domains`` scopes ``http_get``; when omitted the HTTP tool is
    still registered but every request will fail the allowlist check until a
    domain allowlist is configured.
    """
    if roots is None:
        roots = [os.getcwd()]
    roots = [os.path.abspath(r) for r in roots]
    domains = list(allowed_domains or [])

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description=(
                "Evaluate an arithmetic expression using + - * / // % **, "
                "math functions (abs, round, min, max, sqrt, pow, floor, "
                "ceil, log, exp, sin, cos, tan) and constants (pi, e, tau)."
            ),
            params_schema={
                "type": "object",
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
            },
            handler=_calculator_handler,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file located inside an allowed root.",
            params_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
            handler=_make_read_file(roots, _READ_FILE_MAX_BYTES),
            allowed_roots=list(roots),
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description="Write a UTF-8 text file inside an allowed root.",
            params_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
            handler=_make_write_file(roots),
            allowed_roots=list(roots),
        )
    )
    registry.register(
        Tool(
            name="list_dir",
            description="List entries of a directory inside an allowed root.",
            params_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
            handler=_make_list_dir(roots),
            allowed_roots=list(roots),
        )
    )
    registry.register(
        Tool(
            name="http_get",
            description=(
                "Perform an HTTP GET against a hostname in the allowlist and "
                "return the response body as UTF-8 text (size limited)."
            ),
            params_schema={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
            handler=_make_http_get(domains),
            allowed_domains=domains,
        )
    )
    return registry
