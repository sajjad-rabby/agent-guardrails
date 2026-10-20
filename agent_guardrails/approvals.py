"""Human-in-the-loop approval gate for sensitive tool calls."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class ApprovalRequest:
    """A pending tool call awaiting human approval."""

    tool: str
    args: Dict[str, Any]
    reason: str


@dataclass
class ApprovalResult:
    """The human decision on an approval request."""

    approved: bool
    note: str = ""


class ApprovalHandler(Protocol):
    """Anything that can adjudicate an approval request."""

    def handle(self, request: ApprovalRequest) -> ApprovalResult:
        """Return the approval decision for the request."""
        ...


class AutoApproveHandler:
    """Approves every request automatically (for tests and demos)."""

    def handle(self, request: ApprovalRequest) -> ApprovalResult:
        """Approve the request unconditionally."""
        return ApprovalResult(approved=True, note="auto-approved")


class CliApprovalHandler:
    """Asks the operator at the terminal to approve or deny each request."""

    def handle(self, request: ApprovalRequest) -> ApprovalResult:
        """Print the request summary and read a yes/no answer from stdin."""
        print("Approval required")
        print("  tool:   {}".format(request.tool))
        print("  args:   {}".format(json.dumps(request.args, sort_keys=True, default=str)))
        print("  reason: {}".format(request.reason))
        try:
            answer = input("Approve {}? [y/N] ".format(request.tool))
        except (EOFError, KeyboardInterrupt):
            return ApprovalResult(approved=False, note="interrupted")
        normalized = answer.strip().lower()
        if normalized in ("y", "yes"):
            return ApprovalResult(approved=True, note="user response")
        return ApprovalResult(approved=False, note="user declined")
