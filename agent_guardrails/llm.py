"""LLM clients: any OpenAI-compatible endpoint plus a scripted offline stub."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an unusable response."""


class LLMClient(Protocol):
    """Minimal chat-completion interface every agent client must satisfy."""

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        """Return the assistant text for the given chat messages."""
        ...


class OpenAICompatLLM:
    """Chat-completions client for any OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        if not base_url:
            raise LLMError("base_url must be a non-empty string")
        if not model:
            raise LLMError("model must be a non-empty string")
        self.base_url = base_url.rstrip("/")
        self.model = model
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        """Send a chat-completions request and return the assistant text.

        Args:
            messages: Chat messages in OpenAI format.

        Returns:
            The assistant message content.

        Raises:
            LLMError: on transport failure, an HTTP error, or a malformed reply.
        """
        url = self.base_url + "/v1/chat/completions"
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0.0}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        attempts = max(1, int(self.max_retries))
        last_status = "unknown error"
        for attempt in range(attempts):
            request = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 429 or 500 <= exc.code < 600:
                    last_status = "HTTP {0}".format(exc.code)
                    if attempt < attempts - 1:
                        time.sleep(self.retry_backoff_seconds)
                        continue
                raise LLMError(
                    "LLM request rejected with HTTP {0}".format(exc.code)
                )
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                raise LLMError("LLM transport error: {0}".format(exc))
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise LLMError("LLM returned invalid JSON: {0}".format(exc))
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if not choices:
                raise LLMError("LLM response contained no choices")
            first = choices[0] if isinstance(choices[0], dict) else None
            message = first.get("message") if first is not None else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise LLMError("LLM response content was not a string")
            return content
        raise LLMError(
            "LLM request failed after {0} attempts: {1}".format(attempts, last_status)
        )


class ScriptedLLM:
    """Deterministic offline client that replays pre-recorded responses."""

    def __init__(self, responses: List[str]) -> None:
        self._responses: List[str] = list(responses)
        self.prompts: List[List[Dict[str, Any]]] = []

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        """Return the next scripted response, recording the prompt.

        Raises:
            LLMError: when no scripted responses remain.
        """
        self.prompts.append([dict(message) for message in messages])
        if not self._responses:
            raise LLMError("no scripted responses remaining")
        return self._responses.pop(0)
