"""
llm.py — minimal LLM interface.

Single public function:

    complete(messages, *, system=None, max_tokens=None) -> str

Everything else (model, base_url, auth) comes from config.
No streaming, no tool-use, no retries — those belong in callers.

The function raises LLMError (a plain RuntimeError subclass) on any
non-2xx response so callers can catch one exception type instead of
whatever httpx or json raises.
"""

from __future__ import annotations

import os
import json
import httpx

from research.config import config

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when the OpenRouter API returns an error or unexpected shape."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file and re-run."
        )
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        # OpenRouter strongly recommends these for routing/analytics
        "HTTP-Referer": "https://github.com/krishjoshi/research",
        "X-Title": "research",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def complete(
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request to OpenRouter.

    Parameters
    ----------
    messages:
        List of {"role": ..., "content": ...} dicts.
        Roles should be "user" or "assistant".
        Do NOT include a system message here — use the `system` kwarg instead.
    system:
        Optional system prompt. Prepended as {"role": "system", ...} if given.
    max_tokens:
        Override config.llm.max_tokens for this call.

    Returns
    -------
    str
        The assistant's reply text, stripped of leading/trailing whitespace.

    Raises
    ------
    LLMError
        On non-2xx HTTP response or unexpected response shape.
    """
    all_messages: list[dict] = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    payload = {
        "model": config.llm.model,
        "messages": all_messages,
        "max_tokens": max_tokens or config.llm.max_tokens,
    }

    try:
        response = httpx.post(
            f"{config.llm.base_url.rstrip('/')}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=60.0,
        )
    except httpx.RequestError as exc:
        raise LLMError(f"Network error calling OpenRouter: {exc}") from exc

    if response.status_code != 200:
        # Surface the error body if OpenRouter sent one
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise LLMError(
            f"OpenRouter returned {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"Unexpected response shape from OpenRouter: {exc}\n"
            f"Full response: {json.dumps(data, indent=2)}"
        ) from exc
