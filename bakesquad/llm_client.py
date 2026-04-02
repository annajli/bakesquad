"""
Thin LLM wrapper — the ONLY module that knows about backends.

All LLM calls in the codebase go through `chat()`.  Switching backends
requires *only* changing the MODEL_BACKEND env var.

DEVIATION from CONTEXT.md: CONTEXT.md specifies LangChain as the agent
framework.  We bypass LangChain entirely and call the OpenAI-compatible /
Anthropic APIs directly.  Reason: LangChain prompt-template + chain overhead
added ~1-2 s per call in testing, which blows the 60 s budget on batched steps.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Backend detection (once, at import time)
# ---------------------------------------------------------------------------
BACKEND: str = os.environ.get("MODEL_BACKEND", "ollama").lower()

_MODELS: dict[str, str] = {
    "ollama": os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
    "groq": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
    "claude": "claude-sonnet-4-20250514",
}

_TIME_BUDGETS: dict[str, int] = {
    "ollama": 180,   # 3 minutes — consumer hardware can't match API speed
    "groq": 60,
    "claude": 60,
}

if BACKEND not in _MODELS:
    raise ValueError(
        f"Unknown MODEL_BACKEND={BACKEND!r}. Choose from: ollama, groq, claude"
    )

# ---------------------------------------------------------------------------
# Lazy client singletons
# ---------------------------------------------------------------------------
_client: Any = None


def _init_client() -> Any:
    global _client
    if _client is not None:
        return _client

    if BACKEND in ("ollama", "groq"):
        from openai import OpenAI

        if BACKEND == "ollama":
            _client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY env var is required when MODEL_BACKEND=groq"
                )
            _client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )
    elif BACKEND == "claude":
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY env var is required when MODEL_BACKEND=claude"
            )
        _client = anthropic.Anthropic(api_key=api_key)

    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> str:
    """Single-turn LLM call.  Every call in the pipeline routes through here."""
    client = _init_client()
    model = _MODELS[BACKEND]

    if BACKEND in ("ollama", "groq"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content or ""
    else:  # claude
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.content[0].text

    return _strip_think_tags(raw)


def get_backend() -> str:
    return BACKEND


def get_model() -> str:
    return _MODELS[BACKEND]


def get_time_budget() -> int:
    return _TIME_BUDGETS[BACKEND]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Strip qwen3-style <think>…</think> reasoning blocks."""
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict | list:
    """Robustly pull JSON from an LLM response (handles code fences, prose)."""
    # Try markdown code-fence first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    # Try outermost { … }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try outermost [ … ]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON in LLM response: {text[:300]}")
