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
import time
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

    for attempt in range(3):
        try:
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

        except Exception as e:
            err_str = str(e)
            # Rate limit: parse wait time from Groq/OpenAI error message and retry
            if "rate_limit" in err_str.lower() or "429" in err_str or "rate limit" in err_str.lower():
                wait = _parse_retry_after(err_str)
                if attempt < 2:
                    time.sleep(wait)
                    continue
            raise

    raise RuntimeError("chat(): unreachable")


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


def _parse_retry_after(error_msg: str) -> float:
    """Extract retry-after seconds from a rate limit error message."""
    m = re.search(r"try again in\s+([\d.]+)s", error_msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.5   # small buffer
    return 5.0   # safe default


def _strip_think_tags(text: str) -> str:
    """Strip qwen3-style <think>…</think> reasoning blocks."""
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict | list:
    """Robustly pull JSON from an LLM response (handles code fences, prose, fraction values)."""
    # Try markdown code-fence first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    def _try_parse(s: str) -> dict | list | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # LLMs sometimes emit Python-style fractions like "2.0/3" in JSON values.
        # Replace them with their computed float values before retrying.
        fixed = re.sub(
            r':\s*(\d+\.?\d*)\s*/\s*(\d+\.?\d*)',
            lambda m: f": {round(float(m.group(1)) / float(m.group(2)), 4)}",
            s,
        )
        if fixed != s:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        return None

    # Try outermost { … }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        result = _try_parse(text[start:end])
        if result is not None:
            return result

    # Try outermost [ … ]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        result = _try_parse(text[start:end])
        if result is not None:
            return result

    raise ValueError(f"No valid JSON in LLM response: {text[:300]}")
