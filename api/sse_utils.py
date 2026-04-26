"""SSE formatting helpers."""

import json
from typing import Any


def sse(data: Any) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
