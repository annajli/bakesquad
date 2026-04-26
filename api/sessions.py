"""In-memory session store for chat follow-up turns.

Single-user app: a plain dict is sufficient. Sessions are keyed by a UUID
returned to the frontend on first search and included in subsequent chat requests.
"""

from __future__ import annotations

import uuid
from typing import Optional

from bakesquad.session import ConversationSession

_sessions: dict[str, ConversationSession] = {}


def new_session_id() -> str:
    return str(uuid.uuid4())


def get_session(session_id: str) -> Optional[ConversationSession]:
    return _sessions.get(session_id)


def put_session(session_id: str, session: ConversationSession) -> None:
    _sessions[session_id] = session


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
