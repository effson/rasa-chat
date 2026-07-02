"""Domain schemas for the customer service layer.

These models represent the internal conversational domain and are
independent of any transport layer (HTTP, WebSocket, etc.).
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class ObjectData(BaseModel):
    """A structured object attached to a message (order, product, etc.)."""

    type: str
    id: str
    title: Optional[str] = None
    attributes: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Domain message & state
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "bot"]
    text: Optional[str] = None
    object: Optional[ObjectData] = None


class DialogueState(BaseModel):
    """The full state of a conversation, persisted across turns."""

    messages: list[Message] = Field(default_factory=list)

    @classmethod
    def from_json(cls, json_str: str) -> "DialogueState":
        """Deserialize from a JSON string (as stored in the database)."""
        data = json.loads(json_str)
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to a JSON string for database persistence."""
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Inbound message from the user."""

    sender_id: str
    text: Optional[str] = None
    object: Optional[ObjectData] = None
    message_id: Optional[str] = None

    @model_validator(mode="after")
    def _require_text_or_object(self) -> "ChatRequest":
        if self.text is None and self.object is None:
            raise ValueError("At least one of `text` or `object` is required")
        return self


class ChatMessage(BaseModel):
    """A single reply item from the bot."""

    text: Optional[str] = None
    object: Optional[ObjectData] = None


class ChatResponse(BaseModel):
    """Response returned after processing a user message."""

    sender_id: str
    message_id: str
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# GET /api/chat/history
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """A single message in the conversation history."""

    role: str  # "user" | "bot"
    text: Optional[str] = None
    object: Optional[ObjectData] = None


class HistoryResponse(BaseModel):
    """Response returned when querying chat history."""

    sender_id: str
    messages: list[HistoryMessage]
