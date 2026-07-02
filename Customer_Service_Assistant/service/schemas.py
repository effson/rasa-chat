"""Domain schemas for the customer service layer.

These models represent the internal conversational domain and are
independent of any transport layer (HTTP, WebSocket, etc.).
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ObjectData(BaseModel):
    """A structured object attached to a message (order, product, etc.)."""

    type: str
    id: str
    title: Optional[str] = None
    attributes: dict = Field(default_factory=dict)


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
