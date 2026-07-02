"""Pydantic schemas for the chat API — request and response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ObjectData(BaseModel):
    """Structured object attached to a message (e.g. an order or product reference)."""

    type: str
    id: str
    title: Optional[str] = None
    attributes: dict = Field(default_factory=dict)


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


class HistoryMessage(BaseModel):
    """A single message in the conversation history."""

    role: str  # "user" | "bot"
    text: Optional[str] = None
    object: Optional[ObjectData] = None


class HistoryResponse(BaseModel):
    """Response returned when querying chat history."""

    sender_id: str
    messages: list[HistoryMessage]
