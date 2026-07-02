"""Pydantic schemas for the chat API — HTTP request/response models.

These are the API contract. Internal domain logic should use
``Customer_Service_Assistant.service.schemas`` instead.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from Customer_Service_Assistant.service.schemas import (
    Message as ServiceMessage,
    ObjectData as ServiceObjectData,
)


# ---------------------------------------------------------------------------
# Shared API primitives
# ---------------------------------------------------------------------------


class ObjectData(BaseModel):
    """Structured object attached to a message (API representation)."""

    type: str
    id: str
    title: Optional[str] = None
    attributes: dict = Field(default_factory=dict)

    def to_service(self) -> ServiceObjectData:
        """Convert to the service-layer domain model."""
        return ServiceObjectData(
            type=self.type,
            id=self.id,
            title=self.title,
            attributes=self.attributes,
        )

    @classmethod
    def from_service(cls, obj: ServiceObjectData) -> "ObjectData":
        """Build from a service-layer domain model."""
        return cls(
            type=obj.type,
            id=obj.id,
            title=obj.title,
            attributes=obj.attributes,
        )


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

    def to_service_message(self) -> ServiceMessage:
        """Convert the request payload to a service-layer user Message."""
        return ServiceMessage(
            role="user",
            text=self.text,
            object=self.object.to_service() if self.object else None,
        )


class ChatMessage(BaseModel):
    """A single reply item from the bot (API representation)."""

    text: Optional[str] = None
    object: Optional[ObjectData] = None

    @classmethod
    def from_service(cls, msg: ServiceMessage) -> "ChatMessage":
        """Build from a service-layer Message."""
        return cls(
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


class ChatResponse(BaseModel):
    """Response returned after processing a user message."""

    sender_id: str
    message_id: str
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# GET /api/chat/history
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """A single message in the conversation history (API representation)."""

    role: str  # "user" | "bot"
    text: Optional[str] = None
    object: Optional[ObjectData] = None

    @classmethod
    def from_service(cls, msg: ServiceMessage) -> "HistoryMessage":
        """Build from a service-layer Message."""
        return cls(
            role=msg.role,
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


class HistoryResponse(BaseModel):
    """Response returned when querying chat history."""

    sender_id: str
    messages: list[HistoryMessage]
