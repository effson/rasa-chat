"""Pydantic schemas for the chat API — HTTP request/response models.

These are the API contract. Internal domain logic should use
``Customer_Service_Assistant.service.schemas`` instead.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from Customer_Service_Assistant.service.schemas import (
    ChatMessage as ServiceChatMessage,
    ChatRequest as ServiceChatRequest,
    ChatResponse as ServiceChatResponse,
    DialogueState as ServiceDialogueState,
    HistoryMessage as ServiceHistoryMessage,
    HistoryResponse as ServiceHistoryResponse,
    Message as ServiceMessage,
    ObjectData as ServiceObjectData,
)


# ---------------------------------------------------------------------------
# Shared primitives
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
# Domain message & state
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation (API representation)."""

    role: Literal["user", "bot"]
    text: Optional[str] = None
    object: Optional[ObjectData] = None

    def to_service(self) -> ServiceMessage:
        """Convert to the service-layer domain model."""
        return ServiceMessage(
            role=self.role,
            text=self.text,
            object=self.object.to_service() if self.object else None,
        )

    @classmethod
    def from_service(cls, msg: ServiceMessage) -> "Message":
        """Build from a service-layer domain model."""
        return cls(
            role=msg.role,
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


class DialogueState(BaseModel):
    """The full state of a conversation (API representation)."""

    messages: list[Message] = Field(default_factory=list)

    @classmethod
    def from_json(cls, json_str: str) -> "DialogueState":
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json()

    def to_service(self) -> ServiceDialogueState:
        """Convert to the service-layer domain model."""
        return ServiceDialogueState(
            messages=[m.to_service() for m in self.messages],
        )

    @classmethod
    def from_service(cls, state: ServiceDialogueState) -> "DialogueState":
        """Build from a service-layer domain model."""
        return cls(
            messages=[Message.from_service(m) for m in state.messages],
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

    def to_service(self) -> ServiceChatRequest:
        """Convert to the service-layer domain model."""
        return ServiceChatRequest(
            sender_id=self.sender_id,
            text=self.text,
            object=self.object.to_service() if self.object else None,
            message_id=self.message_id,
        )

    @classmethod
    def from_service(cls, req: ServiceChatRequest) -> "ChatRequest":
        """Build from a service-layer domain model."""
        return cls(
            sender_id=req.sender_id,
            text=req.text,
            object=ObjectData.from_service(req.object) if req.object else None,
            message_id=req.message_id,
        )

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

    def to_service(self) -> ServiceChatMessage:
        """Convert to the service-layer domain model."""
        return ServiceChatMessage(
            text=self.text,
            object=self.object.to_service() if self.object else None,
        )

    @classmethod
    def from_service(cls, msg: ServiceChatMessage) -> "ChatMessage":
        """Build from a service-layer domain model."""
        return cls(
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )

    @classmethod
    def from_service_message(cls, msg: ServiceMessage) -> "ChatMessage":
        """Build from a service-layer Message (convenience shortcut)."""
        return cls(
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


class ChatResponse(BaseModel):
    """Response returned after processing a user message."""

    sender_id: str
    message_id: str
    messages: list[ChatMessage]

    def to_service(self) -> ServiceChatResponse:
        """Convert to the service-layer domain model."""
        return ServiceChatResponse(
            sender_id=self.sender_id,
            message_id=self.message_id,
            messages=[m.to_service() for m in self.messages],
        )

    @classmethod
    def from_service(cls, resp: ServiceChatResponse) -> "ChatResponse":
        """Build from a service-layer domain model."""
        return cls(
            sender_id=resp.sender_id,
            message_id=resp.message_id,
            messages=[ChatMessage.from_service(m) for m in resp.messages],
        )


# ---------------------------------------------------------------------------
# GET /api/chat/history
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """A single message in the conversation history (API representation)."""

    role: str  # "user" | "bot"
    text: Optional[str] = None
    object: Optional[ObjectData] = None

    def to_service(self) -> ServiceHistoryMessage:
        """Convert to the service-layer domain model."""
        return ServiceHistoryMessage(
            role=self.role,
            text=self.text,
            object=self.object.to_service() if self.object else None,
        )

    @classmethod
    def from_service(cls, msg: ServiceHistoryMessage) -> "HistoryMessage":
        """Build from a service-layer domain model."""
        return cls(
            role=msg.role,
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )

    @classmethod
    def from_service_message(cls, msg: ServiceMessage) -> "HistoryMessage":
        """Build from a service-layer Message (convenience shortcut)."""
        return cls(
            role=msg.role,
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


class HistoryResponse(BaseModel):
    """Response returned when querying chat history."""

    sender_id: str
    messages: list[HistoryMessage]

    def to_service(self) -> ServiceHistoryResponse:
        """Convert to the service-layer domain model."""
        return ServiceHistoryResponse(
            sender_id=self.sender_id,
            messages=[m.to_service() for m in self.messages],
        )

    @classmethod
    def from_service(cls, resp: ServiceHistoryResponse) -> "HistoryResponse":
        """Build from a service-layer domain model."""
        return cls(
            sender_id=resp.sender_id,
            messages=[HistoryMessage.from_service(m) for m in resp.messages],
        )
