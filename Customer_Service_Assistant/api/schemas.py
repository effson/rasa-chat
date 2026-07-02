"""Pydantic schemas for the chat API — HTTP request/response models.

These are the API contract. Internal domain logic should use
``Customer_Service_Assistant.service.schemas`` instead.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from Customer_Service_Assistant.service.schemas import (
    # Service types used for conversion
    CannotHandleSystemContext as SvcCannotHandleSystemContext,
    CanceledSystemContext as SvcCanceledSystemContext,
    ChatMessage as ServiceChatMessage,
    ChatRequest as ServiceChatRequest,
    ChatResponse as ServiceChatResponse,
    CollectSystemContext as SvcCollectSystemContext,
    CompletedSystemContext as SvcCompletedSystemContext,
    DialogueState as ServiceDialogueState,
    FocusedObject as ServiceFocusedObject,
    HistoryMessage as ServiceHistoryMessage,
    HistoryResponse as ServiceHistoryResponse,
    InterruptedSystemContext as SvcInterruptedSystemContext,
    Message as ServiceMessage,
    ObjectData as ServiceObjectData,
    ResumedSystemContext as SvcResumedSystemContext,
    Session as ServiceSession,
    StartedSystemContext as SvcStartedSystemContext,
    TaskContext as ServiceTaskContext,
    Turn as ServiceTurn,
)


# ============ helper ========================================================


def _exclude_none(d: dict) -> dict:
    """Return *d* with ``None``-valued keys removed (shallow)."""
    return {k: v for k, v in d.items() if v is not None}


# ===========================================================================
# Shared primitives
# ===========================================================================


class ObjectData(BaseModel):
    """Structured object attached to a message (API representation)."""

    type: str
    id: str
    title: Optional[str] = None
    attributes: dict = Field(default_factory=dict)

    def to_service(self) -> ServiceObjectData:
        return ServiceObjectData(
            type=self.type, id=self.id, title=self.title,
            attributes=self.attributes,
        )

    @classmethod
    def from_service(cls, obj: ServiceObjectData) -> "ObjectData":
        return cls(
            type=obj.type, id=obj.id, title=obj.title,
            attributes=obj.attributes,
        )


class FocusedObject(BaseModel):
    """A business object the user has explicitly focused on."""

    type: str
    id: str
    title: str = ""
    attributes: dict = Field(default_factory=dict)

    def to_service(self) -> ServiceFocusedObject:
        return ServiceFocusedObject(**self.model_dump())

    @classmethod
    def from_service(cls, obj: ServiceFocusedObject) -> "FocusedObject":
        return cls(**obj.model_dump())


# ===========================================================================
# Messages
# ===========================================================================


class Message(BaseModel):
    """A single message in a conversation (API representation)."""

    role: Literal["user", "bot"]
    text: Optional[str] = None
    object: Optional[ObjectData] = None

    def to_service(self) -> ServiceMessage:
        return ServiceMessage(
            role=self.role, text=self.text,
            object=self.object.to_service() if self.object else None,
        )

    @classmethod
    def from_service(cls, msg: ServiceMessage) -> "Message":
        return cls(
            role=msg.role, text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


# ===========================================================================
# Task contexts
# ===========================================================================


class TaskContext(BaseModel):
    """Execution progress of a user-facing business flow."""

    flow_id: str
    step_id: Optional[str] = None
    slots: dict = Field(default_factory=dict)

    def to_service(self) -> ServiceTaskContext:
        return ServiceTaskContext(**self.model_dump())

    @classmethod
    def from_service(cls, ctx: ServiceTaskContext) -> "TaskContext":
        return cls(**ctx.model_dump())


# ===========================================================================
# System contexts
# ===========================================================================


class SystemContext(BaseModel):
    """Base for system-initiated flows."""

    flow_id: str
    step_id: Optional[str] = None


class CollectSystemContext(SystemContext):
    flow_id: Literal["system_collect_information"] = "system_collect_information"
    slot_name: str = ""
    response: dict = Field(default_factory=dict)


class StartedSystemContext(SystemContext):
    flow_id: Literal["system_task_started"] = "system_task_started"
    started_flow_id: str = ""
    started_flow_name: str = ""


class ResumedSystemContext(SystemContext):
    flow_id: Literal["system_task_resumed"] = "system_task_resumed"
    resumed_flow_id: str = ""
    resumed_flow_name: str = ""


class CanceledSystemContext(SystemContext):
    flow_id: Literal["system_task_canceled"] = "system_task_canceled"
    canceled_flow_id: str = ""
    canceled_flow_name: str = ""


class InterruptedSystemContext(SystemContext):
    flow_id: Literal["system_task_interrupted"] = "system_task_interrupted"
    interrupted_flow_id: str = ""
    interrupted_flow_name: str = ""
    started_flow_id: str = ""
    started_flow_name: str = ""


class CannotHandleSystemContext(SystemContext):
    flow_id: Literal["system_cannot_handle"] = "system_cannot_handle"
    reason: Optional[str] = None


class CompletedSystemContext(SystemContext):
    flow_id: Literal["system_completed"] = "system_completed"
    previous_flow_name: str = ""


SystemContextUnion = Annotated[
    Union[
        CollectSystemContext,
        StartedSystemContext,
        ResumedSystemContext,
        CanceledSystemContext,
        InterruptedSystemContext,
        CannotHandleSystemContext,
        CompletedSystemContext,
    ],
    Field(discriminator="flow_id"),
]

# Map for converting a service SystemContextUnion to its API counterpart.
_SVC_SYS_CTX_MAP: dict = {
    "system_collect_information": CollectSystemContext,
    "system_task_started":       StartedSystemContext,
    "system_task_resumed":       ResumedSystemContext,
    "system_task_canceled":      CanceledSystemContext,
    "system_task_interrupted":   InterruptedSystemContext,
    "system_cannot_handle":      CannotHandleSystemContext,
    "system_completed":          CompletedSystemContext,
}


# ===========================================================================
# Sessions & turns
# ===========================================================================


class Turn(BaseModel):
    """One request–response round."""

    turn_id: str = Field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:12]}")
    input_message: Message
    assistant_messages: list[Message] = Field(default_factory=list)

    @classmethod
    def from_service(cls, turn: ServiceTurn) -> "Turn":
        return cls(
            turn_id=turn.turn_id,
            input_message=Message.from_service(turn.input_message),
            assistant_messages=[
                Message.from_service(m) for m in turn.assistant_messages
            ],
        )


class Session(BaseModel):
    """A group of turns within an activity time window."""

    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    started_at: float = Field(default_factory=time.time)
    last_activity_at: float = Field(default_factory=time.time)
    closed_at: Optional[float] = None
    turns: list[Turn] = Field(default_factory=list)

    @classmethod
    def from_service(cls, sess: ServiceSession) -> "Session":
        return cls(
            session_id=sess.session_id,
            started_at=sess.started_at,
            last_activity_at=sess.last_activity_at,
            closed_at=sess.closed_at,
            turns=[Turn.from_service(t) for t in sess.turns],
        )


# ===========================================================================
# Dialogue state
# ===========================================================================


class DialogueState(BaseModel):
    """The full state of a conversation (API representation)."""

    sender_id: str = ""

    active_task: Optional[TaskContext] = None
    paused_tasks: list[TaskContext] = Field(default_factory=list)

    active_system_flow: Optional[SystemContextUnion] = None

    focused_object: Optional[FocusedObject] = None

    sessions: list[Session] = Field(default_factory=list)
    current_session_id: Optional[str] = None

    pending_turn: Optional[Turn] = Field(default=None, exclude=True)

    @classmethod
    def from_json(cls, json_str: str) -> "DialogueState":
        data = json.loads(json_str)
        return cls.model_validate(data)

    def to_json(self) -> str:
        return self.model_dump_json(exclude={"pending_turn"})

    def to_service(self) -> ServiceDialogueState:
        svc_sys_flow: Optional = None
        if self.active_system_flow is not None:
            svc_sys_flow = type(self.active_system_flow)(  # same shape
                **self.active_system_flow.model_dump()
            )  # type: ignore[assignment]
        return ServiceDialogueState(
            sender_id=self.sender_id,
            active_task=self.active_task.to_service()
                if self.active_task else None,
            paused_tasks=[t.to_service() for t in self.paused_tasks],
            active_system_flow=svc_sys_flow,
            focused_object=self.focused_object.to_service()
                if self.focused_object else None,
            sessions=[  # Session.from_service actually takes svc→api; invert here
                ServiceSession(**s.model_dump(exclude={"turns"}),
                               turns=[ServiceTurn(**t.model_dump(
                                   exclude={"input_message", "assistant_messages"}),
                                   input_message=t.input_message.to_service(),
                                   assistant_messages=[m.to_service() for m in t.assistant_messages],
                               ) for t in s.turns])
                for s in self.sessions
            ],
            current_session_id=self.current_session_id,
        )

    @classmethod
    def from_service(cls, state: ServiceDialogueState) -> "DialogueState":
        svc_sys_flow: Optional[SystemContextUnion] = None
        if state.active_system_flow is not None:
            cls_name = _SVC_SYS_CTX_MAP.get(
                state.active_system_flow.flow_id, SystemContext
            )
            svc_sys_flow = cls_name(**state.active_system_flow.model_dump())
        return cls(
            sender_id=state.sender_id,
            active_task=TaskContext.from_service(state.active_task)
                if state.active_task else None,
            paused_tasks=[TaskContext.from_service(t) for t in state.paused_tasks],
            active_system_flow=svc_sys_flow,
            focused_object=FocusedObject.from_service(state.focused_object)
                if state.focused_object else None,
            sessions=[Session.from_service(s) for s in state.sessions],
            current_session_id=state.current_session_id,
        )


# ===========================================================================
# POST /api/chat
# ===========================================================================


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
        return ServiceChatRequest(
            sender_id=self.sender_id, text=self.text,
            object=self.object.to_service() if self.object else None,
            message_id=self.message_id,
        )

    @classmethod
    def from_service(cls, req: ServiceChatRequest) -> "ChatRequest":
        return cls(
            sender_id=req.sender_id, text=req.text,
            object=ObjectData.from_service(req.object) if req.object else None,
            message_id=req.message_id,
        )

    def to_service_message(self) -> ServiceMessage:
        return ServiceMessage(
            role="user", text=self.text,
            object=self.object.to_service() if self.object else None,
        )


class ChatMessage(BaseModel):
    """A single reply item from the bot (API representation)."""

    text: Optional[str] = None
    object: Optional[ObjectData] = None

    def to_service(self) -> ServiceChatMessage:
        return ServiceChatMessage(
            text=self.text,
            object=self.object.to_service() if self.object else None,
        )

    @classmethod
    def from_service(cls, msg: ServiceChatMessage) -> "ChatMessage":
        return cls(
            text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )

    @classmethod
    def from_service_message(cls, msg: ServiceMessage) -> "ChatMessage":
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
        return ServiceChatResponse(
            sender_id=self.sender_id, message_id=self.message_id,
            messages=[m.to_service() for m in self.messages],
        )

    @classmethod
    def from_service(cls, resp: ServiceChatResponse) -> "ChatResponse":
        return cls(
            sender_id=resp.sender_id, message_id=resp.message_id,
            messages=[ChatMessage.from_service(m) for m in resp.messages],
        )


# ===========================================================================
# GET /api/chat/history
# ===========================================================================


class HistoryMessage(BaseModel):
    """A single message in the conversation history (API representation)."""

    role: str  # "user" | "bot"
    text: Optional[str] = None
    object: Optional[ObjectData] = None

    def to_service(self) -> ServiceHistoryMessage:
        return ServiceHistoryMessage(
            role=self.role, text=self.text,
            object=self.object.to_service() if self.object else None,
        )

    @classmethod
    def from_service(cls, msg: ServiceHistoryMessage) -> "HistoryMessage":
        return cls(
            role=msg.role, text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )

    @classmethod
    def from_service_message(cls, msg: ServiceMessage) -> "HistoryMessage":
        return cls(
            role=msg.role, text=msg.text,
            object=ObjectData.from_service(msg.object) if msg.object else None,
        )


class HistoryResponse(BaseModel):
    """Response returned when querying chat history."""

    sender_id: str
    messages: list[HistoryMessage]

    def to_service(self) -> ServiceHistoryResponse:
        return ServiceHistoryResponse(
            sender_id=self.sender_id,
            messages=[m.to_service() for m in self.messages],
        )

    @classmethod
    def from_service(cls, resp: ServiceHistoryResponse) -> "HistoryResponse":
        return cls(
            sender_id=resp.sender_id,
            messages=[HistoryMessage.from_service(m) for m in resp.messages],
        )
