"""Domain schemas for the customer service layer.

These models represent the internal conversational domain and are
independent of any transport layer (HTTP, WebSocket, etc.).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Literal, Optional, Union

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


class FocusedObject(BaseModel):
    """A business object the user has explicitly focused on (e.g. by clicking
    an order card in the UI).  Subsequent flows can auto-fill slots from it.
    """

    type: str
    id: str
    title: str = ""
    attributes: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "bot"]
    text: Optional[str] = None
    object: Optional[ObjectData] = None


# ---------------------------------------------------------------------------
# Task contexts
# ---------------------------------------------------------------------------


class TaskContext(BaseModel):
    """Execution progress of a user-facing business flow."""

    flow_id: str
    step_id: Optional[str] = None
    slots: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# System contexts (7 sub-types discriminated by flow_id)
# ---------------------------------------------------------------------------


class SystemContext(BaseModel):
    """Base for system-initiated flows."""

    flow_id: str
    step_id: Optional[str] = None


class CollectSystemContext(SystemContext):
    """System needs to ask the user for a missing slot value."""

    flow_id: Literal["system_collect_information"] = "system_collect_information"
    slot_name: str = ""
    response: dict = Field(default_factory=dict)


class StartedSystemContext(SystemContext):
    """Notify the user that a task has started."""

    flow_id: Literal["system_task_started"] = "system_task_started"
    started_flow_id: str = ""
    started_flow_name: str = ""


class ResumedSystemContext(SystemContext):
    """Notify the user that a paused task has resumed."""

    flow_id: Literal["system_task_resumed"] = "system_task_resumed"
    resumed_flow_id: str = ""
    resumed_flow_name: str = ""


class CanceledSystemContext(SystemContext):
    """Notify the user that a task has been canceled."""

    flow_id: Literal["system_task_canceled"] = "system_task_canceled"
    canceled_flow_id: str = ""
    canceled_flow_name: str = ""


class InterruptedSystemContext(SystemContext):
    """Notify the user that a task was interrupted by a new task."""

    flow_id: Literal["system_task_interrupted"] = "system_task_interrupted"
    interrupted_flow_id: str = ""
    interrupted_flow_name: str = ""
    started_flow_id: str = ""
    started_flow_name: str = ""


class CannotHandleSystemContext(SystemContext):
    """System cannot handle the user's request."""

    flow_id: Literal["system_cannot_handle"] = "system_cannot_handle"
    reason: Optional[str] = None


class CompletedSystemContext(SystemContext):
    """Notify the user that a task has completed."""

    flow_id: Literal["system_completed"] = "system_completed"
    previous_flow_name: str = ""


# Discriminated union — Pydantic uses flow_id to determine the subclass.
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


# ---------------------------------------------------------------------------
# Sessions & turns
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    """One request–response round."""

    turn_id: str = Field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:12]}")
    input_message: Message
    assistant_messages: list[Message] = Field(default_factory=list)


class Session(BaseModel):
    """A group of turns within an activity time window."""

    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    started_at: float = Field(default_factory=time.time)
    last_activity_at: float = Field(default_factory=time.time)
    closed_at: Optional[float] = None
    turns: list[Turn] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dialogue state (top-level aggregate)
# ---------------------------------------------------------------------------


class DialogueState(BaseModel):
    """The full state of a conversation, persisted across turns.

    Attributes are grouped into four categories:

    * **User task context** — ``active_task`` / ``paused_tasks``
    * **System task context** — ``active_system_flow``
    * **Focused object** — ``focused_object``
    * **Conversation history** — ``sessions`` / ``current_session_id``
    """

    sender_id: str = ""

    # -- user task context --------------------------------------------------
    active_task: Optional[TaskContext] = None
    paused_tasks: list[TaskContext] = Field(default_factory=list)

    # -- system task context ------------------------------------------------
    active_system_flow: Optional[SystemContextUnion] = None

    # -- focused object -----------------------------------------------------
    focused_object: Optional[FocusedObject] = None

    # -- conversation history -----------------------------------------------
    sessions: list[Session] = Field(default_factory=list)
    current_session_id: Optional[str] = None

    # -- in-flight turn (transient, not persisted) --------------------------
    pending_turn: Optional[Turn] = Field(default=None, exclude=True)

    # -- helpers ------------------------------------------------------------

    @property
    def current_session(self) -> Optional[Session]:
        """Return the currently active session, if any."""
        if self.current_session_id is None:
            return None
        for s in self.sessions:
            if s.session_id == self.current_session_id:
                return s
        return None

    @property
    def current_messages(self) -> list[Message]:
        """Flat list of messages from the current session + pending turn.

        Used by the Engine to build the LLM prompt for this turn.
        """
        messages: list[Message] = []
        session = self.current_session
        if session:
            for turn in session.turns:
                messages.append(turn.input_message)
                messages.extend(turn.assistant_messages)
        # Include the user message that's currently being processed
        if self.pending_turn:
            messages.append(self.pending_turn.input_message)
        return messages

    def ensure_session(self) -> Session:
        """Return the current session, creating one if none is active."""
        session = self.current_session
        if session is None:
            session = Session()
            self.sessions.append(session)
            self.current_session_id = session.session_id
        return session

    # -- persistence --------------------------------------------------------

    @classmethod
    def from_json(cls, json_str: str) -> "DialogueState":
        """Deserialize from a JSON string (as stored in the database)."""
        data = json.loads(json_str)
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to a JSON string for database persistence."""
        return self.model_dump_json(exclude={"pending_turn"})


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
