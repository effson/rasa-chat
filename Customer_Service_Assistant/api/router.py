"""FastAPI router for the chat API — POST /api/chat and GET /api/chat/history."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from Customer_Service_Assistant.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
)
from Customer_Service_Assistant.infrastructure.db import get_async_session
from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    Message,
)

router = APIRouter(prefix="/api")


def _build_prompt(messages: list[Message]) -> list[dict]:
    """Build an LLM prompt from the full conversation history (service-layer models).

    The history already contains the latest user message as its last entry.
    """
    system = (
        "你是一个电商客服助手。请根据用户的问题提供帮助。"
        "回复要简洁、友好、专业。"
    )

    prompt: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = "user" if msg.role == "user" else "assistant"
        parts: list[str] = []
        if msg.text:
            parts.append(msg.text)
        if msg.object:
            parts.append(f"[{msg.object.type}: {msg.object.title or msg.object.id}]")
        prompt.append({"role": role, "content": "\n".join(parts)})

    return prompt


async def _load_state(
    session: AsyncSession, sender_id: str
) -> DialogueState:
    """Load dialogue state from the database, or return an empty state."""
    result = await session.execute(
        text("SELECT state_json FROM dialogue_states WHERE sender_id = :sid"),
        {"sid": sender_id},
    )
    row = result.fetchone()
    if row is None:
        return DialogueState()
    return DialogueState.from_json(row.state_json)


async def _save_state(
    session: AsyncSession, sender_id: str, state: DialogueState
) -> None:
    """Persist dialogue state to the database (upsert)."""
    state_json = state.to_json()
    await session.execute(
        text(
            "INSERT INTO dialogue_states (sender_id, state_json) "
            "VALUES (:sid, :state) "
            "ON DUPLICATE KEY UPDATE state_json = VALUES(state_json)"
        ),
        {"sid": sender_id, "state": state_json},
    )
    await session.commit()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> ChatResponse:
    """Process a user message and return the bot's reply."""

    message_id = req.message_id or f"msg_{uuid.uuid4().hex[:12]}"

    # Load existing conversation state (service-layer domain model)
    state = await _load_state(session, req.sender_id)

    # Convert API request to a service-layer user message and append
    user_msg = req.to_service_message()
    state.messages.append(user_msg)

    # Build prompt and call LLM
    prompt_messages = _build_prompt(state.messages)
    llm_response = await llm.ainvoke(prompt_messages)
    bot_text = llm_response.content.strip() if llm_response.content else ""

    # Record bot reply as a service-layer message
    bot_msg = Message(role="bot", text=bot_text, object=None)
    state.messages.append(bot_msg)

    # Persist updated state
    await _save_state(session, req.sender_id, state)

    return ChatResponse(
        sender_id=req.sender_id,
        message_id=message_id,
        messages=[ChatMessage.from_service_message(bot_msg)],
    )


@router.get("/chat/history", response_model=HistoryResponse)
async def chat_history(
    sender_id: str = Query(..., description="用户唯一标识"),
    session: AsyncSession = Depends(get_async_session),
) -> HistoryResponse:
    """Return the chat history for a given user."""

    state = await _load_state(session, sender_id)

    messages = [HistoryMessage.from_service_message(m) for m in state.messages]

    return HistoryResponse(sender_id=sender_id, messages=messages)
