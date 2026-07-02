"""FastAPI router for the chat API — POST /api/chat and GET /api/chat/history."""

from __future__ import annotations

import json
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
    ObjectData,
)
from Customer_Service_Assistant.infrastructure.db import get_async_session
from Customer_Service_Assistant.infrastructure.llm import llm

router = APIRouter(prefix="/api")


def _build_prompt(history: list[dict]) -> list[dict]:
    """Build an LLM prompt from the full conversation history.

    The history already contains the latest user message as its last entry.
    """
    system = (
        "你是一个电商客服助手。请根据用户的问题提供帮助。"
        "回复要简洁、友好、专业。"
    )

    messages: list[dict] = [{"role": "system", "content": system}]

    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        parts: list[str] = []
        if msg.get("text"):
            parts.append(msg["text"])
        if msg.get("object"):
            obj = msg["object"]
            parts.append(f"[{obj['type']}: {obj.get('title', obj['id'])}]")
        messages.append({"role": role, "content": "\n".join(parts)})

    return messages


async def _load_state(
    session: AsyncSession, sender_id: str
) -> list[dict]:
    """Load dialogue state from the database, or return an empty list if not found."""
    result = await session.execute(
        text("SELECT state_json FROM dialogue_states WHERE sender_id = :sid"),
        {"sid": sender_id},
    )
    row = result.fetchone()
    if row is None:
        return []
    return json.loads(row.state_json).get("messages", [])


async def _save_state(
    session: AsyncSession, sender_id: str, messages: list[dict]
) -> None:
    """Persist dialogue state to the database (upsert)."""
    state_json = json.dumps({"messages": messages}, ensure_ascii=False)
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

    # Load existing conversation history
    history = await _load_state(session, req.sender_id)

    # Record the user message
    user_msg: dict = {
        "role": "user",
        "text": req.text,
        "object": req.object.model_dump() if req.object else None,
    }
    history.append(user_msg)

    # Build prompt (history now includes the just-recorded user message) and call LLM
    prompt_messages = _build_prompt(history)
    llm_response = await llm.ainvoke(prompt_messages)
    bot_text = llm_response.content.strip() if llm_response.content else ""

    # Record the bot message
    bot_msg: dict = {
        "role": "bot",
        "text": bot_text,
        "object": None,
    }
    history.append(bot_msg)

    # Persist updated state
    await _save_state(session, req.sender_id, history)

    return ChatResponse(
        sender_id=req.sender_id,
        message_id=message_id,
        messages=[ChatMessage(text=bot_text, object=None)],
    )


@router.get("/chat/history", response_model=HistoryResponse)
async def chat_history(
    sender_id: str = Query(..., description="用户唯一标识"),
    session: AsyncSession = Depends(get_async_session),
) -> HistoryResponse:
    """Return the chat history for a given user."""

    history = await _load_state(session, sender_id)

    messages = [
        HistoryMessage(
            role=m["role"],
            text=m.get("text"),
            object=ObjectData(**m["object"]) if m.get("object") else None,
        )
        for m in history
    ]

    return HistoryResponse(sender_id=sender_id, messages=messages)
