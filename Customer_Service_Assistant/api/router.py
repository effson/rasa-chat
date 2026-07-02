"""FastAPI router for the chat API — POST /api/chat and GET /api/chat/history."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from Customer_Service_Assistant.api.dependencies import DialogueService, get_dialogue_service
from Customer_Service_Assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
)
from Customer_Service_Assistant.infrastructure.db import get_async_session
from Customer_Service_Assistant.service.schemas import DialogueState

router = APIRouter(prefix="/api")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    dialogue_service: DialogueService = Depends(get_dialogue_service),
) -> ChatResponse:
    """Process a user message and return the bot's reply."""
    message_id = req.message_id or f"msg_{uuid.uuid4().hex[:12]}"

    # Delegate to the service layer
    service_result = await dialogue_service.process_message(
        sender_id=req.sender_id,
        user_message=req.to_service_message(),
        message_id=message_id,
    )

    # Convert service-layer response to API-layer schema
    return ChatResponse.from_service(service_result)


@router.get("/chat/history", response_model=HistoryResponse)
async def chat_history(
    sender_id: str = Query(..., description="用户唯一标识"),
    session: AsyncSession = Depends(get_async_session),
) -> HistoryResponse:
    """Return the chat history for a given user."""

    result = await session.execute(
        text("SELECT state_json FROM dialogue_states WHERE sender_id = :sid"),
        {"sid": sender_id},
    )
    row = result.fetchone()
    state = DialogueState.from_json(row.state_json) if row else DialogueState()

    # Flatten all messages from all sessions
    all_messages: list = []
    for session in state.sessions:
        for turn in session.turns:
            all_messages.append(turn.input_message)
            all_messages.extend(turn.assistant_messages)

    messages = [HistoryMessage.from_service_message(m) for m in all_messages]

    return HistoryResponse(sender_id=sender_id, messages=messages)
