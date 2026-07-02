"""FastAPI router for the chat API — POST /api/chat and GET /api/chat/history."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query

from Customer_Service_Assistant.api.dependencies import DialogueService, get_dialogue_service
from Customer_Service_Assistant.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
)
from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import Message

router = APIRouter(prefix="/api")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    dialogue_service: DialogueService = Depends(get_dialogue_service),
) -> ChatResponse:
    """Process a user message and return the bot's reply."""

    message_id = req.message_id or f"msg_{uuid.uuid4().hex[:12]}"

    # Load existing conversation state
    state = await dialogue_service.load_state(req.sender_id)

    # Convert API request to a service-layer user message and append
    user_msg = req.to_service_message()
    state.messages.append(user_msg)

    # Build prompt and call LLM
    prompt_messages = dialogue_service.build_prompt(state.messages)
    llm_response = await llm.ainvoke(prompt_messages)
    bot_text = llm_response.content.strip() if llm_response.content else ""

    # Record bot reply as a service-layer message
    bot_msg = Message(role="bot", text=bot_text, object=None)
    state.messages.append(bot_msg)

    # Persist updated state
    await dialogue_service.save_state(req.sender_id, state)

    return ChatResponse(
        sender_id=req.sender_id,
        message_id=message_id,
        messages=[ChatMessage.from_service_message(bot_msg)],
    )


@router.get("/chat/history", response_model=HistoryResponse)
async def chat_history(
    sender_id: str = Query(..., description="用户唯一标识"),
    dialogue_service: DialogueService = Depends(get_dialogue_service),
) -> HistoryResponse:
    """Return the chat history for a given user."""

    state = await dialogue_service.load_state(sender_id)

    messages = [HistoryMessage.from_service_message(m) for m in state.messages]

    return HistoryResponse(sender_id=sender_id, messages=messages)
