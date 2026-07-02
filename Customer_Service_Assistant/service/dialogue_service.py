"""Dialogue service — orchestrates message processing end-to-end."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import (
    ChatMessage,
    ChatResponse,
    DialogueState,
    Message,
)


class DialogueService:
    """Encapsulates dialogue state persistence, prompt building, and LLM
    orchestration.

    Injected as a FastAPI dependency so endpoint handlers stay thin.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- message processing --------------------------------------------------

    async def process_message(
        self, sender_id: str, user_message: Message, message_id: str,
    ) -> ChatResponse:
        """Process an incoming user message end-to-end.

        1. Load conversation state from the database
        2. Append the user message
        3. Build the LLM prompt
        4. Call the LLM
        5. Append the bot reply
        6. Persist the updated state
        7. Return a service-layer ``ChatResponse``
        """
        state = await self._load_state(sender_id)

        # Append user message
        state.messages.append(user_message)

        # Build prompt and call LLM
        prompt = self._build_prompt(state.messages)
        llm_response = await llm.ainvoke(prompt)
        bot_text = llm_response.content.strip() if llm_response.content else ""

        # Record bot reply
        bot_msg = Message(role="bot", text=bot_text, object=None)
        state.messages.append(bot_msg)

        # Persist
        await self._save_state(sender_id, state)

        return ChatResponse(
            sender_id=sender_id,
            message_id=message_id,
            messages=[ChatMessage(text=bot_text, object=None)],
        )

    # -- state persistence --------------------------------------------------

    async def _load_state(self, sender_id: str) -> DialogueState:
        result = await self._session.execute(
            text("SELECT state_json FROM dialogue_states WHERE sender_id = :sid"),
            {"sid": sender_id},
        )
        row = result.fetchone()
        if row is None:
            return DialogueState()
        return DialogueState.from_json(row.state_json)

    async def _save_state(self, sender_id: str, state: DialogueState) -> None:
        """Persist dialogue state to the database (upsert)."""
        state_json = state.to_json()
        await self._session.execute(
            text(
                "INSERT INTO dialogue_states (sender_id, state_json) "
                "VALUES (:sid, :state) "
                "ON DUPLICATE KEY UPDATE state_json = VALUES(state_json)"
            ),
            {"sid": sender_id, "state": state_json},
        )
        await self._session.commit()

    # -- prompt building ----------------------------------------------------

    @staticmethod
    def _build_prompt(messages: list[Message]) -> list[dict]:
        """Build an LLM prompt from the full conversation history.

        The history should already contain the latest user message as its
        last entry.
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
                parts.append(
                    f"[{msg.object.type}: {msg.object.title or msg.object.id}]"
                )
            prompt.append({"role": role, "content": "\n".join(parts)})

        return prompt
