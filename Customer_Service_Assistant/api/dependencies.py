"""FastAPI dependency injection — provides DialogueService and other callables."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from Customer_Service_Assistant.infrastructure.db import get_async_session
from Customer_Service_Assistant.service.schemas import DialogueState, Message


class DialogueService:
    """Encapsulates dialogue state persistence and prompt building.

    Injected as a FastAPI dependency so endpoint handlers stay thin.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- state persistence --------------------------------------------------

    async def load_state(self, sender_id: str) -> DialogueState:
        """Load dialogue state from the database, or return an empty state."""
        result = await self._session.execute(
            text("SELECT state_json FROM dialogue_states WHERE sender_id = :sid"),
            {"sid": sender_id},
        )
        row = result.fetchone()
        if row is None:
            return DialogueState()
        return DialogueState.from_json(row.state_json)

    async def save_state(self, sender_id: str, state: DialogueState) -> None:
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
    def build_prompt(messages: list[Message]) -> list[dict]:
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


async def get_dialogue_service(
    session: AsyncSession = Depends(get_async_session),
) -> DialogueService:
    """FastAPI dependency that yields a DialogueService backed by a DB session."""
    return DialogueService(session)
