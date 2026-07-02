"""Dialogue service — thin orchestration layer above Engine and Repository."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from Customer_Service_Assistant.service.engine import DialogueEngine
from Customer_Service_Assistant.service.schemas import (
    ChatMessage,
    ChatResponse,
    DialogueState,
    Message,
)


class DialogueService:
    """Orchestrates a message turn: load state → engine → save state.

    Injected as a FastAPI dependency so endpoint handlers stay thin.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._engine = DialogueEngine()

    # -- message processing --------------------------------------------------

    async def process_message(
        self, sender_id: str, user_message: Message, message_id: str,
    ) -> ChatResponse:
        """Process an incoming user message end-to-end.

        1. Load conversation state (Repository)
        2. Append the user message
        3. Run the DialogueEngine to get the bot reply
        4. Append the bot reply
        5. Persist (Repository)
        6. Return a service-layer ``ChatResponse``
        """
        # Repository — load
        state = await self._load_state(sender_id)

        # Append user message
        state.messages.append(user_message)

        # Engine — core dispatch
        bot_msg = await self._engine.run(state)
        state.messages.append(bot_msg)

        # Repository — save
        await self._save_state(sender_id, state)

        return ChatResponse(
            sender_id=sender_id,
            message_id=message_id,
            messages=[ChatMessage(text=bot_msg.text, object=None)],
        )

    # -- Repository (stub — will become its own layer) -----------------------

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
