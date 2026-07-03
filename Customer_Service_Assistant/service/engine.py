"""Engine layer — core dispatcher that orchestrates a single turn.

Flow:  Prepare Session → Planning → Route → Track (Task / Knowledge / Chitchat) → Infra
"""

from __future__ import annotations

import time

from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import DialogueState, Message

# ---------------------------------------------------------------------------
# Session timeout (seconds).  If the user's last activity was longer ago than
# this, the current session is considered expired and the engine starts a
# fresh session after clearing all running state.
# ---------------------------------------------------------------------------
SESSION_TIMEOUT = 30 * 60  # 30 minutes


class DialogueEngine:
    """Core dispatcher for one conversation turn.

    Coordinates Planning (prompt building / validation), Routing (track
    selection), and delegates generation to the appropriate track.  When
    Planning and Route layers are built out they will be injected here;
    for now the engine handles everything inline.
    """

    def __init__(self) -> None:
        self._llm = llm

    # -- public API ----------------------------------------------------------

    async def run(self, state: DialogueState) -> Message:
        """Process *state* (with the latest user message already appended)
        and return the bot's reply as a ``Message``.

        Steps (per DialogueEngine 设计.md):

        1. Prepare Session — check timeout, create fresh session if needed
        2. Planning — build the prompt for this turn
        3. Route + Generate — for now, a single LLM call
        4. Return the bot message
        """
        # 1. Prepare Session
        self._prepare_session(state)

        # 2. Planning — build the prompt for this turn
        prompt = self._plan(state)

        # 3. Route + Generate — for now, a single LLM call
        bot_text = await self._generate(prompt)

        # 4. Return the bot message
        return Message(role="bot", text=bot_text)

    # -- Session preparation -------------------------------------------------

    def _prepare_session(self, state: DialogueState) -> None:
        """Ensure the current session is still valid (step 1 in the design doc).

        * If there is **no** current session, create one.
        * If the current session **has not** timed out, keep it.
        * If the current session **has** timed out:

          1. Mark the old session as closed (``closed_at``).
          2. Clear all running state — ``active_task``, ``paused_tasks``,
             ``active_system_flow``, ``focused_object``.
          3. Create a fresh session.
        """
        now = time.time()
        session = state.current_session

        if session is not None:
            elapsed = now - session.last_activity_at
            if elapsed <= SESSION_TIMEOUT:
                # Session is still active — nothing to do
                return
            # Session timed out — close it and clear running state
            session.closed_at = now
            state.current_session_id = None
            state.active_task = None
            state.paused_tasks = []
            state.active_system_flow = None
            state.focused_object = None

        # Create a fresh session (handles both first-time and post-timeout)
        state.ensure_session()

    # -- Planning (stub — will become its own layer) -------------------------

    def _plan(self, state: DialogueState) -> list[dict]:
        """Build the LLM prompt from the full conversation history."""
        system = (
            "你是一个电商客服助手。请根据用户的问题提供帮助。"
            "回复要简洁、友好、专业。"
        )

        prompt: list[dict] = [{"role": "system", "content": system}]

        for msg in state.current_messages:
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

    # -- Route / Generate (stub — will fan out to tracks) --------------------

    async def _generate(self, prompt: list[dict]) -> str:
        """Call the LLM and return the response text.

        When Route is built, this will become a dispatch to
        Task / Knowledge / Chitchat tracks.
        """
        response = await self._llm.ainvoke(prompt)
        return response.content.strip() if response.content else ""
