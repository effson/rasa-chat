"""Engine layer — core dispatcher that orchestrates a single turn.

Flow:  Planning → Route → Track (Task / Knowledge / Chitchat) → Infra
"""

from __future__ import annotations

from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import DialogueState, Message


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
        """
        # 1. Planning — build the prompt for this turn
        prompt = self._plan(state)

        # 2. Route + Generate — for now, a single LLM call
        bot_text = await self._generate(prompt)

        # 3. Return the bot message
        return Message(role="bot", text=bot_text)

    # -- Planning (stub — will become its own layer) -------------------------

    def _plan(self, state: DialogueState) -> list[dict]:
        """Build the LLM prompt from the full conversation history."""
        system = (
            "你是一个电商客服助手。请根据用户的问题提供帮助。"
            "回复要简洁、友好、专业。"
        )

        prompt: list[dict] = [{"role": "system", "content": system}]

        for msg in state.messages:
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
