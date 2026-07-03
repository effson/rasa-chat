"""ChitChatHandler — lightweight handler for casual conversation.

Per the design doc (KnowledgeHandler&ChitChatHandler 设计.md §2):

    Read recent dialogue history, combine with the user's last message,
    call the LLM, and return a natural chitchat reply.

This is the simplest handler — no knowledge retrieval, no task state.
"""

from __future__ import annotations

from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import DialogueState, Message

# ---------------------------------------------------------------------------
# System prompt for chitchat — tuned for friendly, concise replies
# ---------------------------------------------------------------------------

CHITCHAT_SYSTEM_PROMPT = (
    "你是一个电商客服助手，名字叫小慧。用户正在和你闲聊。"
    "请用友好、自然、亲切的语气回复。"
    "回复要简短，一般不超过两句话。"
    "你可以适当地问候、表达关心，但不要过度热情。"
    "如果用户表示感谢，简单回应即可。"
    "如果用户表达负面情绪，先安抚再尝试帮助。"
)


class ChitChatResponder:
    """Build the chitchat prompt and call the LLM to generate a reply."""

    def __init__(self) -> None:
        self._llm = llm

    async def respond(
        self,
        recent_messages: list[Message],
        user_text: str,
    ) -> Message:
        """Generate a natural chitchat reply.

        Parameters
        ----------
        recent_messages:
            The flat message history from the current session (including
            the in-flight user message via ``state.current_messages``).
        user_text:
            The user's last text message, passed separately so the caller
            can control whether the pending turn is included.
        """
        messages: list[dict] = [
            {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT}
        ]

        # -- History ----------------------------------------------------------
        for msg in recent_messages:
            role = "user" if msg.role == "user" else "assistant"
            if msg.text:
                messages.append({"role": role, "content": msg.text})

        # -- Current user message (may already be in recent_messages, but
        #    being explicit ensures the LLM focuses on the latest input) ------
        messages.append({"role": "user", "content": user_text})

        # -- LLM call ---------------------------------------------------------
        response = await self._llm.ainvoke(messages)
        text = response.content.strip() if response.content else ""

        return Message(role="bot", text=text)


class ChitChatHandler:
    """Orchestrate chitchat processing.

    Design doc §2 — ChitChatHandler is the entry point for casual
    conversation.  It delegates prompt construction and LLM interaction
    to ``ChitChatResponder``.
    """

    def __init__(self) -> None:
        self._responder = ChitChatResponder()

    async def handle(self, state: DialogueState, user_text: str) -> Message:
        """Process a chitchat message and return the bot's reply.

        The handler reads ``state.current_messages`` for conversation
        history and passes it together with *user_text* to the responder.
        """
        return await self._responder.respond(
            recent_messages=state.current_messages,
            user_text=user_text,
        )