"""Unit tests for ChitChatHandler and ChitChatResponder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Customer_Service_Assistant.service.chitchat_handler import (
    CHITCHAT_SYSTEM_PROMPT,
    ChitChatHandler,
    ChitChatResponder,
)
from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    Message,
    Session,
    Turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(messages: list[Message] | None = None) -> DialogueState:
    """Build a DialogueState with a session containing the given messages."""
    state = DialogueState(sender_id="u1")
    if messages:
        session = Session()
        # Put pairs of messages into turns
        for i in range(0, len(messages), 2):
            turn_messages = messages[i : i + 2]
            if turn_messages:
                user_msg = turn_messages[0]
                bot_msgs = turn_messages[1:] if len(turn_messages) > 1 else []
                turn = Turn(input_message=user_msg, assistant_messages=bot_msgs)
                session.turns.append(turn)
        state.sessions.append(session)
        state.current_session_id = session.session_id
    return state


def make_mock_llm(response_text: str = "你好！今天天气不错。") -> MagicMock:
    """Return a mock LLM with a preset response."""
    resp = MagicMock()
    resp.content = response_text
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=resp)
    return mock_llm


# ===================================================================
# ChitChatResponder
# ===================================================================


class TestChitChatResponder:
    @pytest.mark.asyncio
    async def test_returns_message_with_llm_response(self):
        mock_llm = make_mock_llm("嗨！有什么可以帮你的？")
        responder = ChitChatResponder()
        responder._llm = mock_llm  # inject mock

        result = await responder.respond(
            recent_messages=[],
            user_text="你好",
        )

        assert isinstance(result, Message)
        assert result.role == "bot"
        assert result.text == "嗨！有什么可以帮你的？"

    @pytest.mark.asyncio
    async def test_includes_system_prompt_in_llm_call(self):
        mock_llm = make_mock_llm()
        responder = ChitChatResponder()
        responder._llm = mock_llm

        await responder.respond(recent_messages=[], user_text="嗨")

        # Verify system prompt was the first message
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == CHITCHAT_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_includes_history_in_llm_call(self):
        mock_llm = make_mock_llm()
        responder = ChitChatResponder()
        responder._llm = mock_llm

        history = [
            Message(role="user", text="今天好忙"),
            Message(role="bot", text="辛苦了，注意休息哦"),
        ]

        await responder.respond(recent_messages=history, user_text="谢谢你")

        call_args = mock_llm.ainvoke.call_args[0][0]
        # System + 2 history + 1 current = 4 messages
        assert len(call_args) == 4
        assert call_args[1]["role"] == "user"
        assert call_args[1]["content"] == "今天好忙"
        assert call_args[2]["role"] == "assistant"
        assert call_args[2]["content"] == "辛苦了，注意休息哦"
        assert call_args[3]["role"] == "user"
        assert call_args[3]["content"] == "谢谢你"

    @pytest.mark.asyncio
    async def test_skips_messages_without_text(self):
        """Messages with only an object (no text) should be skipped."""
        mock_llm = make_mock_llm()
        responder = ChitChatResponder()
        responder._llm = mock_llm

        history = [
            Message(role="user", text=None, object=None),  # no text
            Message(role="user", text="hi"),
        ]

        await responder.respond(recent_messages=history, user_text="hello")

        call_args = mock_llm.ainvoke.call_args[0][0]
        # System + 1 text message + 1 current = 3 (the None-text msg is skipped)
        assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        """When LLM returns None/empty, responder returns empty string."""
        mock_llm = make_mock_llm("")
        responder = ChitChatResponder()
        responder._llm = mock_llm

        result = await responder.respond(recent_messages=[], user_text="嗨")

        assert result.text == ""

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_llm_response(self):
        mock_llm = make_mock_llm("  你好，欢迎光临！  \n")
        responder = ChitChatResponder()
        responder._llm = mock_llm

        result = await responder.respond(recent_messages=[], user_text="你好")

        assert result.text == "你好，欢迎光临！"

    @pytest.mark.asyncio
    async def test_user_message_is_last(self):
        """The current user text always appears last in the prompt."""
        mock_llm = make_mock_llm()
        responder = ChitChatResponder()
        responder._llm = mock_llm

        history = [
            Message(role="user", text="旧消息1"),
            Message(role="bot", text="旧回复1"),
        ]

        await responder.respond(recent_messages=history, user_text="最新消息")

        call_args = mock_llm.ainvoke.call_args[0][0]
        assert call_args[-1]["role"] == "user"
        assert call_args[-1]["content"] == "最新消息"


# ===================================================================
# ChitChatHandler
# ===================================================================


class TestChitChatHandler:
    @pytest.mark.asyncio
    async def test_handle_passes_state_messages_to_responder(self):
        mock_llm = make_mock_llm("嗨！")
        handler = ChitChatHandler()
        handler._responder._llm = mock_llm

        state = make_state([
            Message(role="user", text="你好"),
            Message(role="bot", text="你好，欢迎！"),
        ])

        result = await handler.handle(state, "今天天气真好")

        assert result.role == "bot"
        assert result.text == "嗨！"

        # Verify history was included
        call_args = mock_llm.ainvoke.call_args[0][0]
        # System + 2 history + current = 4
        assert len(call_args) == 4
        assert call_args[1]["content"] == "你好"
        assert call_args[2]["content"] == "你好，欢迎！"
        assert call_args[3]["content"] == "今天天气真好"

    @pytest.mark.asyncio
    async def test_handle_with_empty_history(self):
        mock_llm = make_mock_llm("你好！我是小慧。")
        handler = ChitChatHandler()
        handler._responder._llm = mock_llm

        state = make_state()  # no messages
        result = await handler.handle(state, "嗨")

        assert result.text == "你好！我是小慧。"

    @pytest.mark.asyncio
    async def test_handle_uses_current_messages_only(self):
        """ChitChatHandler should use state.current_messages, not all sessions."""
        mock_llm = make_mock_llm("回复")
        handler = ChitChatHandler()
        handler._responder._llm = mock_llm

        # Create a state with multiple sessions — only current one counts
        state = DialogueState(sender_id="u1")
        old_session = Session()
        old_session.turns.append(
            Turn(
                input_message=Message(role="user", text="旧会话消息"),
                assistant_messages=[Message(role="bot", text="旧会话回复")],
            )
        )
        state.sessions.append(old_session)

        current_session = Session()
        current_session.turns.append(
            Turn(
                input_message=Message(role="user", text="新会话消息"),
                assistant_messages=[Message(role="bot", text="新会话回复")],
            )
        )
        state.sessions.append(current_session)
        state.current_session_id = current_session.session_id

        await handler.handle(state, "当前消息")

        call_args = mock_llm.ainvoke.call_args[0][0]
        # System + 2 current session + 1 current user = 4
        # Old session messages should NOT appear
        assert len(call_args) == 4
        assert call_args[1]["content"] == "新会话消息"
        assert call_args[2]["content"] == "新会话回复"

    @pytest.mark.asyncio
    async def test_default_construction_uses_real_llm(self):
        """Smoke test: ChitChatHandler can be constructed without arguments."""
        handler = ChitChatHandler()
        assert handler._responder is not None
        assert handler._responder._llm is not None


# ===================================================================
# Integration: engine dispatches chitchat to handler
# ===================================================================


class TestEngineChitchatDispatch:
    @pytest.mark.asyncio
    async def test_engine_routes_chitchat_to_handler(self):
        """When TurnPlan direction is chitchat, the engine calls ChitChatHandler."""
        from Customer_Service_Assistant.service.engine import DialogueEngine
        from Customer_Service_Assistant.service.schemas import TurnPlan, ValidationResult
        from Customer_Service_Assistant.service.validator import TurnPlanValidator

        engine = DialogueEngine()

        # Replace the chitchat handler's LLM with a mock
        mock_llm = make_mock_llm("嗨！今天过得怎么样？")
        engine._chitchat._responder._llm = mock_llm

        # Also mock the planner's LLM (step 5 _plan call)
        plan_resp = MagicMock()
        plan_resp.content = '{"direction": "chitchat", "reason": "用户在闲聊"}'
        engine._llm = MagicMock()
        engine._llm.ainvoke = AsyncMock(return_value=plan_resp)

        state = make_state([
            Message(role="user", text="你好"),
        ])

        user_msg = Message(role="user", text="你好呀")

        result = await engine.run(state, user_msg)

        assert result.role == "bot"
        assert result.text == "嗨！今天过得怎么样？"

    @pytest.mark.asyncio
    async def test_engine_still_handles_chitchat_with_empty_text(self):
        """Edge case: chitchat with empty user text should still work."""
        from Customer_Service_Assistant.service.engine import DialogueEngine

        engine = DialogueEngine()
        mock_llm = make_mock_llm("有什么可以帮你的？")
        engine._chitchat._responder._llm = mock_llm

        plan_resp = MagicMock()
        plan_resp.content = '{"direction": "chitchat", "reason": "用户发起了对话"}'
        engine._llm = MagicMock()
        engine._llm.ainvoke = AsyncMock(return_value=plan_resp)

        state = make_state([])
        user_msg = Message(role="user", text="")  # empty text

        result = await engine.run(state, user_msg)

        # Should not crash — empty text is passed through
        assert result.role == "bot"