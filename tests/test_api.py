"""Integration tests for the chat API endpoints with mocked DB + LLM."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------
class TestPostChat:
    @pytest.mark.asyncio
    async def test_first_message_returns_bot_reply(self, async_client: AsyncClient):
        """A first-time user gets a bot reply and state is persisted."""
        resp = await async_client.post(
            "/api/chat",
            json={"sender_id": "u1", "text": "帮我查一下订单状态"},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["sender_id"] == "u1"
        assert body["message_id"].startswith("msg_")
        assert len(body["messages"]) == 1
        assert body["messages"][0]["text"] == "你好！请问有什么可以帮你的？"
        assert body["messages"][0]["object"] is None

    @pytest.mark.asyncio
    async def test_message_id_passthrough(self, async_client: AsyncClient):
        """When the client sends a message_id it is echoed back."""
        resp = await async_client.post(
            "/api/chat",
            json={
                "sender_id": "u1",
                "text": "hello",
                "message_id": "client_abc123",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["message_id"] == "client_abc123"

    @pytest.mark.asyncio
    async def test_conversation_multi_turn(self, async_client: AsyncClient):
        """Multiple turns build up conversation history."""
        # Turn 1
        r1 = await async_client.post(
            "/api/chat", json={"sender_id": "u2", "text": "第一个问题"}
        )
        assert r1.status_code == 200

        # Turn 2 — same sender
        r2 = await async_client.post(
            "/api/chat", json={"sender_id": "u2", "text": "第二个问题"}
        )
        assert r2.status_code == 200
        assert r2.json()["sender_id"] == "u2"

    @pytest.mark.asyncio
    async def test_object_message(self, async_client: AsyncClient):
        """A message with only an object (no text) is accepted."""
        resp = await async_client.post(
            "/api/chat",
            json={
                "sender_id": "u3",
                "object": {
                    "type": "order",
                    "id": "ORDER_10001",
                    "title": "订单 ORDER_10001",
                    "attributes": {"source": "order_list"},
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sender_id"] == "u3"
        assert len(body["messages"]) == 1

    @pytest.mark.asyncio
    async def test_text_and_object_together(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/api/chat",
            json={
                "sender_id": "u4",
                "text": "这个订单怎么样了",
                "object": {"type": "order", "id": "ORDER_10002"},
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_sender_id_returns_422(self, async_client: AsyncClient):
        """sender_id is required — FastAPI/pydantic returns 422."""
        resp = await async_client.post(
            "/api/chat", json={"text": "hello"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_neither_text_nor_object_returns_422(self, async_client: AsyncClient):
        """At least one of text or object must be provided."""
        resp = await async_client.post(
            "/api/chat", json={"sender_id": "u1"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_object_validated(self, async_client: AsyncClient):
        """An object with missing required fields returns 422."""
        resp = await async_client.post(
            "/api/chat",
            json={"sender_id": "u1", "object": {"type": "order"}},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_response_structure_matches_spec(self, async_client: AsyncClient):
        """Verify the response shape matches the API documentation exactly."""
        resp = await async_client.post(
            "/api/chat",
            json={
                "sender_id": "user_001",
                "text": "帮我查一下订单状态",
                "object": {
                    "type": "order",
                    "id": "ORDER_10001",
                    "title": "订单 ORDER_10001",
                    "attributes": {"source": "order_list"},
                },
                "message_id": "msg_001",
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        # Top-level fields
        assert body["sender_id"] == "user_001"
        assert body["message_id"] == "msg_001"
        assert isinstance(body["messages"], list)
        assert len(body["messages"]) == 1

        # Each message item
        msg = body["messages"][0]
        assert "text" in msg
        assert "object" in msg
        assert msg["object"] is None  # bot doesn't return object in this test


# ---------------------------------------------------------------------------
# GET /api/chat/history
# ---------------------------------------------------------------------------
class TestGetChatHistory:
    @pytest.mark.asyncio
    async def test_empty_history_for_unknown_user(self, async_client: AsyncClient):
        """A user with no prior messages gets an empty history."""
        resp = await async_client.get(
            "/api/chat/history", params={"sender_id": "unknown_user"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sender_id"] == "unknown_user"
        assert body["messages"] == []

    @pytest.mark.asyncio
    async def test_history_after_chat(self, async_client: AsyncClient):
        """After sending a message, history includes both user and bot messages."""
        # Send a chat message first
        await async_client.post(
            "/api/chat",
            json={"sender_id": "u_history", "text": "查询订单"},
        )

        # Now query history
        resp = await async_client.get(
            "/api/chat/history", params={"sender_id": "u_history"}
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["sender_id"] == "u_history"
        assert len(body["messages"]) == 2

        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["text"] == "查询订单"
        assert body["messages"][0]["object"] is None

        assert body["messages"][1]["role"] == "bot"
        assert body["messages"][1]["text"] == "你好！请问有什么可以帮你的？"

    @pytest.mark.asyncio
    async def test_history_multi_turn(self, async_client: AsyncClient):
        """Multi-turn conversation: history grows correctly."""
        sid = "u_multi"

        await async_client.post("/api/chat", json={"sender_id": sid, "text": "Q1"})
        await async_client.post("/api/chat", json={"sender_id": sid, "text": "Q2"})
        await async_client.post("/api/chat", json={"sender_id": sid, "text": "Q3"})

        resp = await async_client.get("/api/chat/history", params={"sender_id": sid})
        body = resp.json()

        # Each turn adds 2 messages (user + bot) → 6 total
        assert len(body["messages"]) == 6
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "bot", "user", "bot", "user", "bot"]

    @pytest.mark.asyncio
    async def test_history_with_object_messages(self, async_client: AsyncClient):
        """History preserves object data attached to messages."""
        sid = "u_obj_hist"

        obj_data = {
            "type": "order",
            "id": "ORDER_10001",
            "title": "订单 ORDER_10001",
        }

        await async_client.post(
            "/api/chat",
            json={"sender_id": sid, "text": "查订单", "object": obj_data},
        )

        resp = await async_client.get("/api/chat/history", params={"sender_id": sid})
        body = resp.json()

        user_msg = body["messages"][0]
        assert user_msg["role"] == "user"
        # model_dump() serialises attributes default {} into the stored object,
        # so the round-tripped object gains that key.
        assert user_msg["object"]["type"] == obj_data["type"]
        assert user_msg["object"]["id"] == obj_data["id"]
        assert user_msg["object"]["title"] == obj_data["title"]

    @pytest.mark.asyncio
    async def test_history_missing_sender_id_returns_422(self, async_client: AsyncClient):
        """sender_id query parameter is required."""
        resp = await async_client.get("/api/chat/history")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_history_response_structure_matches_spec(self, async_client: AsyncClient):
        """Verify the history response shape matches the API documentation."""
        sid = "user_001_spec"
        await async_client.post(
            "/api/chat", json={"sender_id": sid, "text": "帮我查一下订单状态"}
        )

        resp = await async_client.get("/api/chat/history", params={"sender_id": sid})
        assert resp.status_code == 200
        body = resp.json()

        assert "sender_id" in body
        assert "messages" in body
        assert isinstance(body["messages"], list)

        for msg in body["messages"]:
            assert "role" in msg
            assert msg["role"] in ("user", "bot")
            assert "text" in msg
            assert "object" in msg


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    """Test the _build_prompt helper directly."""

    def test_system_message_is_first(self):
        from Customer_Service_Assistant.api.router import _build_prompt

        history = [
            {"role": "user", "text": "hello", "object": None},
        ]
        messages = _build_prompt(history)
        assert messages[0]["role"] == "system"
        assert "电商客服" in messages[0]["content"]

    def test_user_and_bot_roles_mapped(self):
        from Customer_Service_Assistant.api.router import _build_prompt

        history = [
            {"role": "user", "text": "查询订单", "object": None},
            {"role": "bot", "text": "请提供订单号", "object": None},
        ]
        messages = _build_prompt(history)
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant"]

    def test_object_appended_to_content(self):
        from Customer_Service_Assistant.api.router import _build_prompt

        history = [
            {
                "role": "user",
                "text": None,
                "object": {"type": "order", "id": "O1", "title": "订单 O1"},
            },
        ]
        messages = _build_prompt(history)
        content = messages[1]["content"]
        assert "[order: 订单 O1]" in content

    def test_object_falls_back_to_id_when_title_missing(self):
        from Customer_Service_Assistant.api.router import _build_prompt

        history = [
            {
                "role": "user",
                "text": "查一下",
                "object": {"type": "product", "id": "PROD_99"},
            },
        ]
        messages = _build_prompt(history)
        content = messages[1]["content"]
        assert "[product: PROD_99]" in content
