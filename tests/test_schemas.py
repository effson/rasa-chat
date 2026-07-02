"""Unit tests for the Pydantic schemas in Customer_Service_Assistant.api.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from Customer_Service_Assistant.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
    ObjectData,
)


# ---------------------------------------------------------------------------
# ObjectData
# ---------------------------------------------------------------------------
class TestObjectData:
    def test_minimal_fields(self):
        """ObjectData requires only type and id."""
        obj = ObjectData(type="order", id="ORDER_10001")
        assert obj.type == "order"
        assert obj.id == "ORDER_10001"
        assert obj.title is None
        assert obj.attributes == {}

    def test_all_fields(self):
        obj = ObjectData(
            type="product",
            id="PROD_1",
            title="iPhone 15",
            attributes={"source": "product_list"},
        )
        assert obj.title == "iPhone 15"
        assert obj.attributes == {"source": "product_list"}

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError) as exc:
            ObjectData(id="X")
        assert "type" in str(exc.value)

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError) as exc:
            ObjectData(type="order")
        assert "id" in str(exc.value)

    def test_attributes_defaults_independent_per_instance(self):
        """attributes uses default_factory so each instance gets its own dict."""
        a = ObjectData(type="t", id="1")
        b = ObjectData(type="t", id="2")
        a.attributes["key"] = "val"
        assert b.attributes == {}


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------
class TestChatRequest:
    def test_text_only(self):
        r = ChatRequest(sender_id="u1", text="hello")
        assert r.sender_id == "u1"
        assert r.text == "hello"
        assert r.object is None
        assert r.message_id is None

    def test_object_only(self):
        obj = ObjectData(type="order", id="O1", title="订单 O1")
        r = ChatRequest(sender_id="u2", object=obj)
        assert r.text is None
        assert r.object is not None
        assert r.object.type == "order"

    def test_text_and_object(self):
        obj = ObjectData(type="product", id="P1")
        r = ChatRequest(sender_id="u3", text="这个商品怎么样", object=obj)
        assert r.text is not None
        assert r.object is not None

    def test_message_id_passthrough(self):
        r = ChatRequest(sender_id="u1", text="hi", message_id="client_msg_42")
        assert r.message_id == "client_msg_42"

    def test_message_id_defaults_to_none(self):
        r = ChatRequest(sender_id="u1", text="hi")
        assert r.message_id is None

    def test_neither_text_nor_object_raises(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(sender_id="u1")
        assert "At least one of `text` or `object`" in str(exc.value)

    def test_sender_id_required(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(text="hi")  # type: ignore[call-arg]
        assert "sender_id" in str(exc.value)


# ---------------------------------------------------------------------------
# ChatMessage & ChatResponse
# ---------------------------------------------------------------------------
class TestChatMessage:
    def test_text_only(self):
        msg = ChatMessage(text="你好")
        assert msg.text == "你好"
        assert msg.object is None

    def test_with_object(self):
        obj = ObjectData(type="order", id="O1")
        msg = ChatMessage(object=obj)
        assert msg.text is None
        assert msg.object == obj

    def test_both_none_is_valid(self):
        """An empty message (both text and object None) is allowed."""
        msg = ChatMessage()
        assert msg.text is None
        assert msg.object is None


class TestChatResponse:
    def test_roundtrip(self):
        resp = ChatResponse(
            sender_id="u1",
            message_id="msg_001",
            messages=[ChatMessage(text="你好！")],
        )
        data = resp.model_dump()
        assert data == {
            "sender_id": "u1",
            "message_id": "msg_001",
            "messages": [{"text": "你好！", "object": None}],
        }

    def test_multiple_messages(self):
        resp = ChatResponse(
            sender_id="u1",
            message_id="msg_001",
            messages=[
                ChatMessage(text="第一句"),
                ChatMessage(text="第二句"),
            ],
        )
        assert len(resp.messages) == 2


# ---------------------------------------------------------------------------
# HistoryMessage & HistoryResponse
# ---------------------------------------------------------------------------
class TestHistoryMessage:
    def test_user_message(self):
        msg = HistoryMessage(role="user", text="查询订单")
        assert msg.role == "user"
        assert msg.text == "查询订单"

    def test_bot_message(self):
        msg = HistoryMessage(role="bot", text="请提供订单号")
        assert msg.role == "bot"

    def test_role_required(self):
        with pytest.raises(ValidationError):
            HistoryMessage(text="hi")  # type: ignore[call-arg]

    def test_with_object(self):
        obj = ObjectData(type="order", id="O1")
        msg = HistoryMessage(role="user", object=obj)
        assert msg.object is not None
        assert msg.object.type == "order"


class TestHistoryResponse:
    def test_empty_history(self):
        resp = HistoryResponse(sender_id="u1", messages=[])
        assert resp.sender_id == "u1"
        assert resp.messages == []

    def test_roundtrip(self):
        resp = HistoryResponse(
            sender_id="u1",
            messages=[
                HistoryMessage(role="user", text="查询订单"),
                HistoryMessage(role="bot", text="请提供订单号"),
            ],
        )
        data = resp.model_dump()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "bot"
