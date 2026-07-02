"""Unit tests for DialogueService._load_state and _save_state."""

from __future__ import annotations

import json

import pytest

from Customer_Service_Assistant.service.dialogue_service import DialogueService
from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    FocusedObject,
    Message,
    ObjectData,
    Session,
    TaskContext,
    Turn,
)


# ---------------------------------------------------------------------------
# FakeSession that stores full DialogueState JSON (unlike conftest's
# FakeSession which stores a simplified ``state["messages"]`` dict).
# ---------------------------------------------------------------------------
class FakeSession:
    """In-memory fake that stores raw ``state_json`` strings keyed by sender_id."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.committed = False

    async def execute(self, statement, params=None):
        stmt = str(statement)

        if "SELECT" in stmt:
            sid = params["sid"]
            if sid in self._store:
                return _FakeResult(
                    fetchone_result=_FakeRow(state_json=self._store[sid])
                )
            return _FakeResult(fetchone_result=None)

        if "INSERT" in stmt or "UPDATE" in stmt:
            sid = params["sid"]
            self._store[sid] = params["state"]
            return _FakeResult()

        return _FakeResult()

    async def commit(self):
        self.committed = True

    async def close(self):
        pass


class _FakeRow:
    """A single row returned by fetchone(), with .state_json attribute."""

    def __init__(self, state_json: str) -> None:
        self.state_json = state_json


class _FakeResult:
    """Minimal result wrapper returned by session.execute()."""

    def __init__(self, fetchone_result=None) -> None:
        self._fetchone_result = fetchone_result

    def fetchone(self):
        return self._fetchone_result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def make_state(**overrides) -> DialogueState:
    """Build a DialogueState with sensible defaults, overridable per test."""
    data = {
        "sender_id": "u1",
        "sessions": [
            Session(
                turns=[
                    Turn(
                        input_message=Message(role="user", text="hello"),
                        assistant_messages=[
                            Message(role="bot", text="你好！有什么可以帮你的？")
                        ],
                    )
                ]
            )
        ],
        "focused_object": FocusedObject(
            type="order", id="ORDER_10001", title="订单 ORDER_10001"
        ),
    }
    data.update(overrides)
    return DialogueState(**data)


# ===================================================================
# _load_state
# ===================================================================
class TestLoadState:
    @pytest.mark.asyncio
    async def test_returns_empty_state_for_unknown_sender(self):
        """When no record exists, _load_state returns a fresh DialogueState."""
        session = FakeSession()
        svc = DialogueService(session)

        state = await svc._load_state("unknown_user")

        assert isinstance(state, DialogueState)
        assert state.sender_id == ""
        assert state.sessions == []
        assert state.current_session_id is None
        assert state.active_task is None
        assert state.focused_object is None

    @pytest.mark.asyncio
    async def test_returns_persisted_state_for_known_sender(self):
        """When a record exists, _load_state deserializes it correctly."""
        original = make_state(sender_id="u_known")
        session = FakeSession()
        session._store["u_known"] = original.to_json()
        svc = DialogueService(session)

        loaded = await svc._load_state("u_known")

        assert loaded.sender_id == "u_known"
        assert loaded.focused_object.type == "order"
        assert loaded.focused_object.id == "ORDER_10001"
        assert len(loaded.sessions) == 1
        assert len(loaded.sessions[0].turns) == 1
        assert loaded.sessions[0].turns[0].input_message.text == "hello"
        assert (
            loaded.sessions[0].turns[0].assistant_messages[0].text
            == "你好！有什么可以帮你的？"
        )

    @pytest.mark.asyncio
    async def test_load_then_save_then_load_is_idempotent(self):
        """A full save → load → save → load round-trip preserves all fields."""
        state_a = make_state(sender_id="u_roundtrip")
        session = FakeSession()
        svc = DialogueService(session)

        await svc._save_state("u_roundtrip", state_a)
        loaded_a = await svc._load_state("u_roundtrip")

        assert loaded_a.sender_id == state_a.sender_id
        assert loaded_a.focused_object == state_a.focused_object
        assert len(loaded_a.sessions) == len(state_a.sessions)
        assert loaded_a.to_json() == state_a.to_json()


# ===================================================================
# _save_state
# ===================================================================
class TestSaveState:
    @pytest.mark.asyncio
    async def test_inserts_new_record(self):
        """_save_state inserts a row when sender_id does not exist."""
        session = FakeSession()
        svc = DialogueService(session)

        state = make_state(sender_id="u_new")
        await svc._save_state("u_new", state)

        # Verify the store now contains the state
        assert "u_new" in session._store
        assert session.committed is True

        # Verify the stored JSON can be loaded back
        loaded = DialogueState.from_json(session._store["u_new"])
        assert loaded.sender_id == "u_new"
        assert loaded.focused_object.type == "order"

    @pytest.mark.asyncio
    async def test_updates_existing_record(self):
        """_save_state updates the row when sender_id already exists."""
        session = FakeSession()
        svc = DialogueService(session)

        # First save
        state_v1 = make_state(sender_id="u_update")
        await svc._save_state("u_update", state_v1)
        first_json = session._store["u_update"]

        # Modify and save again
        state_v2 = make_state(
            sender_id="u_update",
            focused_object=FocusedObject(
                type="product", id="PROD_99", title="暖火暖宝宝贴"
            ),
        )
        await svc._save_state("u_update", state_v2)
        second_json = session._store["u_update"]

        # The stored JSON should differ after update
        assert first_json != second_json
        loaded = DialogueState.from_json(second_json)
        assert loaded.focused_object.type == "product"
        assert loaded.focused_object.id == "PROD_99"

    @pytest.mark.asyncio
    async def test_save_multiple_senders_isolation(self):
        """Each sender_id has its own independent state."""
        session = FakeSession()
        svc = DialogueService(session)

        state_a = make_state(sender_id="u_a")
        state_b = make_state(sender_id="u_b", focused_object=None)

        await svc._save_state("u_a", state_a)
        await svc._save_state("u_b", state_b)

        loaded_a = DialogueState.from_json(session._store["u_a"])
        loaded_b = DialogueState.from_json(session._store["u_b"])

        assert loaded_a.focused_object is not None
        assert loaded_b.focused_object is None

    @pytest.mark.asyncio
    async def test_save_preserves_complex_state(self):
        """Saving a state with tasks, paused tasks, and system flow survives."""
        session = FakeSession()
        svc = DialogueService(session)

        state = DialogueState(
            sender_id="u_complex",
            active_task=TaskContext(flow_id="refund_flow", step_id="confirm"),
            paused_tasks=[
                TaskContext(flow_id="order_query", step_id="select_order")
            ],
            sessions=[
                Session(
                    turns=[
                        Turn(
                            input_message=Message(role="user", text="退货"),
                            assistant_messages=[
                                Message(role="bot", text="请确认退款金额")
                            ],
                        )
                    ]
                )
            ],
        )
        await svc._save_state("u_complex", state)

        loaded = await svc._load_state("u_complex")
        assert loaded.active_task.flow_id == "refund_flow"
        assert loaded.active_task.step_id == "confirm"
        assert len(loaded.paused_tasks) == 1
        assert loaded.paused_tasks[0].flow_id == "order_query"
        assert loaded.sessions[0].turns[0].input_message.text == "退货"

    @pytest.mark.asyncio
    async def test_pending_turn_is_excluded_from_json(self):
        """pending_turn is transient (exclude=True) and must not be persisted."""
        session = FakeSession()
        svc = DialogueService(session)

        state = make_state(sender_id="u_exclude")
        state.pending_turn = Turn(
            input_message=Message(role="user", text="这条消息不应被保存")
        )

        await sv
        c._save_state("u_exclude", state)
        stored_json = session._store["u_exclude"]

        assert "pending_turn" not in json.loads(stored_json)
