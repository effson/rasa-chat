"""Integration tests for DialogueService._load_state / _save_state against real MySQL.

These tests use a **NullPool** engine (no connection pooling) so each test
gets a fresh MySQL connection.  This avoids the Windows ProactorEventLoop /
aiomysql dead-connection issue that occurs when the remote MySQL server drops
idle connections from the pool.

Run with:  pytest tests/test_dialogue_service_integration.py -v
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from Customer_Service_Assistant.config.settings import settings
from Customer_Service_Assistant.service.dialogue_service import DialogueService
from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    FocusedObject,
    Message,
    Session,
    TaskContext,
    Turn,
)


# ---------------------------------------------------------------------------
# Test engine — NullPool means every session gets a brand-new connection.
# ---------------------------------------------------------------------------
_test_engine = create_async_engine(
    settings.database_url,
    echo=False,
    poolclass=NullPool,
)

_test_session_factory = async_sessionmaker(
    _test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unique_sid(prefix: str = "itest") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def make_state(sender_id: str, **overrides) -> DialogueState:
    """Build a DialogueState for integration testing."""
    data: dict = {
        "sender_id": sender_id,
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
    }
    data.update(overrides)
    return DialogueState(**data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def db_session():
    """Yield a fresh AsyncSession (new connection every time via NullPool).

    Cleanup deletes test rows and closes the session — all on the same
    connection, so there is no pool to return a dead connection to.
    """
    async with _test_session_factory() as session:
        yield session
        try:
            await session.execute(
                text("DELETE FROM dialogue_states WHERE sender_id LIKE 'itest_%'")
            )
            await session.commit()
        except Exception:
            pass  # connection may already be dead; next test gets a fresh one
        finally:
            await session.close()


@pytest_asyncio.fixture
async def svc(db_session: AsyncSession) -> DialogueService:
    """Return a DialogueService wired to a real MySQL session."""
    return DialogueService(db_session)


# ---------------------------------------------------------------------------
# _load_state
# ---------------------------------------------------------------------------
class TestLoadStateIntegration:
    @pytest.mark.asyncio
    async def test_returns_empty_state_for_unknown_sender(self, svc: DialogueService):
        sid = _unique_sid("unknown")
        state = await svc._load_state(sid)

        assert isinstance(state, DialogueState)
        assert state.sender_id == ""
        assert state.sessions == []
        assert state.active_task is None

    @pytest.mark.asyncio
    async def test_loads_persisted_state(self, svc: DialogueService):
        sid = _unique_sid("load")
        original = make_state(sid)
        await svc._save_state(sid, original)

        loaded = await svc._load_state(sid)

        assert loaded.sender_id == sid
        assert len(loaded.sessions) == 1
        assert loaded.sessions[0].turns[0].input_message.text == "hello"

    @pytest.mark.asyncio
    async def test_load_with_different_session(self, svc: DialogueService):
        sid = _unique_sid("newsess")
        original = make_state(sid)
        await svc._save_state(sid, original)

        async with _test_session_factory() as new_session:
            svc2 = DialogueService(new_session)
            loaded = await svc2._load_state(sid)
            await new_session.close()

        assert loaded.sender_id == sid


# ---------------------------------------------------------------------------
# _save_state
# ---------------------------------------------------------------------------
class TestSaveStateIntegration:
    @pytest.mark.asyncio
    async def test_inserts_new_row(self, svc: DialogueService):
        sid = _unique_sid("insert")
        state = make_state(sid)
        await svc._save_state(sid, state)

        loaded = await svc._load_state(sid)
        assert loaded.sender_id == sid

    @pytest.mark.asyncio
    async def test_updates_existing_row(self, svc: DialogueService):
        sid = _unique_sid("update")
        v1 = make_state(sid, focused_object=FocusedObject(type="order", id="O1", title="订单O1"))
        await svc._save_state(sid, v1)

        v2 = make_state(sid, focused_object=FocusedObject(type="product", id="P99", title="产品P99"))
        await svc._save_state(sid, v2)

        loaded = await svc._load_state(sid)
        assert loaded.focused_object.type == "product"
        assert loaded.focused_object.id == "P99"

    @pytest.mark.asyncio
    async def test_only_one_row_per_sender(self, svc: DialogueService):
        sid = _unique_sid("norows")
        for _ in range(3):
            await svc._save_state(sid, make_state(sid))

        result = await svc._session.execute(
            text("SELECT COUNT(*) AS cnt FROM dialogue_states WHERE sender_id = :sid"),
            {"sid": sid},
        )
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_preserves_complex_state(self, svc: DialogueService):
        sid = _unique_sid("complex")
        state = DialogueState(
            sender_id=sid,
            active_task=TaskContext(flow_id="refund_flow", step_id="confirm_amount"),
            paused_tasks=[TaskContext(flow_id="order_query", step_id="select_order")],
            focused_object=FocusedObject(type="order", id="ORDER_5001", title="订单5001"),
            sessions=[
                Session(
                    turns=[
                        Turn(
                            input_message=Message(role="user", text="我要退货"),
                            assistant_messages=[
                                Message(role="bot", text="请确认退款金额：¥299.00")
                            ],
                        )
                    ]
                )
            ],
        )
        await svc._save_state(sid, state)

        loaded = await svc._load_state(sid)
        assert loaded.active_task.flow_id == "refund_flow"
        assert len(loaded.paused_tasks) == 1
        assert loaded.focused_object.id == "ORDER_5001"
        assert loaded.sessions[0].turns[0].input_message.text == "我要退货"