"""Shared test fixtures — mock DB session, mock LLM, and async test client."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Windows ProactorEventLoop is incompatible with aiomysql (async MySQL driver):
# when the remote server drops an idle connection, the proactor transport
# crashes with ``AttributeError: 'NoneType' object has no attribute 'send'``.
# Switch to SelectorEventLoop which handles TCP socket lifecycle correctly.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from Customer_Service_Assistant.api.router import router
from Customer_Service_Assistant.infrastructure.db import get_async_session


# ---------------------------------------------------------------------------
# Fake row returned by session.execute(...).fetchone()
# ---------------------------------------------------------------------------
@dataclass
class FakeRow:
    state_json: str


# ---------------------------------------------------------------------------
# In-memory fake for AsyncSession that stores dialogue states in a dict.
# ---------------------------------------------------------------------------
class FakeSession:
    """An in-memory fake that mimics the subset of AsyncSession used by the API.

    Dialogue state is keyed by sender_id, just like the real ``dialogue_states``
    table.
    """

    def __init__(self, initial: Optional[dict[str, list[dict]]] = None) -> None:
        self._store: dict[str, list[dict]] = initial or {}
        self.committed = False

    async def execute(self, statement, params=None):
        """Handle SELECT / INSERT … ON DUPLICATE KEY UPDATE."""
        stmt = str(statement)

        if "SELECT" in stmt:
            sid = params["sid"]
            if sid in self._store:
                state_json = json.dumps(
                    {"messages": self._store[sid]}, ensure_ascii=False
                )
                return _FakeResult(fetchone_result=FakeRow(state_json=state_json))
            return _FakeResult(fetchone_result=None)

        if "INSERT" in stmt:
            sid = params["sid"]
            state = json.loads(params["state"])
            self._store[sid] = state["messages"]
            return _FakeResult()

        return _FakeResult()

    async def commit(self):
        self.committed = True

    async def close(self):
        pass


class _FakeResult:
    """Minimal fake for the result returned by session.execute()."""

    def __init__(self, fetchone_result=None) -> None:
        self._fetchone_result = fetchone_result

    def fetchone(self):
        return self._fetchone_result

    def scalars(self):
        return _FakeScalars(self._fetchone_result)


class _FakeScalars:
    def __init__(self, row) -> None:
        self._row = row

    def all(self):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_session() -> FakeSession:
    """Return a fresh in-memory FakeSession."""
    return FakeSession()


@pytest.fixture
def fake_llm_response():
    """Return a MagicMock that mimics an LLM response with a .content attribute."""
    resp = MagicMock()
    resp.content = "你好！请问有什么可以帮你的？"
    return resp


@pytest_asyncio.fixture
async def async_client(
    fake_session: FakeSession,
    fake_llm_response: MagicMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Build a FastAPI test app with all real dependencies swapped for fakes."""
    app = FastAPI()
    app.include_router(router)

    # Override the DB session dependency — must be a plain async callable that
    # returns the session directly, NOT an async generator.
    async def _get_session():
        return fake_session

    app.dependency_overrides[get_async_session] = _get_session

    # Patch the LLM at the point it's imported in the router module
    with patch(
        "Customer_Service_Assistant.api.router.llm",
        MagicMock(ainvoke=AsyncMock(return_value=fake_llm_response)),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
