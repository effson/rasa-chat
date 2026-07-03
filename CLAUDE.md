# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An e-commerce customer service AI assistant powered by LangChain + FastAPI, using Qwen (Alibaba DashScope, OpenAI-compatible) as the LLM backend. The system integrates with a MySQL-backed commerce database to handle customer inquiries about orders, products, logistics, refunds, and shipping.

## Commands

```bash
# Install dependencies
uv sync

# Run the FastAPI app
uvicorn Customer_Service_Assistant.main:app --host 127.0.0.1 --port 18000 --reload

# Start / stop MySQL (Docker)
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml down

# All tests
pytest tests/ -v

# Unit tests only (in-memory fakes, no MySQL needed)
pytest tests/test_dialogue_service.py tests/test_api.py tests/test_schemas.py -v

# Integration tests only (requires real MySQL)
pytest tests/test_dialogue_service_integration.py -v

# Single test
pytest tests/test_dialogue_service.py::TestLoadState::test_returns_empty_state_for_unknown_sender -v
```

## Architecture

### Layer stack (top → bottom)

```
api/           — HTTP layer: FastAPI router, request/response schemas, DI wiring
  ├─ router.py        POST /api/chat, GET /api/chat/history
  ├─ dependencies.py  get_dialogue_service (AsyncSession → DialogueService)
  └─ schemas.py       HTTP contract models (ChatRequest, ChatResponse, HistoryResponse)
                        Every model has to_service()/from_service() converters.

service/       — Domain layer: orchestration, engine dispatch, domain schemas
  ├─ dialogue_service.py  Load state → create Turn → Engine.run() → save state
  ├─ engine.py            DialogueEngine: _plan (prompt build) → _generate (LLM call)
  └─ schemas.py           Domain models: DialogueState, Session, Turn, Message,
                          TaskContext, SystemContext (7 discriminated sub-types)

infrastructure/ — External I/O: DB, LLM, commerce API client
  ├─ db.py               Async SQLAlchemy engine + session factory (aiomysql)
  ├─ llm.py              ChatOpenAI singleton pointed at DashScope
  └─ api.py              httpx.AsyncClient factory for the commerce backend
```

### Request flow (POST /api/chat)

For task processing design, please refer to`TaskHandler设计.md` `TaskHandler具体设计.md`.

For knowledge ＆ chitchat processing design, please refer to`KnowledgeHandler&ChitChatHandler设计.md` .

For Dialogue's Engine design, please refer to `DialogueEngine设计.md`



### Schema duality

The project maintains parallel schema layers — `api/schemas.py` and `service/schemas.py` — with explicit conversion methods (`to_service()` / `from_service()`). When adding fields, add them to the **service** schema first, then mirror in the **API** schema with converters.

### DialogueState persistence

Conversation state is stored as a JSON blob in `dialogue_states` (`sender_id` PRIMARY KEY → `state_json TEXT`). Serialization uses `DialogueState.to_json()` / `from_json()` via Pydantic `model_dump_json` / `model_validate`. The `pending_turn` field is transient (`exclude=True`) — it is never persisted.

### Key design decisions

- **Dual-database**: `customer_service` (chat state) and `commerce` (business data) are separate databases on the same MySQL instance.
- **Chinese-language domain**: All seed data and system prompts are in Chinese. The e-commerce domain (users, products, orders) uses Chinese naming.
- **Upsert pattern**: `_save_state` uses MySQL `ON DUPLICATE KEY UPDATE` — insert on first message, update on subsequent messages from the same sender.
- **Dependency injection**: FastAPI `Depends(get_dialogue_service)` wires `AsyncSession → DialogueService → DialogueEngine`.
- **Flow definitions**: `flow_config/` contains YAML definitions for user flows and system flows. These are currently configuration artifacts (not consumed by the runtime engine) but define the intended flow orchestration model.
- **Embedded smoke tests**: Several infrastructure modules (`db.py`, `llm.py`, `api.py`, `settings.py`) have `if __name__ == "__main__"` blocks that run quick self-checks. These are developer convenience, not part of the pytest suite.

## Testing

Two layers of tests:

| Layer | File | Database | Purpose |
| --- | --- | --- | --- |
| Unit | `test_dialogue_service.py` | `FakeSession` (in-memory dict) | Fast feedback, logic correctness |
| Integration | `test_dialogue_service_integration.py` | Real MySQL via `NullPool` | SQL correctness, serialization round-trips |

**Test fixtures** (conftest.py):
- `fake_session` — in-memory FakeSession for API tests
- `async_client` — FastAPI test client with mocked DB + LLM
- Integration test fixtures (`db_session`, `svc`) are defined locally in the integration test file

**Windows note**: `conftest.py` sets `WindowsSelectorEventLoopPolicy` — required because the default `ProactorEventLoop` is incompatible with aiomysql. Integration tests use `NullPool` (fresh connection per test) to avoid pool dead-connection issues.

## Environment Configuration

All config in `.env` (not committed):
- `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` — LLM (DashScope, OpenAI-compatible)
- `DATABASE_URL` — `mysql+aiomysql://user:pass@host:3306/customer_service?charset=utf8mb4`
- `COMMERCE_API_BASE_URL` — External commerce API (default `http://127.0.0.1:18000`)
- `APP_HOST`, `APP_PORT` — FastAPI bind address

Settings are loaded via `pydantic-settings` in `Customer_Service_Assistant/config/settings.py`.