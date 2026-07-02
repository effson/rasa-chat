# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An e-commerce customer service AI assistant powered by LangChain + FastAPI, using Qwen (Alibaba DashScope) as the LLM backend. The system integrates with a MySQL-backed commerce database to handle customer inquiries about orders, products, logistics, refunds, and shipping.

## Commands

```bash
# Start MySQL database
docker compose -f docker/docker-compose.yml up -d

# Stop MySQL database
docker compose -f docker/docker-compose.yml down

# Run the FastAPI app
uvicorn Customer_Service_Assistant.main:app --host 127.0.0.1 --port 18000 --reload

# Install dependencies (after cloning)
pip install -e .
```

## Architecture

### Package Structure

- **`Customer_Service_Assistant/`** — Main application package. The `__init__.py` is currently a scaffold; the FastAPI app lives in `main.py` (to be created).

### Databases

Two MySQL databases managed via Docker:

1. **`customer_service`** — Persists conversation state in `dialogue_states` table (`sender_id` → `state_json` TEXT). This is the chatbot's conversation memory, allowing long-running dialogues to survive restarts.

2. **`commerce`** — Business domain data with seed data for testing. Tables: `users`, `products`, `orders`, `order_items`, `logistics_records`, `logistics_traces`, `refund_requests`, `shipping_urge_requests`.

Database access is async via SQLAlchemy + aiomysql (`DATABASE_URL` uses `mysql+aiomysql://` scheme).

### LLM Integration

- **Model**: `qwen-plus` via Alibaba DashScope (`LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`)
- **Framework**: LangChain with `langchain-openai` extension (DashScope is OpenAI-compatible)
- The chat model is OpenAI-compatible, so use `ChatOpenAI` from `langchain-openai` with the base URL pointed at DashScope.

### External Commerce API

The app calls out to a separate commerce backend at `COMMERCE_API_BASE_URL` (default `http://127.0.0.1:18000` — same host, suggesting the commerce API may eventually be external or a separate service).

### Key Design Decisions

- **Conversation state persistence**: Dialogue state is stored as JSON blobs in MySQL keyed by `sender_id`. This means the chatbot must serialize/deserialize state on each turn rather than holding it in memory.
- **Dual-database pattern**: The chatbot's operational data (`customer_service`) is separate from the business domain data (`commerce`), keeping concerns cleanly separated.
- **Seed data is Chinese-language**: The commerce seed data (users, products, orders) is entirely in Chinese, indicating this is a Chinese-market e-commerce assistant.

## Environment Configuration

All config lives in `.env` (not committed — add to `.gitignore`):
- `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` — LLM connection
- `DATABASE_URL` — MySQL connection string
- `COMMERCE_API_BASE_URL` — External commerce API
- `APP_HOST`, `APP_PORT` — FastAPI bind address