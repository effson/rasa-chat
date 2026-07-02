"""FastAPI dependency injection — wires up dependencies for endpoint handlers."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from Customer_Service_Assistant.infrastructure.db import get_async_session
from Customer_Service_Assistant.service.dialogue_service import DialogueService

__all__ = ["DialogueService", "get_dialogue_service"]


async def get_dialogue_service(
    session: AsyncSession = Depends(get_async_session),
) -> DialogueService:
    """FastAPI dependency that yields a DialogueService backed by a DB session."""
    return DialogueService(session)
