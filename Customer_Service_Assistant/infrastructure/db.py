"""Database infrastructure — async SQLAlchemy engine and session for MySQL."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from Customer_Service_Assistant.config.settings import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async SQLAlchemy session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


if __name__ == "__main__":
    import asyncio

    from sqlalchemy import make_url, text
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    async def _test():
        # === 1. Engine config ===
        assert isinstance(engine, AsyncEngine)
        expected = make_url(settings.database_url)
        actual = make_url(str(engine.url))
        assert actual.drivername == expected.drivername
        assert actual.host == expected.host
        assert actual.port == expected.port
        assert actual.database == expected.database

        # === 2. Session factory ===
        session = async_session_factory()
        assert isinstance(session, AsyncSession)
        await session.close()

        # === 3. Real MySQL queries ===
        async with async_session_factory() as s:

            # 3a. customer_service: dialogue_states table exists
            result = await s.execute(
                text("SELECT COUNT(*) FROM dialogue_states")
            )
            count = result.scalar()
            assert isinstance(count, int)
            print(f"  dialogue_states rows: {count}")

            # 3b. commerce: cross-database query — users
            result = await s.execute(
                text("SELECT user_id, nickname, level FROM commerce.users ORDER BY id")
            )
            users = result.fetchall()
            assert len(users) == 3
            assert users[0].nickname == "小李"
            assert users[1].level == "普通会员"
            print(f"  users: {[(u.nickname, u.level) for u in users]}")

            # 3c. commerce: orders with join — count by status
            result = await s.execute(
                text(
                    "SELECT status, COUNT(*) AS cnt "
                    "FROM commerce.orders GROUP BY status ORDER BY status"
                )
            )
            status_counts = result.fetchall()
            assert len(status_counts) >= 4  # 待发货/运输中/已完成/待揽收/已取消
            print(f"  order status counts: {[(r.status, r.cnt) for r in status_counts]}")

            # 3d. commerce: products — check seeded products
            result = await s.execute(
                text("SELECT product_id, title, price FROM commerce.products ORDER BY price")
            )
            products = result.fetchall()
            assert len(products) == 6
            assert products[0].title == "暖火暖宝宝贴 20片装"  # cheapest
            assert products[-1].title == "iPhone 15 Pro 256G 远峰蓝"  # most expensive
            print(f"  products: {[(p.title, float(p.price)) for p in products]}")

            # 3e. commerce: logistics — tracking numbers
            result = await s.execute(
                text(
                    "SELECT tracking_number, logistics_company, status "
                    "FROM commerce.logistics_records ORDER BY id"
                )
            )
            logistics = result.fetchall()
            assert len(logistics) == 3
            assert logistics[0].tracking_number == "JD000123456789"
            print(f"  logistics: {[(r.tracking_number, r.logistics_company) for r in logistics]}")

            # 3f. commerce: refund requests
            result = await s.execute(
                text("SELECT refund_id, status FROM commerce.refund_requests ORDER BY id")
            )
            refunds = result.fetchall()
            assert len(refunds) == 2
            print(f"  refunds: {[(r.refund_id, r.status) for r in refunds]}")

        # === 4. get_async_session generator ===
        agen = get_async_session()
        session2 = await agen.__anext__()
        assert isinstance(session2, AsyncSession)
        try:
            await agen.__anext__()
        except StopAsyncIteration:
            pass

        print("\nAll db tests passed.")

        # Clean up connections before the event loop closes (avoids aiomysql
        # "Event loop is closed" noise on Windows).
        await engine.dispose()

    asyncio.run(_test())
