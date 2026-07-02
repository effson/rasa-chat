"""Commerce API infrastructure — async HTTP client for the external commerce backend."""

from typing import Optional

import httpx

from Customer_Service_Assistant.config.settings import settings


def get_commerce_api_client(
    *,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    **kwargs,
) -> httpx.AsyncClient:
    """Return an httpx.AsyncClient pre-configured for the commerce API.

    Parameters
    ----------
    base_url:
        Override the default commerce API base URL from settings.
    timeout:
        Request timeout in seconds. Defaults to 30.
    **kwargs:
        Forwarded to the httpx.AsyncClient constructor.
    """
    return httpx.AsyncClient(
        base_url=base_url or settings.commerce_api_base_url,
        timeout=timeout,
        **kwargs,
    )


if __name__ == "__main__":
    import asyncio

    async def _test():
        # 1. Default client uses settings
        client = get_commerce_api_client()
        assert str(client.base_url) == settings.commerce_api_base_url
        assert client.timeout == httpx.Timeout(30.0)

        # 2. Override base_url
        client2 = get_commerce_api_client(base_url="http://example.com:9999")
        assert str(client2.base_url) == "http://example.com:9999"

        # 3. Override timeout
        client3 = get_commerce_api_client(timeout=10.0)
        assert client3.timeout == httpx.Timeout(10.0)

        # 4. Each call returns a distinct client
        assert client is not client2

        # 5. Clean up
        await client.aclose()
        await client2.aclose()
        await client3.aclose()

        print("All api tests passed.")

    asyncio.run(_test())
