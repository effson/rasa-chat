"""FastAPI application entry point for the customer service assistant."""

from fastapi import FastAPI

from Customer_Service_Assistant.api.router import router
from Customer_Service_Assistant.config.settings import settings

app = FastAPI(title="Customer Service Assistant")

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Customer_Service_Assistant.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
