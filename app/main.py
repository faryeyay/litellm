"""FastAPI application factory and ASGI entry-point."""

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A fully wired ``FastAPI`` instance ready to serve requests.
    """
    application = FastAPI(title="LiteLLM Multi-Platform Proxy")
    application.include_router(chat_router)

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return application


app = create_app()
