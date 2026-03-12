"""FastAPI dependency-injection wiring.

Provides cached singleton factories so that service objects are created
once and reused across requests.
"""

from functools import lru_cache

from app.application.chat_service import ChatService
from app.config import get_settings
from app.domain.services import LLMService
from app.infrastructure.platform_factory import create_llm_service


@lru_cache
def _llm_service() -> LLMService:
    """Create or return the cached ``LLMService`` singleton for the configured platform."""
    return create_llm_service(get_settings())


@lru_cache
def _chat_service() -> ChatService:
    """Create or return the cached ``ChatService`` singleton."""
    return ChatService(_llm_service())


def get_chat_service() -> ChatService:
    """FastAPI ``Depends`` callable that returns the ``ChatService``.

    Returns:
        The application-level ``ChatService`` instance.
    """
    return _chat_service()


def reset_services() -> None:
    """Clear cached singletons so they are rebuilt on next access.

    Intended for use in tests to swap out dependencies between runs.
    """
    _llm_service.cache_clear()
    _chat_service.cache_clear()
