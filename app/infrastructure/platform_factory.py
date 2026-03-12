"""Factory that instantiates the correct LLM service adapter for a given platform."""

from app.config import LLMPlatform, Settings
from app.domain.services import LLMService
from app.infrastructure.bedrock_client import BedrockLLMService
from app.infrastructure.claude_client import ClaudeLLMService
from app.infrastructure.local_client import LocalLLMService


def create_llm_service(settings: Settings) -> LLMService:
    """Return the ``LLMService`` implementation matching ``settings.llm_platform``.

    Args:
        settings: Application settings used to select and configure the adapter.

    Returns:
        A concrete ``LLMService`` ready to serve completion requests.

    Raises:
        ValueError: If ``settings.llm_platform`` is not a recognised platform.
    """
    if settings.llm_platform == LLMPlatform.CLAUDE:
        return ClaudeLLMService(settings)
    if settings.llm_platform == LLMPlatform.LOCAL:
        return LocalLLMService(settings)
    if settings.llm_platform == LLMPlatform.BEDROCK:
        return BedrockLLMService(settings)
    raise ValueError(f"Unsupported LLM platform: {settings.llm_platform!r}")
