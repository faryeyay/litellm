"""Anthropic Claude direct API adapter via litellm."""

import os

import litellm

from app.config import Settings
from app.domain.models import ChatRequest, ChatResponse
from app.infrastructure.base import LiteLLMBaseService


class ClaudeLLMService(LiteLLMBaseService):
    """LLMService adapter that routes requests to Anthropic Claude via litellm.

    Uses the Anthropic API directly rather than going through AWS Bedrock.

    Args:
        settings: Application settings carrying the Anthropic API key and
            the default Claude model identifier.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Forward a chat request to Anthropic Claude and return a domain response.

        Args:
            request: The incoming chat completion request.

        Returns:
            A ``ChatResponse`` mapped from the litellm response.
        """
        model = request.model or self._settings.claude_default_model

        response = await litellm.acompletion(
            model=model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )

        return self._to_domain(response, model)
