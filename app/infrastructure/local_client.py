"""Local model (Ollama) adapter via litellm."""

import litellm

from app.config import Settings
from app.domain.models import ChatRequest, ChatResponse
from app.infrastructure.base import LiteLLMBaseService


class LocalLLMService(LiteLLMBaseService):
    """LLMService adapter that routes requests to a local Ollama server via litellm.

    Suitable for development and testing with small models such as
    ``qwen2.5:0.5b`` or ``gemma:2b`` that can be embedded in the Docker image.

    Args:
        settings: Application settings carrying the Ollama base URL and
            the default local model identifier.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Forward a chat request to the local Ollama server and return a domain response.

        Args:
            request: The incoming chat completion request.

        Returns:
            A ``ChatResponse`` mapped from the litellm response.
        """
        model = request.model or self._settings.local_default_model

        response = await litellm.acompletion(
            model=model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_base=self._settings.ollama_base_url,
            stream=False,
        )

        return self._to_domain(response, model)
