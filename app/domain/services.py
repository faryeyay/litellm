from abc import ABC, abstractmethod

from app.domain.models import ChatRequest, ChatResponse


class LLMService(ABC):
    """Port defining the contract for any LLM backend."""

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request to the underlying model.

        Args:
            request: The chat request containing messages and parameters.

        Returns:
            A ``ChatResponse`` with the model's generated output.
        """
        ...
