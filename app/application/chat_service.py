from app.domain.models import ChatRequest, ChatResponse
from app.domain.services import LLMService


class ChatService:
    """Application service that orchestrates chat completion use-cases.

    Args:
        llm: An ``LLMService`` implementation used to generate completions.
    """

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a chat completion.

        Args:
            request: The validated chat request from the API layer.

        Returns:
            The model's ``ChatResponse``.
        """
        return await self._llm.complete(request)
