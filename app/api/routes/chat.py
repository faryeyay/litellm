"""Chat completion HTTP routes (OpenAI-compatible ``/v1`` prefix)."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.application.chat_service import ChatService
from app.domain.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Create a chat completion.

    Accepts an OpenAI-style request body and proxies it to the
    configured LLM platform via litellm.

    Args:
        request: Validated chat completion request.
        chat_service: Injected application service.

    Returns:
        The model's ``ChatResponse``.
    """
    return await chat_service.chat(request)
