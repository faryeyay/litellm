"""Shared base class for litellm-backed LLM service adapters."""

import litellm

from app.domain.models import ChatResponse, Choice, Message, Usage
from app.domain.services import LLMService


class LiteLLMBaseService(LLMService):
    """Base adapter that provides shared litellm response mapping.

    Concrete platform adapters extend this class and implement
    ``complete()``, calling ``_to_domain()`` to convert the raw
    litellm response into domain types.
    """

    @staticmethod
    def _to_domain(response: litellm.ModelResponse, model: str) -> ChatResponse:
        """Map a litellm ``ModelResponse`` to a domain ``ChatResponse``.

        Args:
            response: The raw litellm response object.
            model: The model identifier that served the request.

        Returns:
            A ``ChatResponse`` populated with choices and optional usage.
        """
        choices = [
            Choice(
                index=c.index,
                message=Message(
                    role=c.message.role,
                    content=c.message.content or "",
                ),
                finish_reason=c.finish_reason,
            )
            for c in response.choices
        ]

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatResponse(
            id=response.id,
            model=model,
            choices=choices,
            usage=usage,
        )
