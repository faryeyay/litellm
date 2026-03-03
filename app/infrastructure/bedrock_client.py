import os

import litellm

from app.config import Settings
from app.domain.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    Usage,
)
from app.domain.services import LLMService


class BedrockLLMService(LLMService):
    """LLMService adapter that routes requests to AWS Bedrock via litellm.

    Args:
        settings: Application settings carrying AWS credentials and the
            default model identifier.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._configure_env()

    def _configure_env(self) -> None:
        """Push AWS credentials into the process environment for boto3."""
        if self._settings.aws_profile:
            os.environ["AWS_PROFILE"] = self._settings.aws_profile
        if self._settings.aws_region:
            os.environ["AWS_REGION_NAME"] = self._settings.aws_region

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Forward a chat request to Bedrock and return a domain response.

        Args:
            request: The incoming chat completion request.

        Returns:
            A ``ChatResponse`` mapped from the litellm response.
        """
        model = request.model or self._settings.default_model

        response = await litellm.acompletion(
            model=model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )

        return self._to_domain(response, model)

    @staticmethod
    def _to_domain(response: litellm.ModelResponse, model: str) -> ChatResponse:
        """Convert a raw litellm ``ModelResponse`` into a domain ``ChatResponse``.

        Args:
            response: The litellm response object.
            model: The model identifier that was used.

        Returns:
            A ``ChatResponse`` with choices and optional usage stats.
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
