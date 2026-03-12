import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import LLMPlatform, Settings
from app.domain.models import ChatRequest, Message
from app.infrastructure.bedrock_client import BedrockLLMService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_platform=LLMPlatform.BEDROCK,
        aws_region="us-east-1",
        bedrock_default_model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    )


@pytest.fixture
def service(settings: Settings) -> BedrockLLMService:
    return BedrockLLMService(settings)


@pytest.fixture
def chat_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_litellm_response(model: str = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0") -> MagicMock:
    response = MagicMock()
    response.id = "resp-bedrock-123"
    response.choices = [
        MagicMock(
            index=0,
            message=MagicMock(role="assistant", content="Hello from Bedrock!"),
            finish_reason="stop",
        )
    ]
    response.usage = MagicMock(prompt_tokens=5, completion_tokens=8, total_tokens=13)
    return response


class TestBedrockLLMService:
    @pytest.mark.asyncio
    async def test_complete_uses_default_model_when_request_has_none(
        self, service: BedrockLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await service.complete(chat_request)
        assert result.model == "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

    @pytest.mark.asyncio
    async def test_complete_uses_model_override_from_request(
        self, service: BedrockLLMService
    ):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="bedrock/amazon.titan-text-lite-v1",
        )
        mock_response = _mock_litellm_response(model="bedrock/amazon.titan-text-lite-v1")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["model"] == "bedrock/amazon.titan-text-lite-v1"

    @pytest.mark.asyncio
    async def test_complete_returns_mapped_domain_response(
        self, service: BedrockLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await service.complete(chat_request)
        assert result.id == "resp-bedrock-123"
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hello from Bedrock!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.total_tokens == 13

    @pytest.mark.asyncio
    async def test_complete_passes_temperature_and_max_tokens(
        self, service: BedrockLLMService
    ):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            temperature=0.3,
            max_tokens=512,
        )
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["temperature"] == 0.3
        assert mock_call.call_args.kwargs["max_tokens"] == 512

    def test_configure_env_sets_aws_region(self, monkeypatch, settings: Settings):
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        BedrockLLMService(settings)
        assert os.environ.get("AWS_REGION_NAME") == "us-east-1"

    def test_configure_env_sets_aws_profile_when_provided(self, monkeypatch):
        settings = Settings(aws_profile="my-profile", llm_platform=LLMPlatform.BEDROCK)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        BedrockLLMService(settings)
        assert os.environ.get("AWS_PROFILE") == "my-profile"
