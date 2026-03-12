import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import LLMPlatform, Settings
from app.domain.models import ChatRequest, Message
from app.infrastructure.claude_client import ClaudeLLMService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_platform=LLMPlatform.CLAUDE,
        anthropic_api_key="test-api-key",
        claude_default_model="claude-3-5-sonnet-20241022",
    )


@pytest.fixture
def service(settings: Settings) -> ClaudeLLMService:
    return ClaudeLLMService(settings)


@pytest.fixture
def chat_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello Claude")])


def _mock_litellm_response() -> MagicMock:
    response = MagicMock()
    response.id = "resp-claude-456"
    response.choices = [
        MagicMock(
            index=0,
            message=MagicMock(role="assistant", content="Hello from Claude!"),
            finish_reason="stop",
        )
    ]
    response.usage = MagicMock(prompt_tokens=6, completion_tokens=9, total_tokens=15)
    return response


class TestClaudeLLMService:
    def test_init_sets_anthropic_api_key_env(self, monkeypatch, settings: Settings):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ClaudeLLMService(settings)
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-api-key"

    def test_init_does_not_set_api_key_when_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = Settings(llm_platform=LLMPlatform.CLAUDE, anthropic_api_key="")
        ClaudeLLMService(settings)
        assert os.environ.get("ANTHROPIC_API_KEY") is None

    @pytest.mark.asyncio
    async def test_complete_uses_default_model_when_request_has_none(
        self, service: ClaudeLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await service.complete(chat_request)
        assert result.model == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_complete_uses_model_override_from_request(
        self, service: ClaudeLLMService
    ):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="claude-3-haiku-20240307",
        )
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["model"] == "claude-3-haiku-20240307"

    @pytest.mark.asyncio
    async def test_complete_returns_mapped_domain_response(
        self, service: ClaudeLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await service.complete(chat_request)
        assert result.id == "resp-claude-456"
        assert result.choices[0].message.content == "Hello from Claude!"
        assert result.usage is not None
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_complete_passes_temperature_and_max_tokens(
        self, service: ClaudeLLMService
    ):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            temperature=0.2,
            max_tokens=256,
        )
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["temperature"] == 0.2
        assert mock_call.call_args.kwargs["max_tokens"] == 256
