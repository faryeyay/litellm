from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import LLMPlatform, Settings
from app.domain.models import ChatRequest, Message
from app.infrastructure.local_client import LocalLLMService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_platform=LLMPlatform.LOCAL,
        ollama_base_url="http://localhost:11434",
        local_default_model="ollama/qwen2.5:0.5b",
    )


@pytest.fixture
def service(settings: Settings) -> LocalLLMService:
    return LocalLLMService(settings)


@pytest.fixture
def chat_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello local model")])


def _mock_litellm_response() -> MagicMock:
    response = MagicMock()
    response.id = "resp-local-789"
    response.choices = [
        MagicMock(
            index=0,
            message=MagicMock(role="assistant", content="Hello from local!"),
            finish_reason="stop",
        )
    ]
    response.usage = MagicMock(prompt_tokens=4, completion_tokens=7, total_tokens=11)
    return response


class TestLocalLLMService:
    @pytest.mark.asyncio
    async def test_complete_uses_default_local_model(
        self, service: LocalLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await service.complete(chat_request)
        assert result.model == "ollama/qwen2.5:0.5b"

    @pytest.mark.asyncio
    async def test_complete_uses_model_override_from_request(
        self, service: LocalLLMService
    ):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="ollama/gemma:2b",
        )
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["model"] == "ollama/gemma:2b"

    @pytest.mark.asyncio
    async def test_complete_passes_ollama_api_base(
        self, service: LocalLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(chat_request)
        assert mock_call.call_args.kwargs["api_base"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_complete_uses_custom_ollama_base_url(self):
        settings = Settings(
            llm_platform=LLMPlatform.LOCAL,
            ollama_base_url="http://ollama-server:11434",
            local_default_model="ollama/qwen2.5:0.5b",
        )
        service = LocalLLMService(settings)
        request = ChatRequest(messages=[Message(role="user", content="Hi")])
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["api_base"] == "http://ollama-server:11434"

    @pytest.mark.asyncio
    async def test_complete_returns_mapped_domain_response(
        self, service: LocalLLMService, chat_request: ChatRequest
    ):
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await service.complete(chat_request)
        assert result.id == "resp-local-789"
        assert result.choices[0].message.content == "Hello from local!"
        assert result.usage is not None
        assert result.usage.total_tokens == 11

    @pytest.mark.asyncio
    async def test_complete_passes_temperature_and_max_tokens(
        self, service: LocalLLMService
    ):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            temperature=0.1,
            max_tokens=128,
        )
        mock_response = _mock_litellm_response()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await service.complete(request)
        assert mock_call.call_args.kwargs["temperature"] == 0.1
        assert mock_call.call_args.kwargs["max_tokens"] == 128
