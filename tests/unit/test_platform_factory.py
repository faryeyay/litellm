import pytest

from app.config import LLMPlatform, Settings
from app.infrastructure.bedrock_client import BedrockLLMService
from app.infrastructure.claude_client import ClaudeLLMService
from app.infrastructure.local_client import LocalLLMService
from app.infrastructure.platform_factory import create_llm_service


class TestCreateLLMService:
    def test_returns_bedrock_service_for_bedrock_platform(self):
        settings = Settings(llm_platform=LLMPlatform.BEDROCK)
        service = create_llm_service(settings)
        assert isinstance(service, BedrockLLMService)

    def test_returns_claude_service_for_claude_platform(self):
        settings = Settings(llm_platform=LLMPlatform.CLAUDE)
        service = create_llm_service(settings)
        assert isinstance(service, ClaudeLLMService)

    def test_returns_local_service_for_local_platform(self):
        settings = Settings(llm_platform=LLMPlatform.LOCAL)
        service = create_llm_service(settings)
        assert isinstance(service, LocalLLMService)

    def test_bedrock_is_default_when_platform_not_set(self):
        settings = Settings()
        service = create_llm_service(settings)
        assert isinstance(service, BedrockLLMService)

    def test_platform_selectable_via_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PLATFORM", "local")
        settings = Settings()
        service = create_llm_service(settings)
        assert isinstance(service, LocalLLMService)
