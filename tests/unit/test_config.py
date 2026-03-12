import os

from app.config import LLMPlatform, Settings


class TestSettings:
    def test_defaults(self, monkeypatch):
        # Isolate from any AWS env vars that may be set in the local environment.
        for var in ("AWS_REGION", "AWS_REGION_NAME", "AWS_PROFILE", "LLM_PLATFORM",
                    "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        settings = Settings()
        assert settings.llm_platform == LLMPlatform.BEDROCK
        assert settings.aws_region == "us-east-1"
        assert settings.bedrock_default_model == "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        assert settings.claude_default_model == "claude-3-5-sonnet-20241022"
        assert settings.local_default_model == "ollama/qwen2.5:0.5b"
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_aws_profile_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "my-profile")
        settings = Settings()
        assert settings.aws_profile == "my-profile"

    def test_aws_region_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        settings = Settings()
        assert settings.aws_region == "eu-west-1"

    def test_llm_platform_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PLATFORM", "claude")
        settings = Settings()
        assert settings.llm_platform == LLMPlatform.CLAUDE

    def test_llm_platform_local_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PLATFORM", "local")
        settings = Settings()
        assert settings.llm_platform == LLMPlatform.LOCAL

    def test_anthropic_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        settings = Settings()
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test123"

    def test_ollama_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-ollama:11434")
        settings = Settings()
        assert settings.ollama_base_url == "http://my-ollama:11434"

    def test_local_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_DEFAULT_MODEL", "ollama/gemma:2b")
        settings = Settings()
        assert settings.local_default_model == "ollama/gemma:2b"

    def test_bedrock_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_DEFAULT_MODEL", "bedrock/amazon.titan-text-lite-v1")
        settings = Settings()
        assert settings.bedrock_default_model == "bedrock/amazon.titan-text-lite-v1"
