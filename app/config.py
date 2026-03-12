from enum import Enum

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class LLMPlatform(str, Enum):
    """Supported LLM backend platforms."""

    BEDROCK = "bedrock"
    CLAUDE = "claude"
    LOCAL = "local"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        llm_platform: Which LLM backend to use (bedrock, claude, local).

        aws_profile: AWS profile name used by boto3 for Bedrock auth.
        aws_region: AWS region where Bedrock models are deployed.
        bedrock_default_model: LiteLLM model identifier for Bedrock.

        anthropic_api_key: Anthropic API key for direct Claude access.
        claude_default_model: LiteLLM model identifier for Claude.

        ollama_base_url: Base URL of the Ollama server.
        local_default_model: LiteLLM model identifier for the local model.
    """

    llm_platform: LLMPlatform = LLMPlatform.BEDROCK

    # AWS Bedrock
    aws_profile: str = ""
    aws_region: str = "us-east-1"
    bedrock_default_model: str = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Anthropic Claude (direct)
    anthropic_api_key: SecretStr = SecretStr("")
    claude_default_model: str = "claude-3-5-sonnet-20241022"

    # Local (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    local_default_model: str = "ollama/qwen2.5:0.5b"

    model_config = {"env_prefix": "", "case_sensitive": False}


def get_settings() -> Settings:
    """Build a ``Settings`` instance from the current environment.

    Returns:
        A populated ``Settings`` object.
    """
    return Settings()
