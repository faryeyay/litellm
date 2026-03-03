from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        aws_profile: AWS profile name used by boto3 for Bedrock auth.
        aws_region: AWS region where Bedrock models are deployed.
        default_model: LiteLLM model identifier used when the request
            does not specify one.
    """

    aws_profile: str = ""
    aws_region: str = "us-east-1"
    default_model: str = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

    model_config = {"env_prefix": "", "case_sensitive": False}


def get_settings() -> Settings:
    """Build a ``Settings`` instance from the current environment.

    Returns:
        A populated ``Settings`` object.
    """
    return Settings()
