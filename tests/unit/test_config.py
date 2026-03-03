import os

from app.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert settings.aws_region == "us-east-1"
        assert settings.default_model == "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

    def test_aws_profile_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "my-profile")
        settings = Settings()
        assert settings.aws_profile == "my-profile"

    def test_aws_region_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        settings = Settings()
        assert settings.aws_region == "eu-west-1"
