import pytest
from pydantic import ValidationError

from app.domain.models import ChatRequest, ChatResponse, Choice, Message, Usage


class TestMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            msg = Message(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="admin", content="hello")


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(messages=[Message(role="user", content="hi")])
        assert req.model is None
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.stream is False

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[Message(role="user", content="hi")],
                temperature=1.5,
            )

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                messages=[Message(role="user", content="hi")],
                max_tokens=0,
            )


class TestChatResponse:
    def test_round_trip(self):
        resp = ChatResponse(
            id="chatcmpl-123",
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        data = resp.model_dump()
        rebuilt = ChatResponse.model_validate(data)
        assert rebuilt == resp
