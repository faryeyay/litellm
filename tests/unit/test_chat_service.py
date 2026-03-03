import pytest

from app.application.chat_service import ChatService
from app.domain.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    Usage,
)
from app.domain.services import LLMService


class FakeLLMService(LLMService):
    def __init__(self, response: ChatResponse) -> None:
        self._response = response

    async def complete(self, request: ChatRequest) -> ChatResponse:
        return self._response


@pytest.fixture
def fake_response() -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-fake",
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="I'm fine, thanks!"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.mark.asyncio
async def test_chat_delegates_to_llm(fake_response: ChatResponse):
    service = ChatService(llm=FakeLLMService(fake_response))
    request = ChatRequest(messages=[Message(role="user", content="Hello")])

    result = await service.chat(request)

    assert result == fake_response
    assert result.choices[0].message.content == "I'm fine, thanks!"
