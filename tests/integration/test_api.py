from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service, reset_services
from app.application.chat_service import ChatService
from app.domain.models import ChatResponse, Choice, Message, Usage
from app.main import create_app


@pytest.fixture
def mock_chat_service() -> ChatService:
    service = ChatService.__new__(ChatService)
    service.chat = AsyncMock(
        return_value=ChatResponse(
            id="chatcmpl-test",
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello from tests!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=4, total_tokens=9),
        )
    )
    return service


@pytest.fixture
def client(mock_chat_service: ChatService) -> TestClient:
    application = create_app()
    application.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    with TestClient(application) as c:
        yield c
    application.dependency_overrides.clear()
    reset_services()


class TestHealthEndpoint:
    def test_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestChatCompletions:
    def test_successful_completion(self, client: TestClient):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_tokens": 256,
        }
        resp = client.post("/v1/chat/completions", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "chatcmpl-test"
        assert body["choices"][0]["message"]["content"] == "Hello from tests!"

    def test_invalid_role_returns_422(self, client: TestClient):
        payload = {
            "messages": [{"role": "unknown", "content": "Hi"}],
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_empty_messages_returns_422(self, client: TestClient):
        payload = {"messages": []}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_temperature_out_of_range_returns_422(self, client: TestClient):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 2.0,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422
