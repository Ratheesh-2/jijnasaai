import os

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.database import set_db_path, init_db
from backend.routers import health, conversations, documents, costs


def _create_test_app() -> FastAPI:
    """Create a minimal FastAPI app without the full lifespan."""
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(documents.router)
    app.include_router(costs.router)
    return app


@pytest_asyncio.fixture
async def client(tmp_path):
    """Each test gets a fresh database."""
    db_path = str(tmp_path / "test.db")
    set_db_path(db_path)
    await init_db()

    # Create data/uploads for document upload tests
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    app = _create_test_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "document_count" in data
        assert "conversation_count" in data


class TestConversationsEndpoint:
    def test_list_conversations_empty(self, client):
        response = client.get("/conversations")
        assert response.status_code == 200
        assert response.json()["conversations"] == []

    def test_create_conversation(self, client):
        response = client.post(
            "/conversations",
            json={"model_id": "gpt-4o", "title": "Test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["title"] == "Test"
        assert data["model_id"] == "gpt-4o"

    def test_create_and_list(self, client):
        client.post("/conversations", json={"model_id": "gpt-4o"})
        client.post("/conversations", json={"model_id": "gpt-4o-mini"})
        response = client.get("/conversations")
        assert len(response.json()["conversations"]) == 2

    def test_delete_conversation(self, client):
        create_resp = client.post(
            "/conversations", json={"model_id": "gpt-4o"}
        )
        conv_id = create_resp.json()["id"]
        delete_resp = client.delete(f"/conversations/{conv_id}")
        assert delete_resp.status_code == 200

        list_resp = client.get("/conversations")
        assert len(list_resp.json()["conversations"]) == 0

    def test_get_nonexistent_conversation(self, client):
        response = client.get("/conversations/nonexistent-id")
        assert response.status_code == 404


class TestDocumentsEndpoint:
    def test_upload_rejects_unsupported_type(self, client):
        response = client.post(
            "/documents/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_list_documents_empty(self, client):
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json()["documents"] == []


class TestCostsEndpoint:
    def test_cost_summary_empty(self, client):
        response = client.get("/costs/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_cost_usd"] == 0.0
        assert data["total_input_tokens"] == 0
