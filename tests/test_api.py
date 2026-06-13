from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

def test_ask():

    response = client.post(
        "/ask",
        json={
            "question":"What is a list?"
        }
    )

    assert response.status_code == 200
