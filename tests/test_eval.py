import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.eval import (
    evaluate_faithfulness,
    evaluate_relevancy,
    evaluate_correctness,
    evaluate_context_precision,
    evaluate_context_recall
)

client = TestClient(app)

def test_eval_metrics_functions():
    # Test safe score bounds for evaluation functions
    assert 0.0 <= evaluate_faithfulness("Contexte test", "Réponse test") <= 1.0
    assert 0.0 <= evaluate_relevancy("Question test", "Réponse test") <= 1.0
    assert 0.0 <= evaluate_correctness("Réponse test", "Ground truth test") <= 1.0
    assert 0.0 <= evaluate_context_precision("Question test", ["Chunk 1", "Chunk 2"]) <= 1.0
    assert 0.0 <= evaluate_context_recall("Question test", "Ground truth", ["Chunk 1"]) <= 1.0

def test_eval_api_endpoints_auth():
    # Login as admin
    login_resp = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test POST /api/eval/run
    response = client.post(
        "/api/eval/run",
        json={"retrieval_mode": "hybrid", "reranking": True, "top_k": 3},
        headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "running"
    
    job_id = data["job_id"]
    
    # Test GET /api/eval/status/{job_id}
    status_resp = client.get(f"/api/eval/status/{job_id}", headers=headers)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert "progress" in status_data

    # Test GET /api/eval/history
    hist_resp = client.get("/api/eval/history", headers=headers)
    assert hist_resp.status_code == 200
    assert isinstance(hist_resp.json(), list)
