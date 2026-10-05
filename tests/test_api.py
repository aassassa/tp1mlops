from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "OWL Media" in response.text

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_login_success():
    response = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert data["role"] == "admin"
    assert data["username"] == "admin"

def test_login_failure():
    response = client.post("/api/login", json={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401

def test_api_stats_unauthorized():
    response = client.get("/api/stats")
    assert response.status_code == 403 # Missing Bearer token results in 403 from fastapi.security HTTPBearer

def test_api_stats_admin():
    # Login as admin
    login_resp = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    token = login_resp.json()["token"]
    
    response = client.get("/api/stats", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "qdrant_status" in response.json()

def test_api_stats_user_forbidden():
    # Login as normal user
    login_resp = client.post("/api/login", json={"username": "user", "password": "user123"})
    token = login_resp.json()["token"]
    
    # normal user tries to access /api/stats (which requires admin)
    response = client.get("/api/stats", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403 # Forbidden
    assert "Accès refusé" in response.json()["detail"]

def test_api_documents_admin():
    # Login as admin
    login_resp = client.post("/api/login", json={"username": "admin", "password": "admin123"})
    token = login_resp.json()["token"]
    
    response = client.get("/api/documents", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
