import pytest
from fastapi.testclient import TestClient
import nemesis_web_api

@pytest.fixture
def client():
    return TestClient(nemesis_web_api.app)

def test_vault_crud(client):
    token = client.post("/login", data={"username": "admin", "password": "admin"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    # Create
    resp_create = client.post("/vault/credentials", json={"domain": "demo.com", "username": "user", "secret": "pass"}, headers=headers)
    assert resp_create.status_code == 200
    cid = resp_create.json()["id"]
    # View
    resp_list = client.get("/vault/credentials", headers=headers)
    assert resp_list.status_code == 200
    found = any(c["id"] == cid for c in resp_list.json()["credentials"])
    assert found
    # Edit
    resp_edit = client.put(f"/vault/credentials/{cid}", json={"field": "username", "value": "newuser"}, headers=headers)
    assert resp_edit.status_code == 200
    # Delete
    resp_del = client.delete(f"/vault/credentials/{cid}", headers=headers)
    assert resp_del.status_code == 200

def test_status_live(client):
    token = client.post("/login", data={"username": "admin", "password": "admin"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.get("/status/live", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data and "plugins" in data

