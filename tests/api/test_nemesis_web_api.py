import pytest
import nemesis_web_api
from fastapi.testclient import TestClient

def test_api_root():
    client = TestClient(nemesis_web_api.app)
    resp = client.get("/")
    assert resp.status_code in (200, 404)  # adapt according to your API

# Add more endpoint tests here as actual endpoints become available
