import pytest
import nemesis_web_ui
from fastapi.testclient import TestClient

def test_ui_root():
    client = TestClient(nemesis_web_ui.app)
    resp = client.get("/")
    assert resp.status_code in (200, 404)  # adapt based on actual routes
