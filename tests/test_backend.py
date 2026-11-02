"""
Tests for the Flask backend API.
Run with: pytest tests/ -v
"""
import sys
import os
import json
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# â”€â”€ Health & Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TestHealth:
    def test_index_returns_ok(self, client):
        resp = client.get("/")
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_check(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "healthy" in data
        assert "checks" in data


# â”€â”€ Data Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TestDataEndpoints:
    def test_data_summary(self, client):
        resp = client.get("/api/data/summary")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert "total_records" in data
            assert "date_range" in data
            assert "columns" in data

    def test_recent_data(self, client):
        resp = client.get("/api/data/recent?n=10")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, list)
            assert len(data) <= 10


# â”€â”€ Model Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TestModelEndpoints:
    def test_model_results(self, client):
        resp = client.get("/api/models/results")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, dict)

    def test_best_models(self, client):
        resp = client.get("/api/models/best")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            for horizon in ["15min", "1hr", "2hr"]:
                assert horizon in data
                assert "model" in data[horizon]
                assert "MAE" in data[horizon]


# â”€â”€ Uncertainty Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TestUncertaintyEndpoints:
    def test_uncertainty_summary(self, client):
        resp = client.get("/api/uncertainty/summary")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, dict)

    def test_uncertainty_predictions(self, client):
        resp = client.get("/api/uncertainty/predictions?n=5")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, list)


# â”€â”€ EDA Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TestEDAEndpoints:
    def test_eda_summary(self, client):
        resp = client.get("/api/eda/summary")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, dict)

    def test_eda_hourly(self, client):
        resp = client.get("/api/eda/hourly")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, dict)

    def test_eda_daily(self, client):
        resp = client.get("/api/eda/daily")
        assert resp.status_code == 200
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, dict)


# â”€â”€ Integration Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TestIntegration:
    def test_frontend_backend_contract(self, client):
        """Verify the API contract that the Streamlit frontend depends on."""
        endpoints = ["/", "/health", "/api/data/summary", "/api/models/best",
                     "/api/eda/summary", "/api/uncertainty/summary"]
        for ep in endpoints:
            resp = client.get(ep)
            assert resp.status_code == 200, f"{ep} returned {resp.status_code}"
            assert resp.content_type == "application/json", f"{ep} not JSON"
