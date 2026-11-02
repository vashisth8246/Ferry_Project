from unittest.mock import Mock, patch
import pytest
from api_client import fetch_json


def test_client_uses_configured_backend(monkeypatch):
    monkeypatch.setenv('BACKEND_URL', 'http://backend:5000/')
    response = Mock()
    response.json.return_value = {'total_records': 10}
    with patch('api_client.requests.get', return_value=response) as get:
        assert fetch_json('/api/data/summary')['total_records'] == 10
        get.assert_called_once_with('http://backend:5000/api/data/summary', params={}, timeout=60)
        response.raise_for_status.assert_called_once()


def test_client_reports_backend_failure():
    import requests
    with patch('api_client.requests.get', side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match='Start the API'):
            fetch_json('/health')
