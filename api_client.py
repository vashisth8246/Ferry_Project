"""HTTP client shared by the dashboard and integration checks."""
import os
import requests


def fetch_json(path, **params):
    base = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
    try:
        response = requests.get(f"{base}{path}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            "Cannot load dashboard data from the backend. Start the API and check BACKEND_URL."
        ) from exc
