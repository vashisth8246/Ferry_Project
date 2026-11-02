#!/usr/bin/env bash
set -euo pipefail
# Render serves the product UI and API together; no second research server is needed.
exec gunicorn --bind "0.0.0.0:${PORT:-10000}" --workers 1 --threads 4 --timeout 120 backend.app:app
