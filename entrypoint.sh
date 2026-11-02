#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint: starts both the Flask backend and Streamlit frontend
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
trap 'kill ${BACKEND_PID:-} ${FRONTEND_PID:-} 2>/dev/null || true' EXIT
trap 'exit 143' TERM INT

echo "🚀 Starting Ferry Forecasting System..."

# Start Flask backend API in the background
echo "📡 Starting Backend API on port 5000..."
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 backend.app:app &
BACKEND_PID=$!

# Start Streamlit frontend
echo "🖥️  Starting Streamlit Dashboard on port 8501..."
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

echo "✅ Both services started!"
echo "   Backend API:  http://localhost:5000"
echo "   Dashboard:    http://localhost:8501"

# Wait for either process to exit
wait -n $BACKEND_PID $FRONTEND_PID
EXIT_CODE=$?

# If one dies, kill the other
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
exit $EXIT_CODE
