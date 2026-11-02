"""
Flask REST API Backend for Ferry Ticket Demand Forecasting
Serves model predictions and data to the Streamlit frontend
"""
import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Data paths (resolve relative to project root) ──────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR


def _csv_path(name):
    return os.path.join(DATA_DIR, name)


# ── Lazy-loaded caches ─────────────────────────────────────────────────────
_cache = {}


def get_data(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def _load_ferry_data():
    df = pd.read_csv(_csv_path("ferry_tickets_cleaned.csv"))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def _load_model_results():
    return pd.read_csv(_csv_path("all_model_results.csv"), index_col=0)


def _load_uncertainty():
    return pd.read_csv(_csv_path("uncertainty_results.csv"))


def _load_eda_summary():
    path = os.path.join(DATA_DIR, "eda_plots", "summary_statistics.json")
    with open(path) as f:
        return json.load(f)


def _load_uncertainty_summary():
    path = os.path.join(DATA_DIR, "uncertainty_summary.json")
    with open(path) as f:
        return json.load(f)


# ── Health & status ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({"status": "ok", "service": "Ferry Forecasting API", "version": "1.0.0"})


@app.route("/health")
def health():
    """Health-check endpoint used by CI/CD pipeline and load balancers."""
    checks = {}
    try:
        df = get_data("ferry", _load_ferry_data)
        checks["data_loaded"] = True
        checks["records"] = len(df)
    except Exception as e:
        checks["data_loaded"] = False
        checks["error"] = str(e)

    healthy = checks.get("data_loaded", False)
    return jsonify({"healthy": healthy, "checks": checks}), 200 if healthy else 503


# ── Data endpoints ──────────────────────────────────────────────────────────
@app.route("/api/data/summary")
def data_summary():
    """Return dataset summary statistics."""
    df = get_data("ferry", _load_ferry_data)
    return jsonify({
        "total_records": len(df),
        "date_range": {
            "start": str(df["Timestamp"].min()),
            "end": str(df["Timestamp"].max()),
        },
        "columns": list(df.columns),
    })


@app.route("/api/data/recent")
def recent_data():
    """Return the most recent N rows (default 672 = 7 days of 15-min intervals)."""
    n = request.args.get("n", 672, type=int)
    df = get_data("ferry", _load_ferry_data)
    recent = df.tail(n)
    return jsonify(recent.to_dict(orient="records"))


# ── Model endpoints ────────────────────────────────────────────────────────
@app.route("/api/models/results")
def model_results():
    """Return model performance comparison table."""
    results = get_data("models", _load_model_results)
    return jsonify(results.to_dict(orient="index"))


@app.route("/api/models/best")
def best_models():
    """Return the best model per forecast horizon."""
    results = get_data("models", _load_model_results)
    bests = {}
    for horizon in ["15min", "1hr", "2hr"]:
        subset = results[results.index.str.contains(horizon)]
        best_name = subset["MAE"].idxmin()
        bests[horizon] = {
            "model": best_name,
            "MAE": float(subset.loc[best_name, "MAE"]),
            "RMSE": float(subset.loc[best_name, "RMSE"]),
            "MAPE": float(subset.loc[best_name, "MAPE"]),
        }
    return jsonify(bests)


# ── Uncertainty endpoints ──────────────────────────────────────────────────
@app.route("/api/uncertainty/summary")
def uncertainty_summary():
    """Return uncertainty analysis summary."""
    summary = get_data("uncertainty_summary", _load_uncertainty_summary)
    return jsonify(summary)


@app.route("/api/uncertainty/predictions")
def uncertainty_predictions():
    """Return prediction intervals (sampled)."""
    n = request.args.get("n", 500, type=int)
    unc = get_data("uncertainty", _load_uncertainty)
    sample = unc.head(n)
    return jsonify(sample.to_dict(orient="records"))


# ── EDA endpoints ──────────────────────────────────────────────────────────
@app.route("/api/eda/summary")
def eda_summary():
    """Return EDA summary statistics."""
    summary = get_data("eda_summary", _load_eda_summary)
    return jsonify(summary)


@app.route("/api/eda/hourly")
def eda_hourly():
    """Return average hourly patterns."""
    df = get_data("ferry", _load_ferry_data)
    df["hour"] = df["Timestamp"].dt.hour
    hourly = df.groupby("hour")[["Sales Count", "Redemption Count"]].mean()
    return jsonify(hourly.to_dict(orient="index"))


@app.route("/api/eda/daily")
def eda_daily():
    """Return average daily patterns."""
    df = get_data("ferry", _load_ferry_data)
    df["day_name"] = df["Timestamp"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("day_name")[["Sales Count", "Redemption Count"]].mean()
    daily = daily.reindex(day_order)
    return jsonify(daily.to_dict(orient="index"))


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
