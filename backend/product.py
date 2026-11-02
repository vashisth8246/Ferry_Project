"""Seasonal demand planning and durable operating plans for the product demo."""
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from functools import lru_cache
import json
import math
import sqlite3
import uuid

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request, send_from_directory

product = Blueprint('product', __name__)
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def hourly_data():
    data = pd.read_csv(ROOT / 'ferry_tickets_cleaned.csv', parse_dates=['Timestamp'])
    # Missing hours remain missing rather than being represented as zero demand.
    return data.set_index('Timestamp')[['Sales Count', 'Redemption Count']].resample('h').sum(min_count=1).dropna().reset_index()


def estimate(history, dates):
    h = history.copy()
    h['month'] = h.Timestamp.dt.month
    h['weekend'] = h.Timestamp.dt.dayofweek >= 5
    h['hour'] = h.Timestamp.dt.hour
    seasonal = h.groupby(['month', 'weekend', 'hour'])['Sales Count']
    mean = seasonal.mean()
    spread = seasonal.std().fillna(0)
    fallback = h.groupby('hour')['Sales Count'].mean()
    result = []
    for date in dates:
        key = (date.month, date.dayofweek >= 5, date.hour)
        value = max(0, float(mean.get(key, fallback.get(date.hour, 0))))
        deviation = max(0, float(spread.get(key, value * .5)))
        result.append((value, max(0, value - 1.28 * deviation), value + 1.28 * deviation))
    return result


@lru_cache(maxsize=1)
def evidence():
    data = hourly_data()
    cutoff = data.Timestamp.max().normalize() - pd.Timedelta(days=28)
    train, test = data[data.Timestamp < cutoff], data[data.Timestamp >= cutoff]
    pred = np.array([v[0] for v in estimate(train, test.Timestamp)])
    actual = test['Sales Count'].to_numpy()
    baseline = np.full(len(actual), train['Sales Count'].mean())
    mae = float(np.abs(actual - pred).mean())
    base_mae = float(np.abs(actual - baseline).mean())
    return {'model': 'Seasonal demand baseline', 'unit': 'tickets / hour',
            'mae': round(mae, 2), 'baseline_mae': round(base_mae, 2),
            'improvement_pct': round((1 - mae / base_mae) * 100, 1) if base_mae else 0,
            'holdout_hours': len(test), 'train_end': str(train.Timestamp.max()),
            'test_start': str(test.Timestamp.min()), 'test_end': str(test.Timestamp.max()),
            'method': 'Mean historical hourly sales by month, weekend and hour; hourly mean fallback. Last 28 calendar days held out before fitting.',
            'limitation': 'Ticket sales are a demand proxy, not confirmed boardings. Historical spread is not a calibrated confidence interval. No weather, events or live ticket feed is connected.'}


def number(body, name, default, lower, upper, integer=False):
    value = body.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number')
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper or (integer and not value.is_integer()):
        raise ValueError(f'{name} must be between {lower} and {upper}' + (' and a whole number' if integer else ''))
    return int(value) if integer else value


def simulate(body):
    try:
        date = pd.Timestamp(body.get('date', '2026-07-18'))
        if date.tz is not None or date != date.normalize() or not pd.Timestamp('2026-01-01') <= date <= pd.Timestamp('2030-12-31'):
            raise ValueError('Choose a planning date from 2026 through 2030')
        seats = number(body, 'seats', 250, 25, 1500, True)
        departures = number(body, 'departures', 2, 1, 12, True)
        uplift = number(body, 'uplift', 0, -50, 150)
        fare = number(body, 'fare', 9, 0, 100)
        cost = number(body, 'cost', 350, 0, 10000)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(str(exc)) from exc
    dates = pd.date_range(date + pd.Timedelta(hours=6), periods=18, freq='h')
    values = estimate(hourly_data(), dates)
    rows, queue, boarded = [], 0, 0
    for time, (expected, low, high) in zip(dates, values):
        expected, low, high = [int(round(v * (1 + uplift / 100))) for v in (expected, low, high)]
        capacity = seats * departures
        demand = expected + queue
        served = min(demand, capacity)
        queue = max(0, demand - capacity)
        boarded += served
        rows.append({'time': time.isoformat(), 'hour': time.strftime('%H:%M'), 'expected': expected,
                     'low': low, 'high': high, 'capacity': capacity, 'queue': queue,
                     'boarded': served, 'recommended_departures': max(1, math.ceil(expected / (seats * .85)))})
    total = sum(r['expected'] for r in rows)
    peak = max(rows, key=lambda r: r['expected'])
    sailing_count = departures * len(rows)
    return {'inputs': {'date': str(date.date()), 'seats': seats, 'departures': departures,
                       'uplift': uplift, 'fare': fare, 'cost': cost}, 'hours': rows,
            'summary': {'expected': total, 'boarded': boarded, 'unserved': queue,
                        'peak_queue': max(r['queue'] for r in rows), 'peak_hour': peak['hour'],
                        'peak_demand': peak['expected'], 'sailings': sailing_count,
                        'utilization': round(100 * boarded / (seats * sailing_count), 1),
                        'ticket_value': round(boarded * fare, 2), 'operating_cost': round(sailing_count * cost, 2),
                        'contribution': round(boarded * fare - sailing_count * cost, 2)},
            'basis': 'Historical seasonal baseline; scenario estimates, not live predictions.',
            'spread': 'Historical variability band (mean +/- 1.28 standard deviations); uncalibrated.'}


@contextmanager
def db():
    path = Path(current_app.config.get('PLANS_DB', ROOT / 'instance' / 'plans.sqlite3'))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute('CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)')
    try:
        with connection:
            yield connection
    finally:
        connection.close()


@product.get('/app')
@product.get('/app/')
def interface():
    return send_from_directory(ROOT / 'frontend', 'index.html')


@product.get('/app/assets/<path:name>')
def assets(name):
    return send_from_directory(ROOT / 'frontend', name)


@product.get('/api/product/overview')
def overview():
    data = hourly_data()
    daily = data.set_index('Timestamp')['Sales Count'].resample('D').sum(min_count=1).dropna()
    monthly = data.groupby(data.Timestamp.dt.month)['Sales Count'].mean()
    last = data.Timestamp.max()
    recent = data[data.Timestamp >= last - pd.Timedelta(days=7)].copy()
    recent['Timestamp'] = recent.Timestamp.dt.strftime('%Y-%m-%dT%H:%M:%S')
    return jsonify({'records': int(pd.read_csv(ROOT / 'ferry_tickets_cleaned.csv', usecols=['Timestamp']).shape[0]), 'first_date': str(data.Timestamp.min().date()),
                    'last_date': str(last.date()), 'historical_sales': int(data['Sales Count'].sum()),
                    'daily_average': round(float(daily.mean())), 'peak_day': str(daily.idxmax().date()),
                    'monthly': [{'month': int(k), 'demand': round(float(v), 1)} for k, v in monthly.items()],
                    'recent': recent.to_dict('records'), 'evidence': evidence(),
                    'data_mode': 'Historical dataset / planning sandbox', 'network': 'Toronto Island ferry system',
                    'route_note': 'Source data is network-wide; no individual route breakdown is available.'})


@product.post('/api/product/scenario')
def scenario():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({'error': 'Send a JSON object'}), 400
    try:
        return jsonify(simulate(body))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@product.route('/api/product/plans', methods=['GET', 'POST'])
def plans():
    if request.method == 'GET':
        with db() as connection:
            rows = connection.execute('SELECT * FROM plans ORDER BY created_at DESC LIMIT 100').fetchall()
        return jsonify([{'id': r['id'], 'name': r['name'], 'created_at': r['created_at'], **json.loads(r['payload'])} for r in rows])
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('name'), str) or not 1 <= len(body['name'].strip()) <= 80:
        return jsonify({'error': 'Plan name must contain 1 to 80 characters'}), 400
    try:
        result = simulate(body.get('inputs', {}))
    except (ValueError, AttributeError, TypeError) as exc:
        return jsonify({'error': str(exc)}), 400
    plan_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()
    with db() as connection:
        connection.execute('INSERT INTO plans VALUES (?, ?, ?, ?)', (plan_id, body['name'].strip(), created, json.dumps(result)))
    return jsonify({'id': plan_id, 'name': body['name'].strip(), 'created_at': created, **result}), 201
