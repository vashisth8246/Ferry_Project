import pytest
from backend.app import app

@pytest.fixture
def product_client(tmp_path):
    app.config.update(TESTING=True, PLANS_DB=tmp_path / 'plans.sqlite3')
    return app.test_client()

def test_product_assets(product_client):
    for path in ['/app', '/app/assets/style.css', '/app/assets/app.js']:
        assert product_client.get(path).status_code == 200

def test_scenario_queue_and_economics(product_client):
    body={'date':'2026-07-18', 'seats':25, 'departures':1, 'fare':9, 'cost':350}
    result=product_client.post('/api/product/scenario', json=body)
    assert result.status_code == 200
    data=result.get_json()
    assert len(data['hours']) == 18
    previous=0
    for row in data['hours']:
        assert row['queue'] == max(0, row['expected'] + previous - row['capacity'])
        assert row['low'] <= row['expected'] <= row['high']
        previous=row['queue']
    summary=data['summary']
    assert summary['boarded'] + summary['unserved'] == summary['expected']
    assert summary['contribution'] == summary['boarded']*9 - summary['sailings']*350
    body.update(seats=1500,departures=12)
    assert product_client.post('/api/product/scenario',json=body).get_json()['summary']['peak_queue'] == 0

@pytest.mark.parametrize('body',[{'seats':0},{'date':'invalid'},{'departures':2.5},{'fare':'NaN'},{'uplift':151},{'date':'2025-01-01'}, {'seats':True}, []])
def test_scenario_rejects_invalid_inputs(product_client,body):
    assert product_client.post('/api/product/scenario',json=body).status_code == 400

def test_plan_is_recomputed_and_persists(product_client):
    payload={'name':'Summer Saturday','inputs':{'date':'2026-07-18'},'summary':{'contribution':99999999}}
    result=product_client.post('/api/product/plans',json=payload)
    assert result.status_code == 201
    plan=result.get_json()
    assert plan['summary']['contribution'] != 99999999
    assert product_client.get('/api/product/plans').get_json()[0]['id'] == plan['id']
    assert product_client.post('/api/product/plans',json={'name':''}).status_code == 400

def test_evidence_has_held_out_training(product_client):
    result=product_client.get('/api/product/overview')
    assert result.status_code == 200
    e=result.get_json()['evidence']
    assert e['train_end'] < e['test_start']
    assert e['holdout_hours'] > 0
    assert e['mae'] >= 0
