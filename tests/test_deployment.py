import io
import json
from contextlib import contextmanager
from scripts import deploy_render


def test_on_commit_mode_verifies_without_triggering_duplicate_deploy(monkeypatch):
    monkeypatch.setenv('RENDER_AUTO_DEPLOY', 'true')
    monkeypatch.setenv('DEPLOY_COMMIT', 'tested-commit')
    monkeypatch.setenv('RENDER_APP_URL', 'https://tidal-ferry.onrender.com')
    monkeypatch.delenv('DEPLOY_HOOK_URL', raising=False)
    urls = []

    @contextmanager
    def open_url(url, timeout):
        assert isinstance(url, str), 'On Commit must not submit a POST deploy request'
        urls.append(url)
        value = {'commit': 'tested-commit'} if url.endswith('/version') else {'healthy': True}
        yield io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(deploy_render, 'urlopen', open_url)
    deploy_render.main()
    assert urls == ['https://tidal-ferry.onrender.com/api/product/version', 'https://tidal-ferry.onrender.com/health']
