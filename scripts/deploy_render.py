"""Trigger the exact tested Render commit, optionally waiting for its live health."""
import json
import os
import time
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def main():
    hook = os.environ.get('DEPLOY_HOOK_URL', '')
    sha = os.environ['DEPLOY_COMMIT']
    automatic = os.environ.get('RENDER_AUTO_DEPLOY', '').lower() == 'true'
    if automatic:
        print('Render On Commit deployment is enabled; verifying it without triggering a duplicate deploy.')
    else:
        if not hook:
            print('Image published. Render is not connected: configure its deployment settings.')
            return
        parts = urlsplit(hook)
        if parts.scheme != 'https' or parts.hostname != 'api.render.com':
            raise SystemExit('DEPLOY_HOOK_URL must be a Render HTTPS deploy hook.')
        query = dict(parse_qsl(parts.query))
        query['ref'] = sha
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))
        try:
            with urlopen(Request(url, method='POST'), timeout=30) as response:
                response.read()
        except URLError:
            raise SystemExit('Render rejected the deployment request. Check the deploy-hook secret and Render service.') from None
        print(f'Render accepted deployment of {sha}.')
    base = os.environ.get('RENDER_APP_URL', '').rstrip('/')
    if not base:
        print('Set repository variable RENDER_APP_URL to enable live deployment verification.')
        return
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            with urlopen(base + '/api/product/version', timeout=20) as response:
                live = json.load(response)
            if live.get('commit') == sha:
                with urlopen(base + '/health', timeout=20) as response:
                    healthy = json.load(response).get('healthy')
                if healthy:
                    print(f'Live deployment verified: {base}/app')
                    return
        except (URLError, ValueError, TimeoutError):
            pass
        time.sleep(15)
    raise SystemExit('Render did not serve the tested commit within 15 minutes. Inspect its deployment logs.')


if __name__ == '__main__':
    main()
