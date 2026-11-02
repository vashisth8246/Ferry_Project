"""Render every dashboard page against the running backend."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'), default_timeout=90).run()
assert not app.exception, list(app.exception)
assert not app.error, list(app.error)
for page in app.sidebar.radio[0].options:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, (page, list(app.exception))
    assert not app.error, (page, list(app.error))
    print(f'Passed: {page}')
