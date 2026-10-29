import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mpesa_analyzer.config import ROOT


def test_app_runs_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) >= 5
