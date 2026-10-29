import pytest

from tests.conftest import ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_loads_and_reindex_button_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the app's ./chroma_db out of the repo
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception

    # This button used to raise NameError: load_index was used before it was defined.
    reindex = next(b for b in app.button if "Re-index" in b.label)
    reindex.click().run()
    assert not app.exception
