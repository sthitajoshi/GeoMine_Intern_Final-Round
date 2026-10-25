"""Headless app tests (Streamlit AppTest): every scene, model / method, layer toggle and opacity
slider renders without an exception. Skipped when data or models are missing."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READY = (ROOT / "data" / "raw" / "Cuprite_f970619t01p02_r02_sc03.a.rfl.mat").exists() and \
        (ROOT / "models" / "salinas_spatial_cnn3d.pt").exists()
pytestmark = pytest.mark.skipif(not READY, reason="data or models missing")


def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def _toggle_all_layers(at):
    for cb in at.sidebar.checkbox:
        cb.check().run()
        assert not at.exception, (cb.label, at.exception)
    for sl in at.sidebar.slider:
        sl.set_value(0.3).run()
        assert not at.exception, (sl.label, at.exception)


@pytest.mark.parametrize("scene", ["Indian Pines", "Salinas"])
def test_crop_scene_models_layers(scene):
    at = _app()
    at.sidebar.selectbox[0].select(scene.lower().replace(" ", "_")).run()
    assert not at.exception
    for model in ("svm", "cnn3d"):
        at.sidebar.radio[0].set_value(model).run()
        assert not at.exception, at.exception
        assert any("Predicted class" in m.label for m in at.metric)
    at.sidebar.selectbox[1].select("NDRE (red edge)").run()
    _toggle_all_layers(at)
    # pixel inspector via the row/col inputs (the click component sets the same values)
    at.number_input[0].set_value(10).run()
    at.number_input[1].set_value(20).run()
    assert not at.exception
    labels = [b.label for b in at.get("download_button")]
    assert any("PNG" in l for l in labels) and any("PDF" in l for l in labels) and any("TIFF" in l for l in labels)


def test_cuprite_methods_layers_and_inspector():
    at = _app()
    at.sidebar.selectbox[0].select("cuprite").run()
    assert not at.exception
    assert any("no pixel ground truth" in w.value for w in at.warning)
    for method in ("SAM (physics)", "MLP (synthetic-trained)"):
        at.sidebar.radio[0].set_value(method).run()
        assert not at.exception, at.exception
    at.sidebar.selectbox[1].select("calcite").run()
    _toggle_all_layers(at)
    at.number_input[0].set_value(150).run()
    at.number_input[1].set_value(540).run()
    assert not at.exception
    assert any("SAM match" in m.label for m in at.metric)
