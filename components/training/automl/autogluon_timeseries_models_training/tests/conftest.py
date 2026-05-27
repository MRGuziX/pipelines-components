"""Test fixtures for autogluon_timeseries_models_training."""

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def inject_run_status_artifact(monkeypatch, tmp_path):
    """Inject run_status_artifact when tests omit it."""
    from ..component import autogluon_timeseries_models_training

    original = autogluon_timeseries_models_training.python_func

    def wrapper(*args, **kwargs):
        if "run_status_artifact" not in kwargs:
            art = mock.MagicMock()
            art.path = str(tmp_path / "run_status_out")
            art.metadata = {}
            kwargs["run_status_artifact"] = art
        return original(*args, **kwargs)

    monkeypatch.setattr(autogluon_timeseries_models_training, "python_func", wrapper)
