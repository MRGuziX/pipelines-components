"""Test fixtures for leaderboard_evaluation."""

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def inject_run_status_defaults(monkeypatch, tmp_path):
    """Inject workspace_path and run_status_artifact when tests omit them."""
    from ..component import leaderboard_evaluation

    original = leaderboard_evaluation.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("workspace_path", str(tmp_path))
        if "run_status_artifact" not in kwargs:
            art = mock.MagicMock()
            art.path = str(tmp_path / "run_status_out")
            art.metadata = {}
            kwargs["run_status_artifact"] = art
        return original(*args, **kwargs)

    monkeypatch.setattr(leaderboard_evaluation, "python_func", wrapper)
