"""Unit tests for run_status_artifact_initialization."""

import json
from pathlib import Path
from unittest import mock

import pytest
from kfp_components.components.training.automl.shared.run_status import (
    COMPONENT_DATA_LOADER,
    COMPONENT_LEADERBOARD,
    COMPONENT_MODELS_TRAINING,
    PIPELINE_TABULAR_TRAINING,
    RUN_STATUS_ARTIFACT_FILENAME,
)

from ..component import run_status_artifact_initialization


class TestRunStatusArtifactInitialization:
    """Tests for the initialization component."""

    def test_component_is_callable(self):  # noqa: D102
        assert callable(run_status_artifact_initialization)
        assert hasattr(run_status_artifact_initialization, "python_func")

    def test_seeds_workspace_and_publishes_artifact(self, tmp_path):  # noqa: D102
        workspace = tmp_path / "ws"
        workspace.mkdir()
        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "artifact_out")
        artifact.metadata = {}

        run_status_artifact_initialization.python_func(
            workspace_path=str(workspace),
            pipeline_name="tabular-job-1",
            run_id="run-xyz",
            run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
            run_status_artifact=artifact,
        )

        status_file = workspace / ".automl" / "run_status.json"
        assert status_file.is_file()
        doc = json.loads(status_file.read_text())
        assert doc["kfp_run_id"] == "run-xyz"
        assert doc["pipeline_name"] == "tabular-job-1"
        assert doc["run_status_pipeline_id"] == PIPELINE_TABULAR_TRAINING
        assert set(doc["components"]) == {
            COMPONENT_DATA_LOADER,
            COMPONENT_MODELS_TRAINING,
            COMPONENT_LEADERBOARD,
        }
        assert doc["components"][COMPONENT_DATA_LOADER]["state"] == "pending"

        artifact_file = Path(artifact.path) / RUN_STATUS_ARTIFACT_FILENAME
        assert artifact_file.is_file()
        assert artifact.metadata["display_name"] == "automl_run_status"

    def test_rejects_empty_workspace_path(self):  # noqa: D102
        artifact = mock.MagicMock(path="/tmp/out", metadata={})
        with pytest.raises(ValueError, match="workspace_path"):
            run_status_artifact_initialization.python_func(
                workspace_path="",
                pipeline_name="p",
                run_id="r",
                run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
                run_status_artifact=artifact,
            )
