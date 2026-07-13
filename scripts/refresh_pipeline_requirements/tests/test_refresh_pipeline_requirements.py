"""Tests for refresh_pipeline_requirements script."""

from pathlib import Path
from unittest import mock

import pytest

from ..refresh_pipeline_requirements import (
    RefreshRequirementsError,
    build_podman_command,
    compile_pipeline_requirements,
    read_index_url,
    refresh_pipeline_requirements,
    resolve_pipeline_dir,
)


def _write_requirements_in(directory: Path, index_url: str = "https://example.com/simple") -> None:
    directory.mkdir(parents=True)
    (directory / "requirements.in").write_text(
        f"--index-url {index_url}\n\nrequests\n",
        encoding="utf-8",
    )


class TestReadIndexUrl:
    """Tests for read_index_url."""

    def test_reads_index_url(self, tmp_path: Path):
        """Returns the index URL when requirements.in declares one."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text(
            "--index-url https://example.com/simple\n\nrequests\n",
            encoding="utf-8",
        )

        assert read_index_url(requirements_in) == "https://example.com/simple"

    def test_returns_none_when_missing(self, tmp_path: Path):
        """Returns None when requirements.in has no index URL."""
        requirements_in = tmp_path / "requirements.in"
        requirements_in.write_text("requests\n", encoding="utf-8")

        assert read_index_url(requirements_in) is None


class TestResolvePipelineDir:
    """Tests for resolve_pipeline_dir."""

    def test_resolves_relative_pipeline_path(self, tmp_path: Path):
        """Resolves a pipeline path relative to the repository root."""
        pipeline_dir = tmp_path / "pipelines" / "training" / "demo"
        _write_requirements_in(pipeline_dir)

        resolved = resolve_pipeline_dir(tmp_path, "pipelines/training/demo")

        assert resolved == pipeline_dir.resolve()

    def test_raises_when_requirements_in_missing(self, tmp_path: Path):
        """Raises when the pipeline directory lacks requirements.in."""
        pipeline_dir = tmp_path / "pipelines" / "training" / "demo"
        pipeline_dir.mkdir(parents=True)

        with pytest.raises(RefreshRequirementsError, match="Missing requirements.in"):
            resolve_pipeline_dir(tmp_path, pipeline_dir)


class TestBuildPodmanCommand:
    """Tests for build_podman_command."""

    def test_builds_expected_podman_command(self, tmp_path: Path):
        """Builds a verbose Podman command with Hermeto-compatible pip-compile flags."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_podman_command(
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=False,
            dry_run=False,
            verbose=True,
        )

        assert command[0] == "podman"
        assert "PYTHONUNBUFFERED=1" in command
        assert "bash" in command
        assert "-lc" in command
        compile_command = command[command.index("-lc") + 1]
        assert "pip-compile requirements.in" in compile_command
        assert "--generate-hashes" in compile_command
        assert "--allow-unsafe" in compile_command
        assert "--no-header" in compile_command
        assert " -v" in compile_command
        assert "--output-file requirements.txt" in compile_command
        assert "--upgrade" not in compile_command
        assert "--dry-run" not in compile_command
        assert "--quiet" not in compile_command

    def test_quiet_mode_omits_verbose_flags(self, tmp_path: Path):
        """Omits verbose flags when quiet mode is requested."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_podman_command(
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=False,
            dry_run=False,
            verbose=False,
        )

        compile_command = command[command.index("-lc") + 1]
        assert " -v" not in compile_command
        assert "--quiet" in compile_command
        assert "PYTHONUNBUFFERED=1" not in command

    def test_includes_upgrade_and_dry_run_flags(self, tmp_path: Path):
        """Passes upgrade and dry-run flags through to pip-compile."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        command = build_podman_command(
            container_image="registry.example.com/ubi9/python-312:9.8",
            pipeline_dir=pipeline_dir.resolve(),
            upgrade=True,
            dry_run=True,
            verbose=False,
        )

        assert "--upgrade" in command[-1]
        assert "--dry-run" in command[-1]


class TestCompilePipelineRequirements:
    """Tests for compile_pipeline_requirements."""

    def test_runs_podman_compile(self, tmp_path: Path):
        """Invokes Podman to run pip-compile for a valid pipeline directory."""
        pipeline_dir = tmp_path / "pipeline"
        _write_requirements_in(pipeline_dir)

        with mock.patch("subprocess.run") as run_mock:
            compile_pipeline_requirements(pipeline_dir)

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        assert command[0] == "podman"
        assert "pip-compile requirements.in" in command[-1]

    def test_requires_index_url(self, tmp_path: Path):
        """Raises when requirements.in does not declare an index URL."""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        (pipeline_dir / "requirements.in").write_text("requests\n", encoding="utf-8")

        with pytest.raises(RefreshRequirementsError, match="must declare --index-url"):
            compile_pipeline_requirements(pipeline_dir)


class TestRefreshPipelineRequirements:
    """Tests for refresh_pipeline_requirements."""

    def test_refreshes_all_default_pipelines(self, tmp_path: Path):
        """Compiles requirements for each requested pipeline directory."""
        pipelines = [
            tmp_path / "pipelines/training/automl/demo",
            tmp_path / "pipelines/training/autorag/demo",
        ]
        for pipeline_dir in pipelines:
            _write_requirements_in(pipeline_dir)

        with mock.patch(
            "scripts.refresh_pipeline_requirements.refresh_pipeline_requirements.compile_pipeline_requirements"
        ) as compile_mock:
            refresh_pipeline_requirements(
                [str(p.relative_to(tmp_path)) for p in pipelines],
                repo_root=tmp_path,
            )

        assert compile_mock.call_count == 2
