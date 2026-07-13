#!/usr/bin/env python3
"""Refresh Hermeto-compatible requirements.txt lockfiles for RHOAI pipelines.

Compiles ``requirements.in`` into ``requirements.txt`` using ``pip-compile``
with ``--generate-hashes`` inside a Podman container. The RHOAI PyPI index
does not publish macOS-compatible wheels, so compilation must run on Linux
(UBI9 Python 3.12).

See: https://hermetoproject.github.io/hermeto/pip/#requirementstxt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ..lib.discovery import get_repo_root

DEFAULT_CONTAINER_IMAGE = "registry.access.redhat.com/ubi9/python-312:9.8"
DEFAULT_PIPELINES: tuple[str, ...] = (
    "pipelines/training/automl/autogluon_tabular_training_pipeline",
    "pipelines/training/autorag/documents_rag_optimization_pipeline",
)

_INDEX_URL_RE = re.compile(r"^--index-url\s+(\S+)", re.MULTILINE)
_REQUIREMENTS_IN = "requirements.in"
_REQUIREMENTS_TXT = "requirements.txt"


class RefreshRequirementsError(Exception):
    """Raised when requirements refresh cannot proceed."""


def read_index_url(requirements_in: Path) -> str | None:
    """Return the ``--index-url`` value from a requirements input file."""
    content = requirements_in.read_text(encoding="utf-8")
    match = _INDEX_URL_RE.search(content)
    return match.group(1) if match else None


def resolve_pipeline_dir(repo_root: Path, pipeline: str | Path) -> Path:
    """Resolve a pipeline directory and validate required input files exist."""
    pipeline_dir = Path(pipeline)
    if not pipeline_dir.is_absolute():
        pipeline_dir = repo_root / pipeline_dir

    requirements_in = pipeline_dir / _REQUIREMENTS_IN
    if not requirements_in.is_file():
        raise RefreshRequirementsError(f"Missing {_REQUIREMENTS_IN}: {requirements_in}")

    return pipeline_dir.resolve()


def build_podman_command(
    *,
    container_image: str,
    pipeline_dir: Path,
    upgrade: bool,
    dry_run: bool,
    verbose: bool,
) -> list[str]:
    """Build the Podman command that runs pip-compile in the container."""
    compile_flags = [
        "pip-compile",
        _REQUIREMENTS_IN,
        "--generate-hashes",
        "--emit-index-url",
        "--allow-unsafe",
        "--no-header",
        f"--output-file {_REQUIREMENTS_TXT}",
    ]
    if upgrade:
        compile_flags.append("--upgrade")
    if dry_run:
        compile_flags.append("--dry-run")
    if verbose:
        compile_flags.append("-v")

    pip_install = "python3 -m pip install pip-tools" if verbose else "python3 -m pip install --quiet pip-tools"
    compile_command = " ".join([pip_install, "&&", " ".join(compile_flags)])

    command = [
        "podman",
        "run",
        "--rm",
        "-v",
        f"{pipeline_dir}:{pipeline_dir}:z",
        "-w",
        str(pipeline_dir),
    ]
    if verbose:
        command.extend(["-e", "PYTHONUNBUFFERED=1"])
    command.extend([container_image, "bash", "-lc", compile_command])
    return command


def compile_pipeline_requirements(
    pipeline_dir: Path,
    *,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    upgrade: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Compile ``requirements.in`` to ``requirements.txt`` for one pipeline."""
    requirements_in = pipeline_dir / _REQUIREMENTS_IN
    index_url = read_index_url(requirements_in)
    if index_url is None:
        raise RefreshRequirementsError(f"{requirements_in} must declare --index-url for RHOAI package resolution")

    command = build_podman_command(
        container_image=container_image,
        pipeline_dir=pipeline_dir,
        upgrade=upgrade,
        dry_run=dry_run,
        verbose=verbose,
    )

    print(f"Refreshing {pipeline_dir / _REQUIREMENTS_TXT}")
    print(f"  index-url: {index_url}")
    print(f"  image: {container_image}")
    subprocess.run(command, check=True)


def refresh_pipeline_requirements(
    pipelines: Sequence[str | Path],
    *,
    repo_root: Path | None = None,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    upgrade: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Refresh requirements.txt for all given pipeline directories."""
    if repo_root is None:
        repo_root = get_repo_root()

    for pipeline in pipelines:
        pipeline_dir = resolve_pipeline_dir(repo_root, pipeline)
        compile_pipeline_requirements(
            pipeline_dir,
            container_image=container_image,
            upgrade=upgrade,
            dry_run=dry_run,
            verbose=verbose,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Hermeto-compatible requirements.txt lockfiles for RHOAI pipelines using pip-compile in Podman"
        ),
    )
    parser.add_argument(
        "pipelines",
        nargs="*",
        help=(f"Pipeline directories containing requirements.in (default: {', '.join(DEFAULT_PIPELINES)})"),
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_CONTAINER_IMAGE,
        help=f"Podman image to run pip-compile in (default: {DEFAULT_CONTAINER_IMAGE})",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade all dependencies instead of only adding new ones",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what pip-compile would change without writing requirements.txt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live pip-compile progress output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    pipelines = args.pipelines or list(DEFAULT_PIPELINES)

    try:
        refresh_pipeline_requirements(
            pipelines,
            container_image=args.image,
            upgrade=args.upgrade,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except (RefreshRequirementsError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
