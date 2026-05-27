"""Runtime import helpers for AutoML components (embedded or preinstalled package).

KFP adds ``embedded_artifact_path`` to ``sys.path`` at runtime. When the AutoML
image also ships ``kfp-components``, prefer the installed package; otherwise use
the embedded copy of this directory.
"""

from __future__ import annotations

from types import ModuleType


def load_run_status() -> ModuleType:
    """Return the ``run_status`` module from the package or embedded shared dir."""
    try:
        from kfp_components.components.training.automl.shared import run_status
    except ImportError:
        import run_status  # type: ignore[import-not-found]
    else:
        return run_status
    return run_status


def load_leaderboard_utils() -> ModuleType:
    """Return the ``leaderboard_utils`` module from the package or embedded shared dir."""
    try:
        from kfp_components.components.training.automl.shared import leaderboard_utils
    except ImportError:
        import leaderboard_utils  # type: ignore[import-not-found]
    else:
        return leaderboard_utils
    return leaderboard_utils
