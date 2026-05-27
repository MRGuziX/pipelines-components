"""Resolve the AutoML shared directory for ``embedded_artifact_path``."""

from pathlib import Path


def automl_shared_embed_path(component_file: str) -> str:
    """Return the ``components/training/automl/shared`` directory for a component file."""
    path = Path(component_file).resolve()
    parts = path.parts
    if "data_processing" in parts and "automl" in parts:
        components_idx = parts.index("components")
        return str(Path(*parts[: components_idx + 1]) / "training" / "automl" / "shared")
    return str(path.parent.parent / "shared")
