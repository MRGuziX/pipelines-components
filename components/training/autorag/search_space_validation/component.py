from pathlib import Path
from typing import List, Optional

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def search_space_validation(
    test_data: dsl.Input[dsl.Artifact],
    validated_search_space: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    embedding_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
):
    """Validate search space configuration before expensive document processing.

    Thin wrapper that validates model availability, parameter constraints,
    and search space viability via
    ``ai4rag.search_space.prepare.prepare_search_space.prepare_search_space_with_ogx``.
    Runs early in the pipeline — before document discovery and text
    extraction — so that misconfiguration is caught in seconds rather
    than after minutes of compute.

    Args:
        test_data: Input artifact with benchmark questions and expected answers.
        validated_search_space: Output artifact for the serialized validated
            search space (JSON).
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        embedding_models: List of embedding model identifiers to try.
        generation_models: List of generation model identifiers to try.
        component_status: Output artifact containing stage-level progress tracking.
        preset: Pipeline quality tier. "speed" (default) uses recursive chunking.
            "balanced" uses hybrid chunking with LLM contextual enrichment.

    Environment variables (required):
        OGX_CLIENT_BASE_URL, OGX_CLIENT_API_KEY.
    """
    import importlib.util
    import json
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    from ai4rag.components.optimization.search_space_preparation import _validate_model_list, serialize_model
    from ai4rag.components.utils.ogx_client import create_ogx_client
    from ai4rag.search_space.prepare.prepare_search_space import prepare_search_space_with_ogx

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_CHUNKING_METHODS = {"speed": ["recursive"], "balanced": ["recursive", "hybrid"]}
    PRESET_CHUNK_SIZES = {"speed": [128, 256, 512], "balanced": [512, 1024, 2048]}
    PRESET_CHUNK_OVERLAPS = {"speed": [32, 64], "balanced": [0, 128, 256]}

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    chunking_methods = PRESET_CHUNKING_METHODS[preset]
    chunk_sizes = PRESET_CHUNK_SIZES[preset]
    chunk_overlaps = PRESET_CHUNK_OVERLAPS[preset]

    logging.info(
        "Preset %r: chunking_methods=%s, chunk_sizes=%s, chunk_overlaps=%s",
        preset,
        chunking_methods,
        chunk_sizes,
        chunk_overlaps,
    )

    if component_status is None:
        from kfp_components.components.training.autorag.shared.component_status import (  # pyright: ignore[reportMissingImports]
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
    else:
        _embedded_path = Path(embedded_artifact.path)
        _module_path = _embedded_path if _embedded_path.is_file() else _embedded_path / "component_status.py"
        _spec = importlib.util.spec_from_file_location("_autorag_component_status", _module_path)
        if _spec is None or _spec.loader is None:
            raise ValueError(f"Cannot load embedded module from {_module_path}")
        _status_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_status_module)
        status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "search_space_validation")
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Search Space Validation Status")
            component_status.metadata["display_name"] = "Search Space Validation Status"
        with status.stage("validate_search_space"):
            _validate_model_list(embedding_models, "embedding_models")
            _validate_model_list(generation_models, "generation_models")

            payload = {}
            if generation_models:
                payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
            if embedding_models:
                payload["embedding_models"] = [{"model_id": em} for em in embedding_models]
            if chunking_methods is not None:
                payload["chunking_methods"] = chunking_methods
            if chunk_sizes is not None:
                payload["chunk_sizes"] = chunk_sizes
            if chunk_overlaps is not None:
                payload["chunk_overlaps"] = chunk_overlaps

            ogx_client = create_ogx_client(
                base_url=os.environ["OGX_CLIENT_BASE_URL"],
                api_key=os.environ["OGX_CLIENT_API_KEY"],
            )

            import pandas as pd

            benchmark_df = pd.read_json(Path(test_data.path))

            search_space = prepare_search_space_with_ogx(
                payload,
                client=ogx_client,
                benchmark_data=benchmark_df,
            )

            for fm in search_space["foundation_model"].values:
                lang = getattr(fm, "language", None)
                if lang is not None:
                    logging.info("Model %s: detected language %s (%s).", fm.model_id, lang.name, lang.code)
                else:
                    logging.warning("Model %s: language detection was not performed.", fm.model_id)

            valid_combinations = search_space.combinations
            if not valid_combinations:
                raise ValueError(
                    "No valid parameter combinations remain after applying search space rules. "
                    "Check chunking_methods, chunk_sizes, and chunk_overlaps for conflicting constraints."
                )

            logging.info(
                "Search space validated: %d valid combinations. chunking_method=%s chunk_size=%s chunk_overlap=%s",
                len(valid_combinations),
                list(search_space["chunking_method"].values),
                list(search_space["chunk_size"].values),
                list(search_space["chunk_overlap"].values),
            )

            non_model_keys = [
                p.name for p in search_space.params if p.name not in ("foundation_model", "embedding_model")
            ]
            result = {key: list(dict.fromkeys(combo[key] for combo in valid_combinations)) for key in non_model_keys}
            result["foundation_model"] = [serialize_model(m) for m in search_space["foundation_model"].values]
            result["embedding_model"] = [serialize_model(m) for m in search_space["embedding_model"].values]
            result["combination_count"] = len(valid_combinations)

            out_path = Path(validated_search_space.path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    Compiler().compile(
        search_space_validation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
