from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
)
def run_status_artifact_initialization(
    workspace_path: str,
    pipeline_name: str,
    run_id: str,
    run_status_pipeline_id: str,
    run_status_artifact: dsl.Output[dsl.Artifact],
) -> None:
    """Seed workspace run status and publish an initial KFP artifact for dashboards.

    Creates ``{workspace_path}/.automl/run_status.json`` with every pipeline component
    and stage from the manifest set to ``pending``, then copies that document into
    ``run_status_artifact`` so UIs can show progress before data loading starts.

    Args:
        workspace_path: PVC workspace directory for ``.automl/run_status.json``.
        pipeline_name: KFP pipeline job resource name (``dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER``).
        run_id: KFP run ID (``dsl.PIPELINE_JOB_ID_PLACEHOLDER``).
        run_status_pipeline_id: Static ``@dsl.pipeline`` name; must match
            ``run_status_templates/pipelines/<name>.json`` (e.g.
            ``autogluon-tabular-training-pipeline``).
        run_status_artifact: Output artifact containing the initial ``run_status.json`` snapshot.
    """
    from kfp_components.components.training.automl.shared.run_status import (
        RUN_STATUS_ARTIFACT_DISPLAY_NAME,
        init_run_status,
        publish_run_status_artifact,
    )

    if not isinstance(workspace_path, str) or not workspace_path.strip():
        raise ValueError("workspace_path must be a non-empty string.")
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        raise ValueError("pipeline_name must be a non-empty string.")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")
    if not isinstance(run_status_pipeline_id, str) or not run_status_pipeline_id.strip():
        raise ValueError("run_status_pipeline_id must be a non-empty string.")

    init_run_status(
        workspace_path,
        kfp_run_id=run_id,
        pipeline_name=pipeline_name,
        run_status_pipeline_id=run_status_pipeline_id,
    )
    publish_run_status_artifact(
        run_status_artifact.path,
        workspace_path,
        validate=False,
    )
    run_status_artifact.metadata["display_name"] = RUN_STATUS_ARTIFACT_DISPLAY_NAME
