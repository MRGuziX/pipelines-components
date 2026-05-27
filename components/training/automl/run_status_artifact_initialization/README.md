# Run Status Artifact Initialization

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview

First step in AutoML tabular and time series pipelines. Seeds ``{workspace}/.automl/run_status.json`` from the pipeline manifest (all components and stages ``pending``) and publishes ``run_status.json`` to a KFP artifact so dashboards can show run progress before data loading.

## Inputs

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `workspace_path` | `str` | `None` | PVC workspace directory. |
| `pipeline_name` | `str` | `None` | KFP pipeline job resource name. |
| `run_id` | `str` | `None` | KFP run ID. |
| `run_status_pipeline_id` | `str` | `None` | Manifest id (e.g. ``autogluon-tabular-training-pipeline``). |
| `run_status_artifact` | `dsl.Output[dsl.Artifact]` | `None` | Initial run status snapshot for the UI. |

## Outputs

Writes ``run_status.json`` under ``run_status_artifact`` with display name ``automl_run_status``.
