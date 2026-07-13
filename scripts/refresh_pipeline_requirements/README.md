# Refresh pipeline requirements

Refresh Hermeto-compatible `requirements.txt` lockfiles for RHOAI pipelines.

The RHOAI PyPI index does not publish macOS-compatible wheels, so this script
runs `pip-compile` inside Podman with `registry.access.redhat.com/ubi9/python-312:9.8`.

## Usage

Refresh the default AutoML and AutoRAG pipelines:

```bash
uv run python -m scripts.refresh_pipeline_requirements.refresh_pipeline_requirements
```

Refresh a specific pipeline:

```bash
uv run python -m scripts.refresh_pipeline_requirements.refresh_pipeline_requirements \
  pipelines/training/autorag/documents_rag_optimization_pipeline
```

Upgrade all dependencies:

```bash
uv run python -m scripts.refresh_pipeline_requirements.refresh_pipeline_requirements --upgrade
```

Dry run:

```bash
uv run python -m scripts.refresh_pipeline_requirements.refresh_pipeline_requirements --dry-run
```

Suppress live progress output:

```bash
uv run python -m scripts.refresh_pipeline_requirements.refresh_pipeline_requirements --quiet
```

## Defaults

- **Pipelines**
  - `pipelines/training/automl/autogluon_tabular_training_pipeline`
  - `pipelines/training/autorag/documents_rag_optimization_pipeline`
- **Container image**: `registry.access.redhat.com/ubi9/python-312:9.8`

Each pipeline directory must contain a `requirements.in` with a `--index-url` line.
The generated `requirements.txt` keeps that index URL and includes package hashes
for Hermeto offline builds.
