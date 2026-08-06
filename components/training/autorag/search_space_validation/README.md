# Search Space Validation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Validate search space configuration before expensive document processing.

Thin wrapper that validates model availability, parameter constraints, and search space viability via ``ai4rag.search_space.prepare.prepare_search_space.prepare_search_space_with_ogx``. Runs early in the pipeline — before document discovery and text extraction — so that misconfiguration is caught in
seconds rather than after minutes of compute.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with benchmark questions and expected answers. |
| `validated_search_space` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact for the serialized validated search space (JSON). |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `embedding_models` | `Optional[List]` | `None` | List of embedding model identifiers to try. |
| `generation_models` | `Optional[List]` | `None` | List of generation model identifiers to try. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) uses recursive chunking. "balanced" uses hybrid chunking with LLM contextual enrichment. |

## Metadata 🗂️

- **Name**: search_space_validation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: ~=0.10.3
    - Name: pyYaml, Version: >=6.0.0
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - search-space
  - validation
- **Last Verified**: 2026-08-05 00:00:00+00:00
