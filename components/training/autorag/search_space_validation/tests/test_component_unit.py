"""Tests for the search_space_validation thin wrapper component."""

import inspect
from unittest import mock

import pytest

from ..component import search_space_validation

MOCKED_ENV_VARIABLES = {
    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
    "OGX_CLIENT_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag dependencies used by search_space_validation."""
    mock_create_ogx_client = mock.MagicMock(name="create_ogx_client")
    mock_prepare_with_ogx = mock.MagicMock(name="prepare_search_space_with_ogx")
    mock_validate_model_list = mock.MagicMock(name="_validate_model_list")
    mock_serialize_model = mock.MagicMock(name="_serialize_model", return_value={"model_id": "mock"})
    mock_ensure_sqlite3 = mock.MagicMock(name="ensure_sqlite3")
    mock_pd = mock.MagicMock(name="pandas")

    mock_ogx_module = mock.MagicMock()
    mock_ogx_module.create_ogx_client = mock_create_ogx_client

    mock_prepare_module = mock.MagicMock()
    mock_prepare_module.prepare_search_space_with_ogx = mock_prepare_with_ogx

    mock_search_space_prep_module = mock.MagicMock()
    mock_search_space_prep_module._serialize_model = mock_serialize_model
    mock_search_space_prep_module._validate_model_list = mock_validate_model_list

    mock_compat = mock.MagicMock()
    mock_compat.ensure_sqlite3 = mock_ensure_sqlite3

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.ogx_client": mock_ogx_module,
        "ai4rag.components.optimization": mock.MagicMock(),
        "ai4rag.components.optimization.search_space_preparation": mock_search_space_prep_module,
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": mock.MagicMock(),
        "ai4rag.search_space.prepare.prepare_search_space": mock_prepare_module,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": mock_compat,
        "pandas": mock_pd,
    }
    return (
        modules,
        mock_create_ogx_client,
        mock_prepare_with_ogx,
        mock_validate_model_list,
        mock_serialize_model,
        mock_ensure_sqlite3,
        mock_pd,
    )


def _make_mock_search_space():
    """Build a mock AI4RAGSearchSpace with valid combinations."""
    mock_ss = mock.MagicMock(name="AI4RAGSearchSpace")

    mock_fm_param = mock.MagicMock()
    mock_fm_param.name = "foundation_model"
    mock_fm_param.values = [mock.MagicMock(model_id="gen-1")]

    mock_em_param = mock.MagicMock()
    mock_em_param.name = "embedding_model"
    mock_em_param.values = [mock.MagicMock(model_id="emb-1")]

    mock_cm_param = mock.MagicMock()
    mock_cm_param.name = "chunking_method"
    mock_cm_param.values = ("recursive",)

    mock_cs_param = mock.MagicMock()
    mock_cs_param.name = "chunk_size"
    mock_cs_param.values = (512,)

    mock_co_param = mock.MagicMock()
    mock_co_param.name = "chunk_overlap"
    mock_co_param.values = (128,)

    mock_ss.params = [mock_fm_param, mock_em_param, mock_cm_param, mock_cs_param, mock_co_param]
    mock_ss.combinations = [
        {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 128,
            "foundation_model": "gen-1",
            "embedding_model": "emb-1",
        }
    ]
    mock_ss.__getitem__ = lambda self, key: next(p for p in self.params if p.name == key)

    return mock_ss


class TestSearchSpaceValidationUnitTests:
    """Unit tests for the search_space_validation thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(search_space_validation)
        assert hasattr(search_space_validation, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected parameters."""
        sig = inspect.signature(search_space_validation.python_func)
        params = list(sig.parameters)
        assert "test_data" in params
        assert "validated_search_space" in params
        assert "embedding_models" in params
        assert "generation_models" in params
        assert "preset" in params
        assert sig.parameters["preset"].default == "speed"

    def test_component_does_not_require_extracted_text(self):
        """Component must not have an extracted_text parameter."""
        sig = inspect.signature(search_space_validation.python_func)
        assert "extracted_text" not in sig.parameters

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_prepare_search_space_with_ogx(self, tmp_path):
        """Wrapper calls prepare_search_space_with_ogx with the constructed payload."""
        modules, mock_create_ogx, mock_prepare, mock_validate, mock_serialize, mock_sqlite, mock_pd = (
            _make_ai4rag_mocks()
        )
        mock_ogx_client = mock.MagicMock(name="ogx_client_instance")
        mock_create_ogx.return_value = mock_ogx_client
        mock_prepare.return_value = _make_mock_search_space()
        mock_pd.read_json.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "validated.json")

        with mock.patch.dict("sys.modules", modules):
            search_space_validation.python_func(
                test_data=test_data,
                validated_search_space=output_artifact,
                embedding_models=["emb-1"],
                generation_models=["gen-1"],
            )

        mock_sqlite.assert_called_once()
        mock_create_ogx.assert_called_once_with(
            base_url="https://ogx.example.com",
            api_key="test-api-key",
        )
        mock_validate.assert_any_call(["emb-1"], "embedding_models")
        mock_validate.assert_any_call(["gen-1"], "generation_models")
        mock_prepare.assert_called_once()
        call_kwargs = mock_prepare.call_args
        assert call_kwargs[0][0]["foundation_models"] == [{"model_id": "gen-1"}]
        assert call_kwargs[0][0]["embedding_models"] == [{"model_id": "emb-1"}]

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_none_models_passed_through(self, tmp_path):
        """None embedding_models and generation_models result in an empty payload for models."""
        modules, mock_create_ogx, mock_prepare, mock_validate, _, _, mock_pd = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = _make_mock_search_space()
        mock_pd.read_json.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        output = mock.MagicMock()
        output.path = str(tmp_path / "validated.json")

        with mock.patch.dict("sys.modules", modules):
            search_space_validation.python_func(
                test_data=test_data,
                validated_search_space=output,
                embedding_models=None,
                generation_models=None,
            )

        payload = mock_prepare.call_args[0][0]
        assert "foundation_models" not in payload
        assert "embedding_models" not in payload

    def test_missing_ogx_env_raises_key_error(self, tmp_path):
        """Missing OGX env vars raise KeyError."""
        modules, _, _, _, _, _, _ = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        output = mock.MagicMock()
        output.path = str(tmp_path / "validated.json")

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(KeyError):
                    search_space_validation.python_func(
                        test_data=test_data,
                        validated_search_space=output,
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_ogx, mock_prepare, _, _, _, mock_pd = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.side_effect = ValueError("Model not found in OGX")
        mock_pd.read_json.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        output = mock.MagicMock()
        output.path = str(tmp_path / "validated.json")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="Model not found in OGX"):
                search_space_validation.python_func(
                    test_data=test_data,
                    validated_search_space=output,
                )

    def test_preset_validation_rejects_invalid(self, tmp_path):
        """Invalid preset raises ValueError."""
        modules, _, _, _, _, _, _ = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        output = mock.MagicMock()
        output.path = str(tmp_path / "validated.json")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="preset must be one of"):
                search_space_validation.python_func(
                    test_data=test_data,
                    validated_search_space=output,
                    preset="invalid",
                )

    @pytest.mark.parametrize("preset_value", ["speed", "balanced"])
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_valid_presets_accepted(self, tmp_path, preset_value):
        """Both 'speed' and 'balanced' presets are accepted without error."""
        modules, mock_create_ogx, mock_prepare, _, _, _, mock_pd = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = _make_mock_search_space()
        mock_pd.read_json.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        output = mock.MagicMock()
        output.path = str(tmp_path / "validated.json")

        with mock.patch.dict("sys.modules", modules):
            search_space_validation.python_func(
                test_data=test_data,
                validated_search_space=output,
                preset=preset_value,
            )

        mock_prepare.assert_called_once()

    @pytest.mark.parametrize(
        ("preset_value", "expected_chunking", "expected_chunk_sizes", "expected_chunk_overlaps"),
        [
            ("speed", ["recursive"], [128, 256, 512], [32, 64]),
            ("balanced", ["recursive", "hybrid"], [512, 1024, 2048], [0, 128, 256]),
        ],
    )
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_sets_search_space_params(
        self, tmp_path, preset_value, expected_chunking, expected_chunk_sizes, expected_chunk_overlaps
    ):
        """Preset controls chunking_methods, chunk_sizes, and chunk_overlaps in the payload."""
        modules, mock_create_ogx, mock_prepare, _, _, _, mock_pd = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = _make_mock_search_space()
        mock_pd.read_json.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        output = mock.MagicMock()
        output.path = str(tmp_path / "validated.json")

        with mock.patch.dict("sys.modules", modules):
            search_space_validation.python_func(
                test_data=test_data,
                validated_search_space=output,
                preset=preset_value,
            )

        payload = mock_prepare.call_args[0][0]
        assert payload["chunking_methods"] == expected_chunking
        assert payload["chunk_sizes"] == expected_chunk_sizes
        assert payload["chunk_overlaps"] == expected_chunk_overlaps

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_writes_output_json(self, tmp_path):
        """Validated search space is written as JSON to the output artifact path."""
        import json

        modules, mock_create_ogx, mock_prepare, _, _, _, mock_pd = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = _make_mock_search_space()
        mock_pd.read_json.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        out_path = tmp_path / "validated.json"
        output = mock.MagicMock()
        output.path = str(out_path)

        with mock.patch.dict("sys.modules", modules):
            search_space_validation.python_func(
                test_data=test_data,
                validated_search_space=output,
            )

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "combination_count" in data
        assert "foundation_model" in data
        assert "embedding_model" in data

    def test_component_status_defaults_to_none(self):
        """component_status defaults to None, enabling direct notebook usage."""
        sig = inspect.signature(search_space_validation.python_func)
        param = sig.parameters["component_status"]
        assert param.default is None
