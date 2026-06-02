"""Tests for time series back_testing.json builder."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from ..back_testing import (
    _forecast_data_for_item,
    _holdout_frame,
    _item_window_metrics,
    build_back_testing_json,
    filter_finite_metrics,
    serialize_date,
    serialize_timestamp,
)


def _make_panel(item_ids: list[str], timestamps: list[str], target_values: list[float]) -> pd.DataFrame:
    rows = []
    for item_id in item_ids:
        for ts, value in zip(timestamps, target_values, strict=True):
            rows.append((item_id, pd.Timestamp(ts), value))
    index = pd.MultiIndex.from_tuples(
        [(item_id, ts) for item_id, ts, _ in rows],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame({"target": [value for _, _, value in rows]}, index=index)


class TestSerialization:
    """Tests for serialization helpers."""

    def test_filter_finite_metrics_drops_nan(self):
        """Non-finite metric values are omitted."""
        assert filter_finite_metrics({"MASE": 0.5, "MAPE": float("nan"), "RMSE": float("inf")}) == {"MASE": 0.5}

    def test_serialize_timestamp_utc(self):
        """Timestamps serialize to ISO strings with UTC suffix."""
        assert serialize_timestamp(pd.Timestamp("2025-12-08T00:00:00Z")) == "2025-12-08T00:00:00Z"
        assert serialize_timestamp(pd.Timestamp("2025-12-08")) == "2025-12-08T00:00:00Z"

    def test_serialize_date(self):
        """Window bounds serialize to YYYY-MM-DD."""
        assert serialize_date(pd.Timestamp("2025-12-08T15:30:00Z")) == "2025-12-08"


class TestHoldoutHelpers:
    """Tests for holdout and forecast helpers."""

    def test_holdout_frame_takes_last_prediction_length_per_item(self):
        """Holdout uses the last prediction_length rows per series."""
        ts = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        panel = _make_panel(["A"], ts, [1.0, 2.0, 3.0, 4.0])
        holdout = _holdout_frame(panel, prediction_length=2)
        assert len(holdout) == 2
        assert holdout["target"].tolist() == [3.0, 4.0]

    def test_item_window_metrics_computes_mape(self):
        """Per-item window metrics include MAPE from point forecasts."""
        timestamps = ["2025-01-03", "2025-01-04"]
        targets = _make_panel(["A"], timestamps, [100.0, 200.0])
        predictions = pd.DataFrame(
            {"mean": [110.0, 180.0]},
            index=targets.index,
        )
        metrics = _item_window_metrics(predictions, targets, "A", "target", prediction_length=2)
        assert "MAPE" in metrics
        assert metrics["MAPE"] == pytest.approx(10.0)

    def test_forecast_data_includes_actual_and_predicted(self):
        """Forecast rows include actual, predicted, and optional quantile bounds."""
        timestamps = ["2025-01-03"]
        targets = _make_panel(["A"], timestamps, [100.0])
        predictions = pd.DataFrame({"mean": [105.0], "0.1": [95.0], "0.9": [115.0]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=1)
        assert rows[0]["actual"] == 100.0
        assert rows[0]["predicted"] == 105.0
        assert rows[0]["lower_bound"] == 95.0
        assert rows[0]["upper_bound"] == 115.0


class TestBuildBackTestingJson:
    """Tests for build_back_testing_json orchestration."""

    def test_builds_schema_with_mock_predictor(self):
        """Builder emits ADR-shaped payload from mocked AutoGluon backtest APIs."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])

        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        predictor.evaluate.return_value = {"MASE": 0.42, "MAPE": 5.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
            metrics=["MASE", "MAPE"],
        )

        assert payload["model_name"] == "DeepAR_FULL"
        assert payload["num_val_windows"] == 1
        assert payload["per_window_metrics"][0]["test_start"] == "2025-01-03"
        assert payload["per_window_metrics"][0]["metrics"]["MASE"] == 0.42
        assert payload["series_analysis"]["num_series_evaluated"] == 2
        assert payload["series_analysis"]["best_performer"]["item_id"] == "good"
        assert payload["series_analysis"]["worst_performer"]["item_id"] == "bad"
        assert payload["series_analysis"]["best_performer"]["windows"][0]["forecast_data"]
        assert payload["series_analysis"]["best_performer"]["windows"][0]["forecast_data"][0]["timestamp"].endswith("Z")
        assert "schema_version" not in payload
        assert "ranking_metric" not in payload["series_analysis"]

    def test_ranks_by_point_metric_matching_eval_metric(self):
        """Best/worst selection uses eval_metric when it is a computed point-forecast metric."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])

        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        predictor.evaluate.return_value = {"RMSE": 1.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="RMSE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
        )

        assert payload["series_analysis"]["best_performer"]["item_id"] == "good"
        assert "ranking_metric" not in payload["series_analysis"]

    def test_requires_backtest_api(self):
        """Missing backtest methods raise AttributeError."""
        predictor = mock.MagicMock(spec=[])
        with pytest.raises(AttributeError, match="backtest API"):
            build_back_testing_json(
                predictor,
                model_name="DeepAR",
                model_name_full="DeepAR_FULL",
                train_data=pd.DataFrame(),
                eval_metric="MASE",
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                prediction_length=1,
            )
