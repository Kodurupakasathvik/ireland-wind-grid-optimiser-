"""
Tests for the Linear Regression wind-power forecasting baseline.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.forecasting.linear_regression_forecast import (
    LinearRegressionWindForecaster,
)


def make_training_data():
    """
    Create deterministic synthetic wind data.

    The relationship is intentionally simple so the tests
    can verify that the model learns the signal correctly.
    """

    index = pd.date_range(
        "2026-01-01 00:00:00",
        periods=20,
        freq="15min",
    )

    wind = pd.Series(
        np.arange(20, dtype=float) * 10.0,
        index=index,
        name="wind_generation_mw",
    )

    return wind


def test_model_initialises():
    """The forecaster should initialise successfully."""

    model = LinearRegressionWindForecaster()

    assert model is not None
    assert model.horizon == 1


def test_model_can_fit():
    """The model should fit valid historical wind data."""

    wind = make_training_data()

    model = LinearRegressionWindForecaster(
        lags=4,
        horizon=1,
    )

    model.fit(wind)

    assert model.is_fitted


def test_model_can_forecast():
    """The fitted model should produce the requested forecast."""

    wind = make_training_data()

    model = LinearRegressionWindForecaster(
        lags=4,
        horizon=1,
    )

    model.fit(wind)

    forecast = model.predict(wind)

    assert isinstance(forecast, list)
    assert len(forecast) == 1
    assert isinstance(forecast[0], float)


def test_multiple_step_forecast():
    """The model should support multiple future periods."""

    wind = make_training_data()

    model = LinearRegressionWindForecaster(
        lags=4,
        horizon=4,
    )

    model.fit(wind)

    forecast = model.predict(wind)

    assert isinstance(forecast, list)
    assert len(forecast) == 4

    for value in forecast:
        assert isinstance(value, float)
        assert value >= 0.0


def test_forecast_is_non_negative():
    """Wind forecasts must never be negative."""

    wind = make_training_data()

    model = LinearRegressionWindForecaster(
        lags=4,
        horizon=1,
    )

    model.fit(wind)

    forecast = model.predict(wind)

    assert all(
        value >= 0.0
        for value in forecast
    )


def test_empty_input_is_rejected():
    """Empty training data must be rejected."""

    wind = pd.Series(
        dtype=float,
        name="wind_generation_mw",
    )

    model = LinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_negative_wind_is_rejected():
    """Negative wind observations must be rejected."""

    wind = make_training_data()
    wind.iloc[5] = -10.0

    model = LinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_unsorted_index_is_rejected():
    """Training timestamps must be chronological."""

    wind = make_training_data()

    wind = wind.iloc[::-1]

    model = LinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_duplicate_timestamp_is_rejected():
    """Duplicate timestamps must be rejected."""

    wind = make_training_data()

    duplicate = pd.concat(
        [
            wind,
            wind.iloc[[5]],
        ]
    )

    model = LinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(duplicate)


def test_invalid_lag_is_rejected():
    """The lag count must be positive."""

    with pytest.raises(ValueError):
        LinearRegressionWindForecaster(
            lags=0,
        )


def test_invalid_horizon_is_rejected():
    """The forecast horizon must be positive."""

    with pytest.raises(ValueError):
        LinearRegressionWindForecaster(
            horizon=0,
        )


def test_prediction_requires_fitted_model():
    """Prediction before fitting must be rejected."""

    wind = make_training_data()

    model = LinearRegressionWindForecaster(
        lags=4,
        horizon=1,
    )

    with pytest.raises(RuntimeError):
        model.predict(wind)


def test_insufficient_history_is_rejected():
    """There must be enough observations to construct lag features."""

    wind = pd.Series(
        [100.0, 110.0, 120.0],
        index=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="15min",
        ),
        name="wind_generation_mw",
    )

    model = LinearRegressionWindForecaster(
        lags=4,
        horizon=1,
    )

    with pytest.raises(ValueError):
        model.fit(wind)