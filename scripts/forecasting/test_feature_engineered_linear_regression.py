import numpy as np
import pandas as pd
import pytest

from scripts.forecasting.feature_engineered_linear_regression import (
    FeatureEngineeredLinearRegressionWindForecaster,
)


def make_wind_data(periods=200):
    index = pd.date_range(
        "2026-01-01 00:00",
        periods=periods,
        freq="15min",
    )

    values = (
        500.0
        + 100.0
        * np.sin(
            np.arange(periods) / 10.0
        )
    )

    return pd.Series(
        values,
        index=index,
        name="wind_generation_mw",
    )


def test_model_initialises():

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=1,
    )

    assert model.horizon == 1
    assert model.is_fitted is False
    assert model.feature_columns == []


def test_model_can_fit():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=1,
    )

    fitted = model.fit(wind)

    assert fitted is model
    assert model.is_fitted is True
    assert len(model.feature_columns) > 0


def test_expected_features_are_used():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=1,
    )

    model.fit(wind)

    expected_features = {
        "wind_current_mw",
        "wind_lag_1_steps_mw",
        "wind_lag_2_steps_mw",
        "wind_lag_4_steps_mw",
        "wind_lag_8_steps_mw",
        "wind_lag_16_steps_mw",
        "wind_rolling_4_steps_mean_mw",
        "wind_rolling_4_steps_std_mw",
        "wind_rolling_16_steps_mean_mw",
        "wind_rolling_16_steps_std_mw",
        "time_sin",
        "time_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    }

    assert expected_features.issubset(
        set(model.feature_columns)
    )


def test_model_can_forecast():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=1,
    )

    model.fit(wind)

    forecast = model.predict(
        wind
    )

    assert len(forecast) == 1
    assert np.isfinite(forecast).all()


def test_multiple_step_forecast():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=4,
    )

    model.fit(wind)

    forecast = model.predict(
        wind
    )

    assert len(forecast) == 4
    assert np.isfinite(forecast).all()


def test_forecast_is_non_negative():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=4,
    )

    model.fit(wind)

    forecast = model.predict(
        wind
    )

    assert all(
        value >= 0
        for value in forecast
    )


def test_empty_input_is_rejected():

    index = pd.DatetimeIndex([])

    wind = pd.Series(
        [],
        index=index,
        dtype=float,
        name="wind_generation_mw",
    )

    model = FeatureEngineeredLinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_negative_wind_is_rejected():

    wind = make_wind_data()

    wind.iloc[50] = -1.0

    model = FeatureEngineeredLinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_missing_wind_is_rejected():

    wind = make_wind_data()

    wind.iloc[50] = np.nan

    model = FeatureEngineeredLinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_unsorted_index_is_rejected():

    wind = make_wind_data()

    wind = wind.iloc[::-1]

    model = FeatureEngineeredLinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_duplicate_timestamp_is_rejected():

    wind = make_wind_data()

    duplicate = wind.iloc[[20]]

    wind = pd.concat(
        [wind, duplicate]
    )

    model = FeatureEngineeredLinearRegressionWindForecaster()

    with pytest.raises(ValueError):
        model.fit(wind)


def test_invalid_horizon_is_rejected():

    with pytest.raises(ValueError):
        FeatureEngineeredLinearRegressionWindForecaster(
            horizon=0,
        )


def test_invalid_lag_is_rejected():

    with pytest.raises(ValueError):
        FeatureEngineeredLinearRegressionWindForecaster(
            lags=[0],
        )


def test_invalid_rolling_window_is_rejected():

    with pytest.raises(ValueError):
        FeatureEngineeredLinearRegressionWindForecaster(
            rolling_windows=[0],
        )


def test_prediction_requires_fitted_model():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster()

    with pytest.raises(RuntimeError):
        model.predict(wind)


def test_insufficient_prediction_history_is_rejected():

    # Default configuration:
    #
    # max lag = 16
    # max rolling window = 16
    #
    # Rolling features use wind.shift(1), therefore:
    #
    # minimum history = 16 + 1 = 17 observations.
    #
    # 16 observations must therefore be rejected.

    wind = make_wind_data(
        periods=16
    )

    # Fit using sufficient historical data first.
    #
    # This ensures the test is specifically checking the
    # insufficient-history condition rather than the unfitted-model
    # condition.

    training_wind = make_wind_data(
        periods=200
    )

    model = FeatureEngineeredLinearRegressionWindForecaster()

    model.fit(
        training_wind
    )

    with pytest.raises(ValueError):
        model.predict(wind)


def test_custom_horizon_target_configuration():

    wind = make_wind_data()

    model = FeatureEngineeredLinearRegressionWindForecaster(
        horizon=4,
        lags=[1, 2, 4],
        rolling_windows=[4, 8],
    )

    model.fit(wind)

    assert model.horizon == 4
    assert model.lags == (1, 2, 4)
    assert model.rolling_windows == (4, 8)

    forecast = model.predict(
        wind
    )

    assert len(forecast) == 4