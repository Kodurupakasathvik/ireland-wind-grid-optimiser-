import numpy as np
import pandas as pd
import pytest

from scripts.forecasting.direct_multi_horizon_linear_regression import (
    DirectMultiHorizonLinearRegressionWindForecaster,
)


def make_wind_data(periods=300):

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

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    assert model.horizons == (
        1,
        2,
        4,
        8,
        16,
    )

    assert model.is_fitted is False
    assert model.models == {}
    assert model.feature_columns == {}


def test_model_can_fit():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    fitted = model.fit(wind)

    assert fitted is model
    assert model.is_fitted is True

    assert set(model.models) == {
        1,
        2,
        4,
        8,
        16,
    }


def test_each_horizon_has_independent_model():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    model.fit(wind)

    assert len(model.models) == 5

    assert (
        model.models[1]
        is not model.models[2]
    )

    assert (
        model.models[2]
        is not model.models[4]
    )


def test_expected_features_are_used():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
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

    for horizon in model.horizons:

        assert expected_features.issubset(
            set(
                model.feature_columns[horizon]
            )
        )


def test_model_can_forecast():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    model.fit(wind)

    forecast = model.predict(wind)

    assert set(forecast) == {
        1,
        2,
        4,
        8,
        16,
    }

    assert all(
        np.isfinite(value)
        for value in forecast.values()
    )


def test_individual_horizon_forecast():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    model.fit(wind)

    forecast = model.predict_horizon(
        wind,
        4,
    )

    assert np.isfinite(forecast)
    assert forecast >= 0


def test_forecasts_are_non_negative():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    model.fit(wind)

    forecast = model.predict(wind)

    assert all(
        value >= 0
        for value in forecast.values()
    )


def test_prediction_requires_fitted_model():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(RuntimeError):
        model.predict(wind)


def test_insufficient_history_is_rejected():

    wind = make_wind_data(
        periods=20
    )

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(ValueError):
        model.predict(wind)


def test_empty_input_is_rejected():

    index = pd.DatetimeIndex([])

    wind = pd.Series(
        [],
        index=index,
        dtype=float,
        name="wind_generation_mw",
    )

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(ValueError):
        model.fit(wind)


def test_negative_wind_is_rejected():

    wind = make_wind_data()

    wind.iloc[50] = -1.0

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(ValueError):
        model.fit(wind)


def test_missing_wind_is_rejected():

    wind = make_wind_data()

    wind.iloc[50] = np.nan

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(ValueError):
        model.fit(wind)


def test_unsorted_index_is_rejected():

    wind = make_wind_data()

    wind = wind.iloc[::-1]

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(ValueError):
        model.fit(wind)


def test_duplicate_timestamp_is_rejected():

    wind = make_wind_data()

    duplicate = wind.iloc[[20]]

    wind = pd.concat(
        [
            wind,
            duplicate,
        ]
    )

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster()
    )

    with pytest.raises(ValueError):
        model.fit(wind)


def test_invalid_horizon_configuration():

    with pytest.raises(ValueError):
        DirectMultiHorizonLinearRegressionWindForecaster(
            horizons=[]
        )

    with pytest.raises(ValueError):
        DirectMultiHorizonLinearRegressionWindForecaster(
            horizons=[0]
        )

    with pytest.raises(ValueError):
        DirectMultiHorizonLinearRegressionWindForecaster(
            horizons=[1, 1]
        )


def test_invalid_lag_configuration():

    with pytest.raises(ValueError):
        DirectMultiHorizonLinearRegressionWindForecaster(
            lags=[0]
        )


def test_invalid_rolling_configuration():

    with pytest.raises(ValueError):
        DirectMultiHorizonLinearRegressionWindForecaster(
            rolling_windows=[0]
        )


def test_custom_configuration():

    wind = make_wind_data()

    model = (
        DirectMultiHorizonLinearRegressionWindForecaster(
            horizons=[1, 4, 8],
            lags=[1, 2, 4],
            rolling_windows=[4, 8],
        )
    )

    model.fit(wind)

    assert model.horizons == (
        1,
        4,
        8,
    )

    assert model.lags == (
        1,
        2,
        4,
    )

    assert model.rolling_windows == (
        4,
        8,
    )

    forecast = model.predict(wind)

    assert set(forecast) == {
        1,
        4,
        8,
    }