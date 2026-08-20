"""
Tests for the persistence wind forecast.
"""

import pytest

from scripts.forecasting.persistence_forecast import (
    persistence_forecast,
)


def test_persistence_forecast_uses_latest_observation():
    """
    The forecast should equal the most recent observed value.
    """

    historical_wind = [
        100.0,
        120.0,
        150.0,
    ]

    forecast = persistence_forecast(
        historical_wind
    )

    assert forecast == [150.0]


def test_persistence_forecast_multiple_periods():
    """
    Every forecast period should use the latest observation.
    """

    historical_wind = [
        100.0,
        120.0,
        150.0,
    ]

    forecast = persistence_forecast(
        historical_wind,
        horizon=3,
    )

    assert forecast == [
        150.0,
        150.0,
        150.0,
    ]


def test_zero_wind_forecast():
    """
    Zero wind is physically valid.
    """

    forecast = persistence_forecast(
        [0.0]
    )

    assert forecast == [0.0]


def test_empty_input_is_rejected():
    """
    The model needs at least one historical observation.
    """

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):

        persistence_forecast([])


def test_invalid_horizon_is_rejected():
    """
    Horizon must be positive.
    """

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):

        persistence_forecast(
            [100.0],
            horizon=0,
        )


def test_negative_wind_is_rejected():
    """
    Wind generation cannot be negative.
    """

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):

        persistence_forecast(
            [-10.0]
        )