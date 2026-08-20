"""
Leakage-safe feature engineering for EirGrid wind forecasting.

The feature set is designed for quarter-hourly wind-generation data.

Features:
    - lagged wind generation
    - rolling historical statistics
    - cyclic time features

IMPORTANT
---------
All features use only information available at or before time t.

The target is the future wind-generation value at time t + horizon.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


WIND_COLUMN = "wind_generation_mw"

DEFAULT_LAGS = (
    1,    # 15 minutes
    2,    # 30 minutes
    4,    # 1 hour
    8,    # 2 hours
    16,   # 4 hours
)

DEFAULT_ROLLING_WINDOWS = (
    4,    # 1 hour
    16,   # 4 hours
)


def create_wind_features(
    dataframe: pd.DataFrame,
    *,
    target_horizon: int = 1,
    wind_column: str = WIND_COLUMN,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Create leakage-safe supervised-learning features.

    Parameters
    ----------
    dataframe:
        DataFrame containing a DateTimeIndex and wind-generation data.

    target_horizon:
        Number of 15-minute periods ahead to predict.

        Example:
            1 = 15-minute forecast
            2 = 30-minute forecast
            4 = 1-hour forecast

    wind_column:
        Name of the observed wind-generation column.

    lags:
        Lag periods expressed in 15-minute steps.

    rolling_windows:
        Rolling-window sizes expressed in 15-minute steps.

    Returns
    -------
    pandas.DataFrame
        Feature matrix plus the future target.

    Notes
    -----
    The target is:

        wind(t + target_horizon)

    Every predictor is constructed exclusively from information
    available at time t or earlier.
    """

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(
            "dataframe must use a pandas DatetimeIndex."
        )

    if wind_column not in dataframe.columns:
        raise ValueError(
            f"Required wind column is missing: {wind_column}"
        )

    if target_horizon < 1:
        raise ValueError(
            "target_horizon must be at least 1."
        )

    if dataframe.empty:
        raise ValueError(
            "dataframe cannot be empty."
        )

    if dataframe.index.has_duplicates:
        raise ValueError(
            "DataFrame index contains duplicate timestamps."
        )

    if not dataframe.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be sorted chronologically."
        )

    # --------------------------------------------------------------
    # Validate wind values
    # --------------------------------------------------------------

    wind = pd.to_numeric(
        dataframe[wind_column],
        errors="coerce",
    )

    if wind.isna().any():
        raise ValueError(
            "Wind-generation data contains missing or "
            "non-numeric values."
        )

    if (wind < 0).any():
        raise ValueError(
            "Wind-generation data cannot contain negative values."
        )

    # --------------------------------------------------------------
    # Validate lag/window configuration
    # --------------------------------------------------------------

    lags = tuple(int(value) for value in lags)
    rolling_windows = tuple(
        int(value)
        for value in rolling_windows
    )

    if not lags:
        raise ValueError(
            "At least one lag must be provided."
        )

    if any(value < 1 for value in lags):
        raise ValueError(
            "All lag values must be at least 1."
        )

    if not rolling_windows:
        raise ValueError(
            "At least one rolling window must be provided."
        )

    if any(value < 1 for value in rolling_windows):
        raise ValueError(
            "All rolling-window values must be at least 1."
        )

    # --------------------------------------------------------------
    # Build result
    # --------------------------------------------------------------

    result = pd.DataFrame(
        index=dataframe.index.copy()
    )

    # Current observation.
    #
    # This is allowed because the forecast is made at time t
    # using the observation available at time t.
    result["wind_current_mw"] = wind

    # --------------------------------------------------------------
    # Lag features
    # --------------------------------------------------------------

    for lag in lags:
        result[
            f"wind_lag_{lag}_steps_mw"
        ] = wind.shift(lag)

    # --------------------------------------------------------------
    # Rolling historical statistics
    #
    # IMPORTANT:
    #
    # We shift by one period before rolling.
    #
    # Therefore a feature at time t only uses observations
    # through t-1 and cannot accidentally include the future
    # target t+horizon.
    # --------------------------------------------------------------

    historical_wind = wind.shift(1)

    for window in rolling_windows:

        rolling = historical_wind.rolling(
            window=window,
            min_periods=window,
        )

        result[
            f"wind_rolling_{window}_steps_mean_mw"
        ] = rolling.mean()

        result[
            f"wind_rolling_{window}_steps_std_mw"
        ] = rolling.std()

    # --------------------------------------------------------------
    # Cyclic time features
    # --------------------------------------------------------------

    minutes_since_midnight = (
        result.index.hour * 60
        + result.index.minute
    )

    day_fraction = (
        minutes_since_midnight
        / (24 * 60)
    )

    result["time_sin"] = np.sin(
        2 * np.pi * day_fraction
    )

    result["time_cos"] = np.cos(
        2 * np.pi * day_fraction
    )

    day_of_year = (
        result.index.dayofyear - 1
    )

    year_length = np.where(
        result.index.is_leap_year,
        366,
        365,
    )

    year_fraction = (
        day_of_year / year_length
    )

    result["day_of_year_sin"] = np.sin(
        2 * np.pi * year_fraction
    )

    result["day_of_year_cos"] = np.cos(
        2 * np.pi * year_fraction
    )

    # --------------------------------------------------------------
    # Future target
    # --------------------------------------------------------------

    result[
        f"target_wind_{target_horizon}_steps_ahead_mw"
    ] = wind.shift(-target_horizon)

    # --------------------------------------------------------------
    # Remove rows where features or target cannot be constructed.
    #
    # No interpolation is performed.
    # --------------------------------------------------------------

    result = result.dropna()

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    if result.empty:
        raise ValueError(
            "No complete feature rows remain after feature construction."
        )

    return result


__all__ = [
    "WIND_COLUMN",
    "DEFAULT_LAGS",
    "DEFAULT_ROLLING_WINDOWS",
    "create_wind_features",
]