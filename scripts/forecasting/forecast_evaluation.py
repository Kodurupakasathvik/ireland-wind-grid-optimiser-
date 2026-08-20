"""
Forecast evaluation utilities.

Provides standard metrics for comparing wind-power forecasts
against observed future wind generation.

All power values are expressed in MW.

The module is intentionally independent of any forecasting model.
It can evaluate persistence forecasts, statistical forecasts,
and later ML forecasts using the same metrics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_inputs(
    actual_mw: Sequence[float],
    forecast_mw: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:

    actual = np.asarray(actual_mw, dtype=float)
    forecast = np.asarray(forecast_mw, dtype=float)

    if actual.ndim != 1 or forecast.ndim != 1:
        raise ValueError(
            "actual_mw and forecast_mw must be one-dimensional."
        )

    if len(actual) == 0:
        raise ValueError(
            "actual_mw cannot be empty."
        )

    if len(actual) != len(forecast):
        raise ValueError(
            "actual_mw and forecast_mw must have the same length."
        )

    if not np.isfinite(actual).all():
        raise ValueError(
            "actual_mw contains NaN or infinite values."
        )

    if not np.isfinite(forecast).all():
        raise ValueError(
            "forecast_mw contains NaN or infinite values."
        )

    if (actual < 0).any():
        raise ValueError(
            "actual_mw cannot contain negative values."
        )

    if (forecast < 0).any():
        raise ValueError(
            "forecast_mw cannot contain negative values."
        )

    return actual, forecast


def mean_absolute_error(
    actual_mw: Sequence[float],
    forecast_mw: Sequence[float],
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE = mean(|actual - forecast|)
    """

    actual, forecast = _validate_inputs(
        actual_mw,
        forecast_mw,
    )

    return float(
        np.mean(
            np.abs(actual - forecast)
        )
    )


def root_mean_squared_error(
    actual_mw: Sequence[float],
    forecast_mw: Sequence[float],
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE = sqrt(mean((actual - forecast)^2))
    """

    actual, forecast = _validate_inputs(
        actual_mw,
        forecast_mw,
    )

    return float(
        np.sqrt(
            np.mean(
                (actual - forecast) ** 2
            )
        )
    )


def normalized_mae(
    actual_mw: Sequence[float],
    forecast_mw: Sequence[float],
) -> float:
    """
    Calculate normalized MAE.

    The normalization denominator is the mean actual
    wind generation.

        NMAE = MAE / mean(actual)

    Returns a fraction, not a percentage.
    """

    actual, forecast = _validate_inputs(
        actual_mw,
        forecast_mw,
    )

    mean_actual = float(
        np.mean(actual)
    )

    if mean_actual == 0:
        raise ValueError(
            "Normalized MAE is undefined when "
            "mean actual wind generation is zero."
        )

    return float(
        np.mean(
            np.abs(actual - forecast)
        )
        / mean_actual
    )


def normalized_mae_percent(
    actual_mw: Sequence[float],
    forecast_mw: Sequence[float],
) -> float:
    """
    Calculate normalized MAE as a percentage.
    """

    return (
        normalized_mae(
            actual_mw,
            forecast_mw,
        )
        * 100.0
    )


def forecast_metrics(
    actual_mw: Sequence[float],
    forecast_mw: Sequence[float],
) -> dict[str, float]:
    """
    Calculate the complete standard forecast metric set.

    Returns
    -------
    dict
        Keys:

        - mae_mw
        - rmse_mw
        - nmae
        - nmae_percent
    """

    return {
        "mae_mw": mean_absolute_error(
            actual_mw,
            forecast_mw,
        ),
        "rmse_mw": root_mean_squared_error(
            actual_mw,
            forecast_mw,
        ),
        "nmae": normalized_mae(
            actual_mw,
            forecast_mw,
        ),
        "nmae_percent": normalized_mae_percent(
            actual_mw,
            forecast_mw,
        ),
    }


__all__ = [
    "mean_absolute_error",
    "root_mean_squared_error",
    "normalized_mae",
    "normalized_mae_percent",
    "forecast_metrics",
]