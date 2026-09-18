"""
Linear Regression wind-power forecasting model.

This module provides a supervised-learning forecasting baseline using
lagged wind-generation observations.

For each prediction:

    X(t) = [P(t-1), P(t-2), ..., P(t-lags)]

The model learns:

    P(t+h) = f(X(t))

For multi-step forecasting, predictions are generated recursively:
the newest prediction becomes the next lagged observation.

All power values are expressed in MW.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressionWindForecaster:
    """
    Linear Regression wind-power forecaster using lagged observations.

    Parameters
    ----------
    lags:
        Number of previous observations used as model features.

    horizon:
        Number of future 15-minute periods to forecast.
    """

    def __init__(
        self,
        lags: int = 4,
        horizon: int = 1,
    ) -> None:

        if not isinstance(lags, int) or isinstance(lags, bool):
            raise ValueError("lags must be a positive integer.")

        if lags < 1:
            raise ValueError("lags must be a positive integer.")

        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise ValueError("horizon must be a positive integer.")

        if horizon < 1:
            raise ValueError(
                "horizon must be a positive integer."
            )

        self.lags = lags
        self.horizon = horizon

        self.model = LinearRegression()

        self.is_fitted = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_series(
        wind_power_mw: pd.Series,
    ) -> pd.Series:
        """
        Validate and return a clean wind-power Series.
        """

        if not isinstance(wind_power_mw, pd.Series):
            raise ValueError(
                "wind_power_mw must be a pandas Series."
            )

        if wind_power_mw.empty:
            raise ValueError(
                "wind_power_mw cannot be empty."
            )

        if not isinstance(
            wind_power_mw.index,
            pd.DatetimeIndex,
        ):
            raise ValueError(
                "wind_power_mw must use a DatetimeIndex."
            )

        if not wind_power_mw.index.is_monotonic_increasing:
            raise ValueError(
                "wind_power_mw timestamps must be sorted "
                "chronologically."
            )

        if wind_power_mw.index.has_duplicates:
            raise ValueError(
                "wind_power_mw timestamps must be unique."
            )

        values = pd.to_numeric(
            wind_power_mw,
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                "wind_power_mw contains missing or "
                "non-numeric values."
            )

        if (values < 0).any():
            raise ValueError(
                "Wind generation cannot be negative."
            )

        return values.astype(float)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _create_training_data(
        self,
        wind_power_mw: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:

        if len(wind_power_mw) <= self.lags:
            raise ValueError(
                "Insufficient history for the requested "
                f"number of lags ({self.lags})."
            )

        values = wind_power_mw.to_numpy(
            dtype=float
        )

        features = []
        targets = []

        for index in range(
            self.lags,
            len(values),
        ):

            lag_window = values[
                index - self.lags:index
            ]

            features.append(
                lag_window
            )

            targets.append(
                values[index]
            )

        X = np.asarray(
            features,
            dtype=float,
        )

        y = np.asarray(
            targets,
            dtype=float,
        )

        return X, y

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        wind_power_mw: pd.Series,
    ) -> "LinearRegressionWindForecaster":
        """
        Fit the Linear Regression model.

        Parameters
        ----------
        wind_power_mw:
            Historical wind-generation observations.

        Returns
        -------
        LinearRegressionWindForecaster
            The fitted forecaster.
        """

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        X, y = self._create_training_data(
            wind_power_mw
        )

        self.model.fit(
            X,
            y,
        )

        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        wind_power_mw: pd.Series,
    ) -> list[float]:
        """
        Generate a recursive multi-step wind forecast.

        Parameters
        ----------
        wind_power_mw:
            Most recent observed wind-generation history.

        Returns
        -------
        list[float]
            Forecast wind generation in MW.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "The forecaster must be fitted before prediction."
            )

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        if len(wind_power_mw) < self.lags:
            raise ValueError(
                "Insufficient history for prediction. "
                f"At least {self.lags} observations are required."
            )

        history = list(
            wind_power_mw.to_numpy(
                dtype=float
            )
        )

        forecasts: list[float] = []

        for _ in range(self.horizon):

            features = np.asarray(
                history[-self.lags:],
                dtype=float,
            ).reshape(
                1,
                -1,
            )

            prediction = float(
                self.model.predict(features)[0]
            )

            # Wind generation cannot physically be negative.
            prediction = max(
                0.0,
                prediction,
            )

            forecasts.append(
                prediction
            )

            history.append(
                prediction
            )

        return forecasts


__all__ = [
    "LinearRegressionWindForecaster",
]