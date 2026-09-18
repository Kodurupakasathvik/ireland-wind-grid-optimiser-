"""
Feature-engineered Linear Regression wind-power forecasting model.

This model extends the lag-only Linear Regression baseline by using
the leakage-safe feature set defined in feature_engineering.py.

Features include:
    - current wind generation
    - lagged wind generation
    - rolling historical statistics
    - cyclic time features

The model predicts:

    P(t + horizon)

All features are constructed using information available at or before
the forecast origin.

All power values are expressed in MW.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.forecasting.feature_engineering import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    WIND_COLUMN,
    create_wind_features,
)


class FeatureEngineeredLinearRegressionWindForecaster:
    """
    Leakage-safe feature-engineered Linear Regression forecaster.

    Parameters
    ----------
    horizon:
        Number of future 15-minute periods to forecast.

    lags:
        Lag periods used as predictors.

    rolling_windows:
        Historical rolling-window sizes.

    wind_column:
        Name of the wind-generation column.
    """

    def __init__(
        self,
        horizon: int = 1,
        lags: Sequence[int] = DEFAULT_LAGS,
        rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
        wind_column: str = WIND_COLUMN,
    ) -> None:

        # --------------------------------------------------------------
        # Validate horizon
        # --------------------------------------------------------------

        if (
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
        ):
            raise ValueError(
                "horizon must be a positive integer."
            )

        if horizon < 1:
            raise ValueError(
                "horizon must be a positive integer."
            )

        self.horizon = horizon

        # --------------------------------------------------------------
        # Validate lags
        # --------------------------------------------------------------

        self.lags = tuple(
            int(value)
            for value in lags
        )

        if not self.lags:
            raise ValueError(
                "At least one lag must be provided."
            )

        if any(
            value < 1
            for value in self.lags
        ):
            raise ValueError(
                "All lag values must be at least 1."
            )

        # --------------------------------------------------------------
        # Validate rolling windows
        # --------------------------------------------------------------

        self.rolling_windows = tuple(
            int(value)
            for value in rolling_windows
        )

        if not self.rolling_windows:
            raise ValueError(
                "At least one rolling window must be provided."
            )

        if any(
            value < 1
            for value in self.rolling_windows
        ):
            raise ValueError(
                "All rolling-window values must be at least 1."
            )

        # --------------------------------------------------------------
        # Validate wind column
        # --------------------------------------------------------------

        if not isinstance(
            wind_column,
            str,
        ):
            raise ValueError(
                "wind_column must be a string."
            )

        if not wind_column:
            raise ValueError(
                "wind_column cannot be empty."
            )

        self.wind_column = wind_column

        # --------------------------------------------------------------
        # Model state
        # --------------------------------------------------------------

        self.model = LinearRegression()

        self.feature_columns: list[str] = []

        self.is_fitted = False

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_series(
        wind_power_mw: pd.Series,
    ) -> pd.Series:
        """
        Validate wind-generation observations.
        """

        if not isinstance(
            wind_power_mw,
            pd.Series,
        ):
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

    # ==================================================================
    # FEATURE CONSTRUCTION
    # ==================================================================

    def _build_features(
        self,
        wind_power_mw: pd.Series,
    ) -> pd.DataFrame:
        """
        Construct the leakage-safe feature matrix.
        """

        dataframe = pd.DataFrame(
            {
                self.wind_column: wind_power_mw,
            },
            index=wind_power_mw.index,
        )

        return create_wind_features(
            dataframe,
            target_horizon=self.horizon,
            wind_column=self.wind_column,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
        )

    # ==================================================================
    # FIT
    # ==================================================================

    def fit(
        self,
        wind_power_mw: pd.Series,
    ) -> "FeatureEngineeredLinearRegressionWindForecaster":
        """
        Fit the feature-engineered Linear Regression model.
        """

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        features = self._build_features(
            wind_power_mw
        )

        target_column = (
            f"target_wind_{self.horizon}"
            f"_steps_ahead_mw"
        )

        if target_column not in features.columns:
            raise ValueError(
                "Expected target column was not created: "
                f"{target_column}"
            )

        # --------------------------------------------------------------
        # Predictor columns
        # --------------------------------------------------------------

        self.feature_columns = [
            column
            for column in features.columns
            if column != target_column
        ]

        if not self.feature_columns:
            raise ValueError(
                "No predictor features were created."
            )

        # --------------------------------------------------------------
        # Training arrays
        # --------------------------------------------------------------

        X = features[
            self.feature_columns
        ].to_numpy(
            dtype=float
        )

        y = features[
            target_column
        ].to_numpy(
            dtype=float
        )

        if len(X) == 0:
            raise ValueError(
                "No training samples were created."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Training features contain NaN or infinite values."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "Training target contains NaN or infinite values."
            )

        # --------------------------------------------------------------
        # Fit model
        # --------------------------------------------------------------

        self.model.fit(
            X,
            y,
        )

        self.is_fitted = True

        return self

    # ==================================================================
    # PREDICT
    # ==================================================================

    def predict(
        self,
        wind_power_mw: pd.Series,
    ) -> list[float]:
        """
        Generate a recursive multi-step forecast.

        For each step, the newest prediction is appended to the
        history before constructing the next feature row.

        Validation order is intentional:

        1. Validate the input series.
        2. Check that sufficient history exists.
        3. Check that the model has been fitted.
        4. Generate the forecast.
        """

        # --------------------------------------------------------------
        # Validate input first.
        #
        # This ensures malformed input and insufficient history are
        # reported independently of fitting state.
        # --------------------------------------------------------------

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        # --------------------------------------------------------------
        # Determine minimum history.
        #
        # Rolling features are calculated from:
        #
        #     wind.shift(1)
        #
        # Therefore a rolling window of N requires N + 1 observations
        # at the forecast origin.
        # --------------------------------------------------------------

        minimum_history = (
            max(
                max(self.lags),
                max(self.rolling_windows),
            )
            + 1
        )

        if len(wind_power_mw) < minimum_history:
            raise ValueError(
                "Insufficient history for prediction. "
                f"At least {minimum_history} observations "
                "are required."
            )

        # --------------------------------------------------------------
        # Check fitted state after input/history validation.
        # --------------------------------------------------------------

        if not self.is_fitted:
            raise RuntimeError(
                "The forecaster must be fitted before prediction."
            )

        # --------------------------------------------------------------
        # Recursive forecasting
        # --------------------------------------------------------------

        history = wind_power_mw.copy()

        forecasts: list[float] = []

        for _ in range(self.horizon):

            # Construct one prediction feature row using only
            # observations currently available.
            features = self._build_prediction_features(
                history
            )

            X = features[
                self.feature_columns
            ].to_numpy(
                dtype=float
            )

            prediction = float(
                self.model.predict(X)[0]
            )

            # Wind generation cannot physically be negative.
            prediction = max(
                0.0,
                prediction,
            )

            forecasts.append(
                prediction
            )

            # ----------------------------------------------------------
            # Recursive step:
            #
            # The prediction becomes the next available observation.
            # ----------------------------------------------------------

            next_timestamp = (
                history.index[-1]
                + pd.Timedelta(minutes=15)
            )

            history.loc[
                next_timestamp
            ] = prediction

        return forecasts

    # ==================================================================
    # PREDICTION FEATURE CONSTRUCTION
    # ==================================================================

    def _build_prediction_features(
        self,
        history: pd.Series,
    ) -> pd.DataFrame:
        """
        Construct one feature row for the next forecast point.

        Unlike training, there is no future target available here,
        so features are constructed directly from historical data.
        """

        dataframe = pd.DataFrame(
            {
                self.wind_column: history,
            },
            index=history.index,
        )

        wind = dataframe[
            self.wind_column
        ]

        # --------------------------------------------------------------
        # Result row
        # --------------------------------------------------------------

        result = pd.DataFrame(
            index=history.index[-1:]
        )

        # --------------------------------------------------------------
        # Current observation
        # --------------------------------------------------------------

        result["wind_current_mw"] = wind.iloc[-1]

        # --------------------------------------------------------------
        # Lag features
        # --------------------------------------------------------------

        for lag in self.lags:

            result[
                f"wind_lag_{lag}_steps_mw"
            ] = wind.shift(lag).iloc[-1]

        # --------------------------------------------------------------
        # Rolling historical features
        #
        # Shift first so the current observation is not included.
        # --------------------------------------------------------------

        historical_wind = wind.shift(1)

        for window in self.rolling_windows:

            rolling = historical_wind.rolling(
                window=window,
                min_periods=window,
            )

            result[
                f"wind_rolling_{window}_steps_mean_mw"
            ] = rolling.mean().iloc[-1]

            result[
                f"wind_rolling_{window}_steps_std_mw"
            ] = rolling.std().iloc[-1]

        # --------------------------------------------------------------
        # Cyclic time features
        # --------------------------------------------------------------

        timestamp = result.index

        minutes_since_midnight = (
            timestamp.hour * 60
            + timestamp.minute
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

        # --------------------------------------------------------------
        # Day-of-year cyclic features
        # --------------------------------------------------------------

        day_of_year = (
            timestamp.dayofyear - 1
        )

        year_length = np.where(
            timestamp.is_leap_year,
            366,
            365,
        )

        year_fraction = (
            day_of_year
            / year_length
        )

        result["day_of_year_sin"] = np.sin(
            2 * np.pi * year_fraction
        )

        result["day_of_year_cos"] = np.cos(
            2 * np.pi * year_fraction
        )

        # --------------------------------------------------------------
        # Match exact training feature order
        # --------------------------------------------------------------

        result = result[
            self.feature_columns
        ]

        # --------------------------------------------------------------
        # Final numerical validation
        # --------------------------------------------------------------

        if not np.isfinite(
            result.to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                "Prediction features contain NaN or "
                "infinite values."
            )

        return result


__all__ = [
    "FeatureEngineeredLinearRegressionWindForecaster",
]