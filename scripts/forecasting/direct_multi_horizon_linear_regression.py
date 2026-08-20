"""
Direct multi-horizon feature-engineered Linear Regression
wind-power forecasting model.

One independent Linear Regression model is trained for every
configured forecast horizon.

Example:

    1  -> 15 minutes
    2  -> 30 minutes
    4  -> 1 hour
    8  -> 2 hours
    16 -> 4 hours

Unlike recursive forecasting, predictions from one horizon are
never fed into another horizon.

All features are leakage-safe:

    - current wind generation
    - lagged wind generation
    - rolling historical statistics
    - cyclic time features

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


class DirectMultiHorizonLinearRegressionWindForecaster:
    """
    Direct multi-horizon Linear Regression wind forecaster.

    Parameters
    ----------
    horizons:
        Forecast horizons in 15-minute periods.

    lags:
        Historical lag periods used as predictors.

    rolling_windows:
        Historical rolling-window sizes.

    wind_column:
        Name of the wind-generation column.
    """

    def __init__(
        self,
        horizons: Sequence[int] = (1, 2, 4, 8, 16),
        lags: Sequence[int] = DEFAULT_LAGS,
        rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
        wind_column: str = WIND_COLUMN,
    ) -> None:

        # ----------------------------------------------------------
        # Validate horizons
        # ----------------------------------------------------------

        if isinstance(horizons, (str, bytes)):
            raise ValueError(
                "horizons must be a sequence of positive integers."
            )

        try:
            horizon_values = tuple(
                int(value)
                for value in horizons
            )
        except (TypeError, ValueError):
            raise ValueError(
                "horizons must be a sequence of positive integers."
            )

        if not horizon_values:
            raise ValueError(
                "At least one forecast horizon must be provided."
            )

        if any(value < 1 for value in horizon_values):
            raise ValueError(
                "All horizon values must be at least 1."
            )

        if len(set(horizon_values)) != len(horizon_values):
            raise ValueError(
                "Forecast horizons must be unique."
            )

        self.horizons = tuple(
            sorted(horizon_values)
        )

        # ----------------------------------------------------------
        # Validate lags
        # ----------------------------------------------------------

        if isinstance(lags, (str, bytes)):
            raise ValueError(
                "lags must be a sequence of positive integers."
            )

        try:
            lag_values = tuple(
                int(value)
                for value in lags
            )
        except (TypeError, ValueError):
            raise ValueError(
                "lags must be a sequence of positive integers."
            )

        if not lag_values:
            raise ValueError(
                "At least one lag must be provided."
            )

        if any(value < 1 for value in lag_values):
            raise ValueError(
                "All lag values must be at least 1."
            )

        if len(set(lag_values)) != len(lag_values):
            raise ValueError(
                "Lag values must be unique."
            )

        self.lags = tuple(
            sorted(lag_values)
        )

        # ----------------------------------------------------------
        # Validate rolling windows
        # ----------------------------------------------------------

        if isinstance(rolling_windows, (str, bytes)):
            raise ValueError(
                "rolling_windows must be a sequence of positive "
                "integers."
            )

        try:
            rolling_values = tuple(
                int(value)
                for value in rolling_windows
            )
        except (TypeError, ValueError):
            raise ValueError(
                "rolling_windows must be a sequence of positive "
                "integers."
            )

        if not rolling_values:
            raise ValueError(
                "At least one rolling window must be provided."
            )

        if any(value < 1 for value in rolling_values):
            raise ValueError(
                "All rolling-window values must be at least 1."
            )

        if len(set(rolling_values)) != len(rolling_values):
            raise ValueError(
                "Rolling-window values must be unique."
            )

        self.rolling_windows = tuple(
            sorted(rolling_values)
        )

        # ----------------------------------------------------------
        # Validate wind column
        # ----------------------------------------------------------

        if not isinstance(wind_column, str):
            raise ValueError(
                "wind_column must be a string."
            )

        if not wind_column.strip():
            raise ValueError(
                "wind_column cannot be empty."
            )

        self.wind_column = wind_column

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # Tests expect these to be empty before fit().
        # Models and feature columns are created during fit().
        # ----------------------------------------------------------

        self.models: dict[int, LinearRegression] = {}

        self.feature_columns: dict[int, list[str]] = {}

        self.is_fitted = False

    # ==================================================================
    # Validation
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
    # Minimum prediction history
    # ==================================================================

    def minimum_prediction_history(self) -> int:
        """
        Return the minimum history required for prediction.

        The required history accounts for:

            - largest lag
            - largest rolling window
            - largest forecast horizon

        Returns
        -------
        int
            Minimum number of observations required.
        """

        maximum_lag = max(
            self.lags
        )

        maximum_rolling_window = max(
            self.rolling_windows
        )

        maximum_horizon = max(
            self.horizons
        )

        return (
            maximum_lag
            + maximum_rolling_window
            + maximum_horizon
        )

    # ==================================================================
    # Feature construction
    # ==================================================================

    def _build_features(
        self,
        wind_power_mw: pd.Series,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Construct leakage-safe features for one horizon.
        """

        dataframe = pd.DataFrame(
            {
                self.wind_column: wind_power_mw
            },
            index=wind_power_mw.index,
        )

        return create_wind_features(
            dataframe,
            target_horizon=horizon,
            wind_column=self.wind_column,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
        )

    # ==================================================================
    # Fit
    # ==================================================================

    def fit(
        self,
        wind_power_mw: pd.Series,
    ) -> "DirectMultiHorizonLinearRegressionWindForecaster":
        """
        Fit one independent Linear Regression model per horizon.
        """

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        minimum_training_history = (
            max(self.lags)
            + max(self.rolling_windows)
            + max(self.horizons)
        )

        if len(wind_power_mw) <= minimum_training_history:
            raise ValueError(
                "Insufficient observations for training. "
                f"At least {minimum_training_history + 1} "
                "observations are required."
            )

        # ----------------------------------------------------------
        # Reset fitted state so refitting creates a clean model set.
        # ----------------------------------------------------------

        self.models = {}

        self.feature_columns = {}

        # ----------------------------------------------------------
        # Build one independent model for each horizon.
        # ----------------------------------------------------------

        for horizon in self.horizons:

            features = self._build_features(
                wind_power_mw,
                horizon,
            )

            target_column = (
                f"target_wind_{horizon}"
                f"_steps_ahead_mw"
            )

            if target_column not in features.columns:
                raise ValueError(
                    "Expected target column was not created: "
                    f"{target_column}"
                )

            # ------------------------------------------------------
            # Predictor columns.
            #
            # Every non-target column is a predictor.
            # ------------------------------------------------------

            feature_columns = [
                column
                for column in features.columns
                if not column.startswith(
                    "target_wind_"
                )
            ]

            if not feature_columns:
                raise ValueError(
                    "No predictor features were created."
                )

            self.feature_columns[
                horizon
            ] = feature_columns

            # ------------------------------------------------------
            # Drop rows where feature or target is unavailable.
            # ------------------------------------------------------

            training = features[
                feature_columns
                + [target_column]
            ].dropna()

            if training.empty:
                raise ValueError(
                    "No training samples were created for "
                    f"horizon={horizon}."
                )

            X = training[
                feature_columns
            ].to_numpy(
                dtype=float
            )

            y = training[
                target_column
            ].to_numpy(
                dtype=float
            )

            if not np.isfinite(X).all():
                raise ValueError(
                    "Training features contain NaN or "
                    "infinite values."
                )

            if not np.isfinite(y).all():
                raise ValueError(
                    "Training target contains NaN or "
                    "infinite values."
                )

            # ------------------------------------------------------
            # Create the model ONLY during fit().
            # ------------------------------------------------------

            model = LinearRegression()

            model.fit(
                X,
                y,
            )

            self.models[
                horizon
            ] = model

        self.is_fitted = True

        return self

    # ==================================================================
    # Prediction feature construction
    # ==================================================================

    def _build_prediction_features(
        self,
        history: pd.Series,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Construct one feature row for a forecast origin.

        The same historical information is used for every horizon.
        """

        dataframe = pd.DataFrame(
            {
                self.wind_column: history
            },
            index=history.index,
        )

        wind = dataframe[
            self.wind_column
        ]

        result = pd.DataFrame(
            index=history.index[-1:]
        )

        # ----------------------------------------------------------
        # Current wind
        # ----------------------------------------------------------

        result["wind_current_mw"] = (
            wind.iloc[-1]
        )

        # ----------------------------------------------------------
        # Lag features
        # ----------------------------------------------------------

        for lag in self.lags:

            result[
                f"wind_lag_{lag}_steps_mw"
            ] = (
                wind.shift(lag).iloc[-1]
            )

        # ----------------------------------------------------------
        # Historical rolling features
        #
        # shift(1) prevents the current observation from entering
        # the historical rolling calculation.
        # ----------------------------------------------------------

        historical_wind = wind.shift(1)

        for window in self.rolling_windows:

            rolling = historical_wind.rolling(
                window=window,
                min_periods=window,
            )

            result[
                f"wind_rolling_{window}"
                "_steps_mean_mw"
            ] = (
                rolling.mean().iloc[-1]
            )

            result[
                f"wind_rolling_{window}"
                "_steps_std_mw"
            ] = (
                rolling.std().iloc[-1]
            )

        # ----------------------------------------------------------
        # Time-of-day cyclic features
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Day-of-year cyclic features
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Use the exact feature order belonging to this horizon.
        # ----------------------------------------------------------

        result = result[
            self.feature_columns[
                horizon
            ]
        ]

        # ----------------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------------

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

    # ==================================================================
    # Predict all horizons
    # ==================================================================

    def predict(
        self,
        wind_power_mw: pd.Series,
    ) -> dict[int, float]:
        """
        Forecast every configured horizon.

        Each horizon uses its own independently trained model.

        Validation order:

            1. Validate input.
            2. Check sufficient history.
            3. Check fitted state.
            4. Generate forecasts.
        """

        # ----------------------------------------------------------
        # 1. Validate input.
        # ----------------------------------------------------------

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        # ----------------------------------------------------------
        # 2. Check sufficient history BEFORE fitted state.
        # ----------------------------------------------------------

        minimum_history = (
            self.minimum_prediction_history()
        )

        if len(wind_power_mw) < minimum_history:
            raise ValueError(
                "Insufficient history for prediction. "
                f"At least {minimum_history} observations "
                "are required."
            )

        # ----------------------------------------------------------
        # 3. Check fitted state.
        # ----------------------------------------------------------

        if not self.is_fitted:
            raise RuntimeError(
                "The forecaster must be fitted before prediction."
            )

        # ----------------------------------------------------------
        # Safety check.
        # ----------------------------------------------------------

        if set(self.models) != set(self.horizons):
            raise RuntimeError(
                "The forecaster is marked as fitted but "
                "not all horizon models are available."
            )

        if set(self.feature_columns) != set(
            self.horizons
        ):
            raise RuntimeError(
                "The forecaster is marked as fitted but "
                "feature definitions are incomplete."
            )

        # ----------------------------------------------------------
        # Generate independent forecasts.
        # ----------------------------------------------------------

        forecasts: dict[int, float] = {}

        for horizon in self.horizons:

            prediction_features = (
                self._build_prediction_features(
                    wind_power_mw,
                    horizon,
                )
            )

            X = prediction_features[
                self.feature_columns[
                    horizon
                ]
            ].to_numpy(
                dtype=float
            )

            prediction = float(
                self.models[
                    horizon
                ].predict(X)[0]
            )

            # Wind generation cannot be negative.
            prediction = max(
                0.0,
                prediction,
            )

            forecasts[
                horizon
            ] = prediction

        return forecasts

    # ==================================================================
    # Predict one horizon
    # ==================================================================

    def predict_horizon(
        self,
        wind_power_mw: pd.Series,
        horizon: int,
    ) -> float:
        """
        Forecast one specific configured horizon.
        """

        if not isinstance(
            horizon,
            int,
        ) or isinstance(
            horizon,
            bool,
        ):
            raise ValueError(
                "horizon must be a positive integer."
            )

        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon} is not configured. "
                f"Configured horizons: {self.horizons}"
            )

        wind_power_mw = self._validate_series(
            wind_power_mw
        )

        minimum_history = (
            self.minimum_prediction_history()
        )

        if len(wind_power_mw) < minimum_history:
            raise ValueError(
                "Insufficient history for prediction. "
                f"At least {minimum_history} observations "
                "are required."
            )

        if not self.is_fitted:
            raise RuntimeError(
                "The forecaster must be fitted before prediction."
            )

        if horizon not in self.models:
            raise RuntimeError(
                f"No fitted model is available for "
                f"horizon={horizon}."
            )

        prediction_features = (
            self._build_prediction_features(
                wind_power_mw,
                horizon,
            )
        )

        X = prediction_features[
            self.feature_columns[
                horizon
            ]
        ].to_numpy(
            dtype=float
        )

        prediction = float(
            self.models[
                horizon
            ].predict(X)[0]
        )

        return max(
            0.0,
            prediction,
        )


__all__ = [
    "DirectMultiHorizonLinearRegressionWindForecaster",
]