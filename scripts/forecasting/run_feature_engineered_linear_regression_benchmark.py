"""
EirGrid feature-engineered Linear Regression wind-forecast benchmark.

Evaluates the leakage-safe feature-engineered Linear Regression model
against real EirGrid quarter-hourly wind-generation observations.

Forecast horizons:
    15 minutes
    30 minutes
    1 hour
    2 hours
    4 hours

A chronological train/test split is used to prevent future-data
leakage.

Walk-forward evaluation:
    1. Train using observations available before the test period.
    2. Forecast the next horizon.
    3. Compare against observations that actually occurred.
    4. Add those newly observed values to history.
    5. Retrain using only information available at that time.
    6. Repeat.

Results are saved to:
    data/processed/
    feature_engineered_linear_regression_forecast_benchmark.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.data.wind_data_loader import load_wind_data

from scripts.forecasting.forecast_evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    normalized_mae,
    normalized_mae_percent,
)

from scripts.forecasting.feature_engineered_linear_regression import (
    FeatureEngineeredLinearRegressionWindForecaster,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAIN_FRACTION = 0.80

LAGS = (
    1,
    2,
    4,
    8,
    16,
)

ROLLING_WINDOWS = (
    4,
    16,
)

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "1hour": 4,
    "2hour": 8,
    "4hour": 16,
}

OUTPUT_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "feature_engineered_linear_regression_forecast_benchmark.csv"
)


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------


def build_train_test_split(
    wind: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Create a chronological train/test split.

    No random shuffling is used because this is a time-series
    forecasting problem.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        train:
            Observations available before the test period.

        test:
            Observations belonging to the test period only.
    """

    if wind.empty:
        raise ValueError(
            "Wind dataset cannot be empty."
        )

    split_index = int(
        len(wind) * TRAIN_FRACTION
    )

    minimum_training_history = (
        max(
            max(LAGS),
            max(ROLLING_WINDOWS),
        )
        + 1
    )

    if split_index < minimum_training_history:
        raise ValueError(
            "Insufficient observations for the requested "
            "feature-engineering configuration."
        )

    if split_index >= len(wind):
        raise ValueError(
            "Training split leaves no observations for testing."
        )

    train = wind.iloc[:split_index].copy()

    test = wind.iloc[split_index:].copy()

    return train, test


# ---------------------------------------------------------------------
# Horizon evaluation
# ---------------------------------------------------------------------


def evaluate_horizon(
    wind: pd.Series,
    horizon_name: str,
    horizon_steps: int,
) -> dict[str, object]:
    """
    Evaluate feature-engineered Linear Regression for one horizon.

    Walk-forward evaluation is used.

    At every forecast point:

        history
            |
        fit feature-engineered model
            |
        forecast next horizon
            |
        compare against actual
            |
        add actual observations to history
            |
        repeat

    No future test observations are used before they become
    available.
    """

    train, test = build_train_test_split(
        wind
    )

    actual_values: list[float] = []
    forecast_values: list[float] = []

    # -------------------------------------------------------------
    # History initially contains ONLY the training period.
    #
    # This prevents future test data from entering the model.
    # -------------------------------------------------------------

    history = train.copy()

    # -------------------------------------------------------------
    # Walk forward through the test period.
    #
    # Each iteration forecasts horizon_steps observations.
    # -------------------------------------------------------------

    for position in range(
        0,
        len(test) - horizon_steps + 1,
        horizon_steps,
    ):

        actual = test.iloc[
            position:
            position + horizon_steps
        ]

        if len(actual) != horizon_steps:
            continue

        # ---------------------------------------------------------
        # Fit using only observations available at this point.
        # ---------------------------------------------------------

        rolling_model = (
            FeatureEngineeredLinearRegressionWindForecaster(
                horizon=horizon_steps,
                lags=LAGS,
                rolling_windows=ROLLING_WINDOWS,
            )
        )

        rolling_model.fit(
            history
        )

        # ---------------------------------------------------------
        # Generate recursive forecast.
        # ---------------------------------------------------------

        forecast = rolling_model.predict(
            history
        )

        if len(forecast) != horizon_steps:
            raise ValueError(
                "Forecast length does not match requested horizon."
            )

        # ---------------------------------------------------------
        # Store actual and forecast values.
        # ---------------------------------------------------------

        actual_values.extend(
            float(value)
            for value in actual.tolist()
        )

        forecast_values.extend(
            float(value)
            for value in forecast
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Reveal actual observations ONLY after the forecast has
        # been evaluated.
        #
        # This preserves proper walk-forward evaluation.
        # ---------------------------------------------------------

        history = pd.concat(
            [
                history,
                actual,
            ]
        )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if not actual_values:
        raise ValueError(
            f"No evaluation samples were produced for "
            f"horizon={horizon_name}."
        )

    if len(actual_values) != len(forecast_values):
        raise ValueError(
            "Actual and forecast lengths do not match."
        )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    metrics = {
        "horizon": horizon_name,
        "horizon_steps": horizon_steps,
        "samples": len(actual_values),
        "mae_mw": mean_absolute_error(
            actual_values,
            forecast_values,
        ),
        "rmse_mw": root_mean_squared_error(
            actual_values,
            forecast_values,
        ),
        "nmae": normalized_mae(
            actual_values,
            forecast_values,
        ),
        "nmae_percent": normalized_mae_percent(
            actual_values,
            forecast_values,
        ),
    }

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID FEATURE-ENGINEERED "
        "LINEAR REGRESSION FORECAST BENCHMARK"
    )
    print("=" * 80)
    print()

    print("Loading EirGrid wind data...")

    dataframe = load_wind_data()

    if "wind_generation_mw" not in dataframe.columns:
        raise ValueError(
            "Required column 'wind_generation_mw' "
            "was not found in the EirGrid dataset."
        )

    wind = dataframe[
        "wind_generation_mw"
    ].dropna()

    print(
        f"Observations available: {len(wind)}"
    )

    print(
        f"Training fraction: {TRAIN_FRACTION:.0%}"
    )

    print(
        f"Lags: {LAGS}"
    )

    print(
        f"Rolling windows: {ROLLING_WINDOWS}"
    )

    print()

    results: list[dict[str, object]] = []

    for horizon_name, horizon_steps in HORIZONS.items():

        print(
            f"Evaluating {horizon_name} "
            f"({horizon_steps} steps)..."
        )

        result = evaluate_horizon(
            wind=wind,
            horizon_name=horizon_name,
            horizon_steps=horizon_steps,
        )

        results.append(
            result
        )

    results_dataframe = pd.DataFrame(
        results
    )

    print()
    print("=" * 80)
    print(
        "FEATURE-ENGINEERED "
        "LINEAR REGRESSION RESULTS"
    )
    print("=" * 80)
    print()

    print(
        results_dataframe.to_string(
            index=False
        )
    )

    print()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()