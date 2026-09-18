"""
EirGrid Direct Multi-Horizon Linear Regression benchmark.

Evaluates the direct multi-horizon feature-engineered linear
regression forecaster against the EirGrid wind-generation
dataset.

Unlike recursive forecasting, each horizon has its own model:

    15 min  -> model trained directly for t+1
    30 min  -> model trained directly for t+2
    1 hour  -> model trained directly for t+4
    2 hours -> model trained directly for t+8
    4 hours -> model trained directly for t+16

This prevents prediction-error accumulation across horizons.

Evaluation uses chronological walk-forward validation.

Results are saved to:

    data/processed/
        direct_multi_horizon_linear_regression_forecast_benchmark.csv
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

from scripts.forecasting.direct_multi_horizon_linear_regression import (
    DirectMultiHorizonLinearRegressionWindForecaster,
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
    / "direct_multi_horizon_linear_regression_forecast_benchmark.csv"
)


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------


def build_train_test_split(
    wind: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Create a chronological train/test split.

    No random shuffling is permitted for time-series forecasting.
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

    if split_index <= minimum_training_history:
        raise ValueError(
            "Insufficient observations for the requested "
            "training configuration."
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
    Evaluate one direct forecast horizon.

    The model is retrained at every walk-forward point.

    Importantly, predictions from one horizon are never fed into
    another horizon.
    """

    train, test = build_train_test_split(
        wind
    )

    actual_values: list[float] = []

    forecast_values: list[float] = []

    history = train.copy()

    # -------------------------------------------------------------
    # Walk through the test set.
    #
    # We move by horizon_steps so each forecast block is evaluated
    # once without overlapping forecast windows.
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
        # Create a fresh direct multi-horizon model.
        #
        # Each horizon receives an independent regression model.
        # ---------------------------------------------------------

        model = (
            DirectMultiHorizonLinearRegressionWindForecaster(
                horizons=tuple(HORIZONS.values()),
                lags=LAGS,
                rolling_windows=ROLLING_WINDOWS,
            )
        )

        # ---------------------------------------------------------
        # Fit using ONLY observations available at this point.
        # ---------------------------------------------------------

        model.fit(history)

        # ---------------------------------------------------------
        # Forecast every configured horizon.
        # ---------------------------------------------------------

        forecasts = model.predict(history)

        # ---------------------------------------------------------
        # Extract the requested horizon.
        # ---------------------------------------------------------

        forecast_value = forecasts[
            horizon_steps
        ]

        # ---------------------------------------------------------
        # For direct forecasting, the prediction corresponds to
        # exactly one target timestamp.
        #
        # For a horizon of h steps, compare the prediction against
        # the h-th future observation.
        # ---------------------------------------------------------

        actual_value = float(
            actual.iloc[-1]
        )

        actual_values.append(
            actual_value
        )

        forecast_values.append(
            float(forecast_value)
        )

        # ---------------------------------------------------------
        # Reveal the complete observed block only AFTER evaluation.
        #
        # This prevents future-data leakage.
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

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    return {
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID DIRECT MULTI-HORIZON "
        "LINEAR REGRESSION FORECAST BENCHMARK"
    )
    print("=" * 80)
    print()

    print("Loading EirGrid wind data...")

    dataframe = load_wind_data()

    if "wind_generation_mw" not in dataframe.columns:
        raise ValueError(
            "Expected column 'wind_generation_mw' "
            "was not found."
        )

    wind = (
        dataframe[
            "wind_generation_mw"
        ]
        .dropna()
        .copy()
    )

    print(
        f"Observations available: {len(wind)}"
    )

    print(
        f"Training fraction: "
        f"{TRAIN_FRACTION:.0%}"
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

        results.append(result)

    results_dataframe = pd.DataFrame(
        results
    )

    print()
    print("=" * 80)
    print(
        "DIRECT MULTI-HORIZON "
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

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()