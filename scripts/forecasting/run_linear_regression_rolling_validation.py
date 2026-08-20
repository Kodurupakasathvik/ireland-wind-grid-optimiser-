"""
EirGrid Linear Regression Rolling / Expanding Time-Series Validation.

Purpose
-------
Validate the selected Linear Regression wind forecasting baseline
across multiple unseen chronological validation windows.

Missing wind observations are removed before validation.
No interpolation or synthetic observations are introduced.

Forecast horizons:
    15 minutes
    30 minutes
    1 hour
    2 hours
    4 hours

Metrics:
    MAE
    RMSE
    NMAE
    NMAE %

Validation:
    Expanding-window / walk-forward validation.

Outputs:
    data/processed/linear_regression_rolling_validation.csv
    data/processed/linear_regression_rolling_validation_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.wind_data_loader import load_wind_data

from scripts.forecasting.forecast_evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    normalized_mae,
    normalized_mae_percent,
)

from scripts.forecasting.linear_regression_forecast import (
    LinearRegressionWindForecaster,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED
    / "linear_regression_rolling_validation.csv"
)

SUMMARY_FILE = (
    PROCESSED
    / "linear_regression_rolling_validation_summary.csv"
)

# Number of historical lag observations used by the baseline.
LAGS = 4

# Forecast horizons in 15-minute steps.
HORIZONS = {
    "15min": 1,
    "30min": 2,
    "1hour": 4,
    "2hour": 8,
    "4hour": 16,
}

# Earliest point at which validation may begin.
MINIMUM_TRAIN_FRACTION = 0.60

# Number of chronological validation windows.
NUMBER_OF_WINDOWS = 5

# Fraction of the complete dataset assigned to each validation window.
TEST_FRACTION = 0.05


# ============================================================================
# DATA VALIDATION
# ============================================================================


def validate_wind_series(
    wind: pd.Series,
) -> tuple[pd.Series, int]:
    """
    Validate and clean the wind-generation series.

    Missing observations are removed.

    Missing values are NOT interpolated because interpolation would
    introduce synthetic observations into the forecasting dataset.

    Returns
    -------
    tuple[pd.Series, int]
        cleaned wind series
        number of removed missing observations
    """

    if not isinstance(
        wind.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Wind series must use a DatetimeIndex."
        )

    if wind.empty:
        raise ValueError(
            "Wind dataset cannot be empty."
        )

    if wind.index.has_duplicates:
        raise ValueError(
            "Wind dataset contains duplicate timestamps."
        )

    if not wind.index.is_monotonic_increasing:
        raise ValueError(
            "Wind dataset must be chronologically sorted."
        )

    missing_count = int(
        wind.isna().sum()
    )

    if missing_count > 0:

        print()
        print(
            f"WARNING: Found {missing_count} "
            "missing wind observations."
        )

        print(
            "Removing missing observations."
        )

        print(
            "No interpolation or synthetic "
            "values will be used."
        )

        wind = wind.dropna().copy()

    if wind.empty:
        raise ValueError(
            "No observations remain after "
            "removing missing values."
        )

    if (wind < 0).any():
        raise ValueError(
            "Wind generation cannot contain "
            "negative values."
        )

    return (
        wind.astype(float),
        missing_count,
    )


# ============================================================================
# VALIDATION WINDOW GENERATION
# ============================================================================


def build_validation_windows(
    wind: pd.Series,
) -> list[tuple[int, int, int]]:
    """
    Build chronological expanding-window validation splits.

    Each tuple contains:

        (
            window_number,
            train_end_position,
            test_end_position,
        )

    The training set always begins at the first available
    observation and expands forward through time.
    """

    total = len(wind)

    minimum_train = int(
        total * MINIMUM_TRAIN_FRACTION
    )

    test_size = max(
        1,
        int(total * TEST_FRACTION),
    )

    if minimum_train <= LAGS:
        raise ValueError(
            "Minimum training window is too small."
        )

    if minimum_train + test_size > total:
        raise ValueError(
            "Insufficient observations for "
            "rolling validation."
        )

    maximum_start = (
        total - test_size
    )

    if NUMBER_OF_WINDOWS == 1:

        starts = [
            minimum_train
        ]

    else:

        starts = np.linspace(
            minimum_train,
            maximum_start,
            NUMBER_OF_WINDOWS,
            dtype=int,
        )

    starts = sorted(
        set(
            int(value)
            for value in starts
        )
    )

    windows: list[
        tuple[int, int, int]
    ] = []

    for (
        window_number,
        train_end,
    ) in enumerate(
        starts,
        start=1,
    ):

        test_end = min(
            train_end + test_size,
            total,
        )

        if train_end <= LAGS:
            continue

        if test_end <= train_end:
            continue

        windows.append(
            (
                window_number,
                train_end,
                test_end,
            )
        )

    if not windows:
        raise ValueError(
            "No valid validation windows "
            "could be constructed."
        )

    return windows


# ============================================================================
# SINGLE WINDOW / HORIZON EVALUATION
# ============================================================================


def evaluate_window(
    wind: pd.Series,
    window_number: int,
    train_end: int,
    test_end: int,
    horizon_name: str,
    horizon_steps: int,
) -> dict[str, object]:
    """
    Evaluate one forecast horizon in one validation window.

    Walk-forward evaluation is used.

    The model only receives observations that would have been
    available at the forecast time.

    Actual future observations are added to history only AFTER
    the corresponding forecast has been evaluated.
    """

    train = wind.iloc[
        :train_end
    ].copy()

    test = wind.iloc[
        train_end:test_end
    ].copy()

    if len(train) <= LAGS:
        raise ValueError(
            f"Window {window_number} has "
            "insufficient training history."
        )

    # Only complete forecast horizons are evaluated.
    usable_length = (
        len(test)
        // horizon_steps
    ) * horizon_steps

    if usable_length <= 0:
        raise ValueError(
            f"Window {window_number} has "
            f"insufficient test observations "
            f"for {horizon_name}."
        )

    test = test.iloc[
        :usable_length
    ]

    history = train.copy()

    actual_values: list[float] = []

    forecast_values: list[float] = []

    # ------------------------------------------------------------------
    # Walk forward through the validation period.
    # ------------------------------------------------------------------

    for position in range(
        0,
        len(test),
        horizon_steps,
    ):

        actual = test.iloc[
            position:
            position + horizon_steps
        ]

        if len(actual) != horizon_steps:
            continue

        # --------------------------------------------------------------
        # Fit using only observations currently available.
        # --------------------------------------------------------------

        model = LinearRegressionWindForecaster(
            lags=LAGS,
            horizon=horizon_steps,
        )

        model.fit(
            history
        )

        # --------------------------------------------------------------
        # Forecast the next horizon.
        # --------------------------------------------------------------

        forecast = model.predict(
            history
        )

        # --------------------------------------------------------------
        # Store actual and predicted values.
        # --------------------------------------------------------------

        actual_values.extend(
            float(value)
            for value in actual.tolist()
        )

        forecast_values.extend(
            float(value)
            for value in forecast
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Only after evaluating the forecast do we reveal the
        # actual observations to the next iteration.
        #
        # This prevents future-data leakage.
        # --------------------------------------------------------------

        history = pd.concat(
            [
                history,
                actual,
            ]
        )

    if not actual_values:
        raise ValueError(
            f"No forecast samples generated for "
            f"window={window_number}, "
            f"horizon={horizon_name}."
        )

    # ------------------------------------------------------------------
    # Calculate metrics.
    # ------------------------------------------------------------------

    return {
        "window": window_number,
        "horizon": horizon_name,
        "horizon_steps": horizon_steps,
        "train_observations": len(train),
        "test_observations": len(test),
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


# ============================================================================
# SUMMARY
# ============================================================================


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate validation performance by forecast horizon.
    """

    summary = (
        results
        .groupby(
            [
                "horizon",
                "horizon_steps",
            ],
            as_index=False,
        )
        .agg(
            windows_evaluated=(
                "window",
                "count",
            ),
            total_samples=(
                "samples",
                "sum",
            ),
            mean_mae_mw=(
                "mae_mw",
                "mean",
            ),
            std_mae_mw=(
                "mae_mw",
                "std",
            ),
            worst_mae_mw=(
                "mae_mw",
                "max",
            ),
            mean_rmse_mw=(
                "rmse_mw",
                "mean",
            ),
            std_rmse_mw=(
                "rmse_mw",
                "std",
            ),
            worst_rmse_mw=(
                "rmse_mw",
                "max",
            ),
            mean_nmae_percent=(
                "nmae_percent",
                "mean",
            ),
            std_nmae_percent=(
                "nmae_percent",
                "std",
            ),
            worst_nmae_percent=(
                "nmae_percent",
                "max",
            ),
        )
    )

    # A single validation window has zero standard deviation.
    summary = summary.fillna(
        {
            "std_mae_mw": 0.0,
            "std_rmse_mw": 0.0,
            "std_nmae_percent": 0.0,
        }
    )

    # Lower is better.
    #
    # Mean error + variability gives a simple robustness indicator.
    summary["robustness_score"] = (
        summary["mean_nmae_percent"]
        + summary["std_nmae_percent"]
    )

    return summary


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID LINEAR REGRESSION "
        "ROLLING TIME-SERIES VALIDATION"
    )
    print("=" * 80)
    print()

    print(
        "Primary metric: NMAE (%)"
    )

    print(
        "Lower NMAE is better."
    )

    print()

    # ======================================================================
    # LOAD DATA
    # ======================================================================

    print(
        "Loading EirGrid wind data..."
    )

    dataframe = load_wind_data()

    if "wind_generation_mw" not in dataframe.columns:
        raise ValueError(
            "Expected column "
            "'wind_generation_mw' "
            "was not found."
        )

    wind = dataframe[
        "wind_generation_mw"
    ].copy()

    original_count = len(wind)

    wind, missing_count = (
        validate_wind_series(
            wind
        )
    )

    print()

    print(
        f"Original observations: "
        f"{original_count}"
    )

    print(
        f"Missing observations removed: "
        f"{missing_count}"
    )

    print(
        f"Observations used: "
        f"{len(wind)}"
    )

    print(
        f"Minimum training fraction: "
        f"{MINIMUM_TRAIN_FRACTION:.0%}"
    )

    print(
        f"Validation windows: "
        f"{NUMBER_OF_WINDOWS}"
    )

    print(
        f"Test fraction/window: "
        f"{TEST_FRACTION:.0%}"
    )

    print(
        f"Lags: {LAGS}"
    )

    print()

    # ======================================================================
    # BUILD VALIDATION WINDOWS
    # ======================================================================

    windows = build_validation_windows(
        wind
    )

    print(
        "Validation windows:"
    )

    for (
        window_number,
        train_end,
        test_end,
    ) in windows:

        train_start_time = (
            wind.index[0]
        )

        train_end_time = (
            wind.index[train_end - 1]
        )

        test_start_time = (
            wind.index[train_end]
        )

        test_end_time = (
            wind.index[test_end - 1]
        )

        print(
            f"  Window {window_number}: "
            f"train={train_start_time} "
            f"-> {train_end_time} | "
            f"test={test_start_time} "
            f"-> {test_end_time}"
        )

    print()

    # ======================================================================
    # EVALUATE
    # ======================================================================

    results: list[
        dict[str, object]
    ] = []

    for (
        window_number,
        train_end,
        test_end,
    ) in windows:

        print(
            f"Evaluating validation "
            f"window {window_number}..."
        )

        for (
            horizon_name,
            horizon_steps,
        ) in HORIZONS.items():

            print(
                f"  Evaluating {horizon_name} "
                f"({horizon_steps} steps)...",
                end=" ",
            )

            result = evaluate_window(
                wind=wind,
                window_number=window_number,
                train_end=train_end,
                test_end=test_end,
                horizon_name=horizon_name,
                horizon_steps=horizon_steps,
            )

            results.append(
                result
            )

            print(
                f"NMAE="
                f"{result['nmae_percent']:.4f}%"
            )

    if not results:
        raise ValueError(
            "No validation results were generated."
        )

    results_dataframe = pd.DataFrame(
        results
    )

    summary = build_summary(
        results_dataframe
    )

    # ======================================================================
    # DETAILED RESULTS
    # ======================================================================

    print()
    print("=" * 80)
    print(
        "ROLLING VALIDATION RESULTS"
    )
    print("=" * 80)
    print()

    print(
        results_dataframe[
            [
                "window",
                "horizon",
                "samples",
                "mae_mw",
                "rmse_mw",
                "nmae_percent",
            ]
        ].to_string(
            index=False
        )
    )

    # ======================================================================
    # ROBUSTNESS SUMMARY
    # ======================================================================

    print()
    print("=" * 80)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 80)
    print()

    print(
        summary[
            [
                "horizon",
                "windows_evaluated",
                "mean_nmae_percent",
                "std_nmae_percent",
                "worst_nmae_percent",
                "robustness_score",
            ]
        ].to_string(
            index=False
        )
    )

    # ======================================================================
    # OVERALL ASSESSMENT
    # ======================================================================

    print()
    print("=" * 80)
    print(
        "MODEL VALIDATION ASSESSMENT"
    )
    print("=" * 80)
    print()

    overall_mean = float(
        summary[
            "mean_nmae_percent"
        ].mean()
    )

    overall_std = float(
        summary[
            "std_nmae_percent"
        ].mean()
    )

    overall_worst = float(
        summary[
            "worst_nmae_percent"
        ].max()
    )

    print(
        f"Mean NMAE across horizons: "
        f"{overall_mean:.4f}%"
    )

    print(
        f"Mean cross-window NMAE std: "
        f"{overall_std:.4f}%"
    )

    print(
        f"Worst observed horizon NMAE: "
        f"{overall_worst:.4f}%"
    )

    print()

    if overall_std < 2.0:

        print(
            "Assessment: Forecast performance "
            "appears relatively stable across "
            "validation windows."
        )

    elif overall_std < 5.0:

        print(
            "Assessment: Forecast performance "
            "shows moderate variation across "
            "validation windows."
        )

    else:

        print(
            "Assessment: Forecast performance "
            "shows substantial variation across "
            "validation windows."
        )

    # ======================================================================
    # SAVE RESULTS
    # ======================================================================

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "OUTPUT"
    )
    print("=" * 80)
    print()

    print(
        f"Saved detailed results: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_FILE}"
    )

    print()
    print(
        "Rolling validation complete."
    )


if __name__ == "__main__":
    main()