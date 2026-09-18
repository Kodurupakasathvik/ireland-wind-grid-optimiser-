"""
EirGrid Direct Multi-Horizon Linear Regression
rolling time-series validation.

Uses expanding-window chronological validation.

Forecast horizons:
    15 minutes
    30 minutes
    1 hour
    2 hours
    4 hours

The validation protocol intentionally matches the standard
Linear Regression rolling-validation experiment so that the
two models can be compared fairly.

Important:
    - No random shuffling.
    - No interpolation.
    - Missing wind observations are removed.
    - Future observations are never used during training.
    - Each forecast horizon has its own independently trained model.

Outputs:
    data/processed/
        direct_multi_horizon_rolling_validation.csv
        direct_multi_horizon_rolling_validation_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.wind_data_loader import load_wind_data

from scripts.forecasting.direct_multi_horizon_linear_regression import (
    DirectMultiHorizonLinearRegressionWindForecaster,
)

from scripts.forecasting.forecast_evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    normalized_mae_percent,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED
    / "direct_multi_horizon_rolling_validation.csv"
)

SUMMARY_FILE = (
    PROCESSED
    / "direct_multi_horizon_rolling_validation_summary.csv"
)

TRAIN_FRACTION = 0.60

VALIDATION_WINDOWS = 5

TEST_FRACTION = 0.05

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

HORIZONS = (
    1,
    2,
    4,
    8,
    16,
)

HORIZON_NAMES = {
    1: "15min",
    2: "30min",
    4: "1hour",
    8: "2hour",
    16: "4hour",
}


# ============================================================================
# VALIDATION
# ============================================================================


def validate_wind_series(
    wind: pd.Series,
) -> pd.Series:
    """
    Validate and clean the EirGrid wind-generation series.

    Missing observations are removed rather than interpolated.

    This deliberately matches the existing rolling-validation
    methodology used for Standard Linear Regression.
    """

    if not isinstance(wind, pd.Series):
        raise TypeError(
            "wind must be a pandas Series."
        )

    if wind.empty:
        raise ValueError(
            "Wind dataset is empty."
        )

    wind = wind.copy()

    wind.index = pd.DatetimeIndex(
        wind.index
    )

    if not wind.index.is_monotonic_increasing:
        raise ValueError(
            "Wind timestamps must be sorted "
            "in chronological order."
        )

    if wind.index.has_duplicates:
        raise ValueError(
            "Wind timestamps must not contain "
            "duplicate observations."
        )

    wind = pd.to_numeric(
        wind,
        errors="coerce",
    )

    missing_count = int(
        wind.isna().sum()
    )

    if missing_count > 0:

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

        wind = wind.dropna()

    if wind.empty:
        raise ValueError(
            "No observations remain after "
            "removing missing values."
        )

    negative_count = int(
        (wind < 0).sum()
    )

    if negative_count > 0:
        raise ValueError(
            "Wind dataset contains negative "
            "generation values."
        )

    return wind


# ============================================================================
# VALIDATION WINDOWS
# ============================================================================


def build_validation_windows(
    wind: pd.Series,
) -> list[tuple[pd.Series, pd.Series]]:
    """
    Build expanding chronological validation windows.

    Each window contains:

        training data
        +
        immediately following test data

    The training set expands for every subsequent window.
    """

    n = len(wind)

    minimum_train = int(
        n * TRAIN_FRACTION
    )

    test_size = int(
        n * TEST_FRACTION
    )

    if test_size <= 0:
        raise ValueError(
            "Test window contains no observations."
        )

    windows = []

    for window_number in range(
        VALIDATION_WINDOWS
    ):

        train_end = (
            minimum_train
            + window_number * test_size
        )

        test_start = train_end

        test_end = (
            test_start
            + test_size
        )

        if test_end > n:
            break

        train = wind.iloc[
            :train_end
        ].copy()

        test = wind.iloc[
            test_start:test_end
        ].copy()

        if train.empty or test.empty:
            continue

        windows.append(
            (
                train,
                test,
            )
        )

    if not windows:
        raise ValueError(
            "No valid validation windows "
            "could be constructed."
        )

    return windows


# ============================================================================
# SINGLE-HORIZON EVALUATION
# ============================================================================


def evaluate_horizon(
    train: pd.Series,
    test: pd.Series,
    horizon_steps: int,
) -> dict[str, float | int]:
    """
    Evaluate one Direct Multi-Horizon forecast horizon.

    The model is fitted only on observations available
    before the test window.

    Forecast blocks are evaluated chronologically.

    After each block has been evaluated, the corresponding
    actual observations become available and are appended
    to the history.

    This creates an expanding walk-forward evaluation.
    """

    if horizon_steps not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon_steps}"
        )

    history = train.copy()

    actual_values: list[float] = []

    forecast_values: list[float] = []

    position = 0

    while position < len(test):

        remaining = len(test) - position

        if remaining < horizon_steps:
            break

        actual_block = test.iloc[
            position:
            position + horizon_steps
        ].copy()

        # --------------------------------------------------------------
        # Train a fresh direct multi-horizon model.
        #
        # The model contains independent estimators for each
        # configured horizon.
        # --------------------------------------------------------------

        model = (
            DirectMultiHorizonLinearRegressionWindForecaster(
                horizons=HORIZONS,
                lags=LAGS,
                rolling_windows=ROLLING_WINDOWS,
            )
        )

        model.fit(
            history
        )

        # --------------------------------------------------------------
        # Forecast the requested horizon.
        # --------------------------------------------------------------

        predictions = model.predict(
            history
        )

        if horizon_steps not in predictions:
            raise RuntimeError(
                "Requested horizon was not returned "
                f"by the forecaster: {horizon_steps}"
            )

        forecast_value = float(
            predictions[horizon_steps]
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Direct Multi-Horizon predict() returns ONE forecast
        # for each horizon, rather than a recursive sequence.
        #
        # For a fair direct-horizon validation, the requested
        # horizon prediction corresponds to the endpoint of
        # this forecast block.
        # --------------------------------------------------------------

        actual_target = float(
            actual_block.iloc[-1]
        )

        actual_values.append(
            actual_target
        )

        forecast_values.append(
            forecast_value
        )

        # --------------------------------------------------------------
        # Reveal the complete actual block only AFTER forecasting.
        # --------------------------------------------------------------

        history = pd.concat(
            [
                history,
                actual_block,
            ]
        )

        position += horizon_steps

    if not actual_values:
        raise ValueError(
            "No evaluation samples were produced "
            f"for horizon={horizon_steps}."
        )

    mae = mean_absolute_error(
        actual_values,
        forecast_values,
    )

    rmse = root_mean_squared_error(
        actual_values,
        forecast_values,
    )

    nmae_percent = normalized_mae_percent(
        actual_values,
        forecast_values,
    )

    return {
        "samples": len(actual_values),
        "mae_mw": float(mae),
        "rmse_mw": float(rmse),
        "nmae_percent": float(nmae_percent),
    }


# ============================================================================
# MAIN VALIDATION
# ============================================================================


def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID DIRECT MULTI-HORIZON LINEAR REGRESSION "
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

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    print(
        "Loading EirGrid wind data..."
    )

    dataframe = load_wind_data()

    wind_raw = dataframe[
        "wind_generation_mw"
    ].copy()

    original_observations = len(
        wind_raw
    )

    original_missing = int(
        wind_raw.isna().sum()
    )

    wind = validate_wind_series(
        wind_raw
    )

    print()

    print(
        f"Original observations: "
        f"{original_observations}"
    )

    print(
        f"Missing observations removed: "
        f"{original_missing}"
    )

    print(
        f"Observations used: "
        f"{len(wind)}"
    )

    print(
        f"Minimum training fraction: "
        f"{TRAIN_FRACTION:.0%}"
    )

    print(
        f"Validation windows: "
        f"{VALIDATION_WINDOWS}"
    )

    print(
        f"Test fraction/window: "
        f"{TEST_FRACTION:.0%}"
    )

    print(
        f"Lags: {LAGS}"
    )

    print(
        f"Rolling windows: "
        f"{ROLLING_WINDOWS}"
    )

    print()

    # ------------------------------------------------------------------------
    # Build windows
    # ------------------------------------------------------------------------

    windows = build_validation_windows(
        wind
    )

    print(
        "Validation windows:"
    )

    for index, (
        train,
        test,
    ) in enumerate(
        windows,
        start=1,
    ):

        print(
            f"  Window {index}: "
            f"train={train.index[0]} -> "
            f"{train.index[-1]} | "
            f"test={test.index[0]} -> "
            f"{test.index[-1]}"
        )

    print()

    # ------------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------------

    results: list[dict[str, object]] = []

    for window_number, (
        train,
        test,
    ) in enumerate(
        windows,
        start=1,
    ):

        print(
            f"Evaluating validation window "
            f"{window_number}..."
        )

        for horizon_steps in HORIZONS:

            horizon_name = HORIZON_NAMES[
                horizon_steps
            ]

            print(
                f"  Evaluating {horizon_name} "
                f"({horizon_steps} steps)...",
                end=" ",
            )

            metrics = evaluate_horizon(
                train=train,
                test=test,
                horizon_steps=horizon_steps,
            )

            print(
                f"NMAE="
                f"{metrics['nmae_percent']:.4f}%"
            )

            results.append(
                {
                    "window": window_number,
                    "horizon": horizon_name,
                    "horizon_steps": horizon_steps,
                    "samples": metrics[
                        "samples"
                    ],
                    "mae_mw": metrics[
                        "mae_mw"
                    ],
                    "rmse_mw": metrics[
                        "rmse_mw"
                    ],
                    "nmae_percent": metrics[
                        "nmae_percent"
                    ],
                }
            )

    results_dataframe = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------------------
    # Detailed results
    # ------------------------------------------------------------------------

    print()

    print("=" * 80)
    print(
        "ROLLING VALIDATION RESULTS"
    )
    print("=" * 80)
    print()

    print(
        results_dataframe.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Robustness summary
    # ------------------------------------------------------------------------

    summary_rows: list[dict[str, object]] = []

    for horizon_steps in HORIZONS:

        horizon_name = HORIZON_NAMES[
            horizon_steps
        ]

        subset = results_dataframe[
            results_dataframe[
                "horizon_steps"
            ]
            == horizon_steps
        ]

        values = subset[
            "nmae_percent"
        ].astype(float)

        mean_nmae = float(
            values.mean()
        )

        std_nmae = float(
            values.std(
                ddof=0
            )
        )

        worst_nmae = float(
            values.max()
        )

        robustness_score = (
            mean_nmae
            + std_nmae
        )

        summary_rows.append(
            {
                "horizon": horizon_name,
                "windows_evaluated": len(
                    values
                ),
                "mean_nmae_percent": mean_nmae,
                "std_nmae_percent": std_nmae,
                "worst_nmae_percent": worst_nmae,
                "robustness_score": robustness_score,
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    # ------------------------------------------------------------------------
    # Robustness display
    # ------------------------------------------------------------------------

    print()

    print("=" * 80)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 80)
    print()

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Overall assessment
    # ------------------------------------------------------------------------

    mean_nmae_overall = float(
        summary_dataframe[
            "mean_nmae_percent"
        ].mean()
    )

    mean_std_overall = float(
        summary_dataframe[
            "std_nmae_percent"
        ].mean()
    )

    worst_observed = float(
        summary_dataframe[
            "worst_nmae_percent"
        ].max()
    )

    print()

    print("=" * 80)
    print(
        "MODEL VALIDATION ASSESSMENT"
    )
    print("=" * 80)
    print()

    print(
        f"Mean NMAE across horizons: "
        f"{mean_nmae_overall:.4f}%"
    )

    print(
        f"Mean cross-window NMAE std: "
        f"{mean_std_overall:.4f}%"
    )

    print(
        f"Worst observed horizon NMAE: "
        f"{worst_observed:.4f}%"
    )

    print()

    if mean_std_overall <= 2.0:

        print(
            "Assessment: Forecast performance "
            "appears relatively stable across "
            "validation windows."
        )

    elif mean_std_overall <= 4.0:

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

    # ------------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------------

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary_dataframe.to_csv(
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
        "Direct Multi-Horizon rolling "
        "validation complete."
    )


if __name__ == "__main__":
    main()