"""
EirGrid Feature-Engineered Linear Regression
Direct-Horizon Rolling Time-Series Validation.

IMPORTANT DESIGN:
    X(t) -> y(t + horizon)

Each horizon has its own Linear Regression model.

Predictions are NEVER fed back into history.

The authoritative EirGrid wind loader is used as the source of truth.
Missing observations are never interpolated or fabricated.

Outputs:
    data/processed/
        feature_engineered_linear_regression_rolling_validation.csv
        feature_engineered_linear_regression_rolling_validation_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.data.wind_data_loader import load_wind_data


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.05
VALIDATION_WINDOWS = 5

WIND_COLUMN = "wind_generation_mw"

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "1hour": 4,
    "2hour": 8,
    "4hour": 16,
}

DEFAULT_LAGS = (
    1,
    2,
    4,
    8,
    16,
    32,
    96,
)

DEFAULT_ROLLING_WINDOWS = (
    4,
    8,
    16,
    32,
    96,
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
)

DETAILED_OUTPUT = (
    OUTPUT_DIR
    / "feature_engineered_linear_regression_rolling_validation.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "feature_engineered_linear_regression_rolling_validation_summary.csv"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_authoritative_wind_data() -> pd.Series:
    """
    Load authoritative EirGrid wind-generation observations.

    The loader creates the complete 15-minute timeline but does NOT
    fabricate missing wind measurements.

    Returns
    -------
    pandas.Series
        Wind generation in MW indexed by 15-minute timestamps.
    """

    dataframe = load_wind_data()

    if WIND_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required wind column '{WIND_COLUMN}' "
            "was not found."
        )

    wind = dataframe[WIND_COLUMN].copy()

    wind.index = pd.DatetimeIndex(wind.index)

    wind = pd.to_numeric(
        wind,
        errors="coerce",
    ).astype(float)

    wind = wind.sort_index()

    if wind.index.has_duplicates:
        raise ValueError(
            "Authoritative EirGrid wind data contains "
            "duplicate timestamps."
        )

    if not wind.index.is_monotonic_increasing:
        raise ValueError(
            "Authoritative EirGrid wind timestamps are not "
            "chronologically ordered."
        )

    if (wind.dropna() < 0).any():
        raise ValueError(
            "Authoritative EirGrid wind generation contains "
            "negative observations."
        )

    missing = wind[wind.isna()]

    print(
        f"Observations on authoritative timeline: {len(wind)}"
    )

    print(
        f"Timestamp range: {wind.index.min()} -> "
        f"{wind.index.max()}"
    )

    print(
        f"Wind column: {WIND_COLUMN}"
    )

    if len(missing) > 0:

        print(
            f"\nWARNING: Authoritative EirGrid wind generation "
            f"contains {len(missing)} missing observations."
        )

        print(
            "Missing observations will NOT be interpolated, "
            "forward-filled, or otherwise fabricated."
        )

        print("Missing timestamps:")

        for timestamp in missing.index:
            print(f"  {timestamp}")

    else:

        print(
            "\nNo missing wind-generation observations detected."
        )

    # -------------------------------------------------------------------------
    # Validate timeline
    # -------------------------------------------------------------------------

    differences = (
        wind.index.to_series()
        .diff()
        .dropna()
    )

    irregular = differences[
        differences != pd.Timedelta(minutes=15)
    ]

    if len(irregular) > 0:

        print(
            f"\nWARNING: Detected {len(irregular)} "
            "irregular timestamp gaps."
        )

    else:

        print(
            "Timestamp grid: continuous 15-minute timeline"
        )

    return wind


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_feature_row(
    history: pd.Series,
    timestamp: pd.Timestamp,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.Series | None:
    """
    Construct X(t) using observations available at or before t.

    IMPORTANT:
        No future observation is used.

    Parameters
    ----------
    history:
        Authoritative wind series.

    timestamp:
        Forecast origin t.

    Returns
    -------
    pandas.Series or None
        Feature row, or None if required historical observations
        are unavailable.
    """

    if timestamp not in history.index:
        return None

    # -------------------------------------------------------------------------
    # Current observation
    # -------------------------------------------------------------------------

    current = history.loc[timestamp]

    if pd.isna(current):
        return None

    features: dict[str, float] = {}

    features["wind_current_mw"] = float(current)

    # -------------------------------------------------------------------------
    # Lag features
    # -------------------------------------------------------------------------

    for lag in lags:

        lag_timestamp = (
            timestamp
            - pd.Timedelta(
                minutes=15 * lag
            )
        )

        if lag_timestamp not in history.index:
            return None

        value = history.loc[lag_timestamp]

        if pd.isna(value):
            return None

        features[
            f"wind_lag_{lag}_steps_mw"
        ] = float(value)

    # -------------------------------------------------------------------------
    # Rolling historical features
    #
    # IMPORTANT:
    #     shift(1)
    #
    # Therefore the rolling statistics do not include
    # the current observation.
    # -------------------------------------------------------------------------

    for window in rolling_windows:

        values = []

        for step in range(
            1,
            window + 1,
        ):

            historical_timestamp = (
                timestamp
                - pd.Timedelta(
                    minutes=15 * step
                )
            )

            if historical_timestamp not in history.index:
                return None

            value = history.loc[
                historical_timestamp
            ]

            if pd.isna(value):
                return None

            values.append(
                float(value)
            )

        values_array = np.asarray(
            values,
            dtype=float,
        )

        if not np.isfinite(
            values_array
        ).all():

            return None

        features[
            f"wind_rolling_{window}_steps_mean_mw"
        ] = float(
            np.mean(values_array)
        )

        # pandas rolling/std uses sample standard deviation (ddof=1)
        if len(values_array) > 1:

            features[
                f"wind_rolling_{window}_steps_std_mw"
            ] = float(
                np.std(
                    values_array,
                    ddof=1,
                )
            )

        else:

            features[
                f"wind_rolling_{window}_steps_std_mw"
            ] = 0.0

    # -------------------------------------------------------------------------
    # Time-of-day cyclic features
    # -------------------------------------------------------------------------

    minutes_since_midnight = (
        timestamp.hour * 60
        + timestamp.minute
    )

    day_fraction = (
        minutes_since_midnight
        / (24 * 60)
    )

    features["time_sin"] = float(
        np.sin(
            2 * np.pi * day_fraction
        )
    )

    features["time_cos"] = float(
        np.cos(
            2 * np.pi * day_fraction
        )
    )

    # -------------------------------------------------------------------------
    # Day-of-year cyclic features
    # -------------------------------------------------------------------------

    day_of_year = (
        timestamp.dayofyear - 1
    )

    year_length = (
        366
        if timestamp.is_leap_year
        else 365
    )

    year_fraction = (
        day_of_year
        / year_length
    )

    features["day_of_year_sin"] = float(
        np.sin(
            2 * np.pi * year_fraction
        )
    )

    features["day_of_year_cos"] = float(
        np.cos(
            2 * np.pi * year_fraction
        )
    )

    result = pd.Series(
        features,
        dtype=float,
    )

    if not np.isfinite(
        result.to_numpy()
    ).all():

        return None

    return result


# =============================================================================
# BUILD DIRECT TRAINING DATA
# =============================================================================

def build_direct_training_data(
    wind: pd.Series,
    train_end: pd.Timestamp,
    horizon_steps: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct direct-horizon training samples.

    Each sample is:

        X(t) -> y(t + horizon)

    Training targets must remain inside the training period.

    No validation-period observations are used.

    Returns
    -------
    X, y
    """

    train_wind = wind.loc[
        wind.index <= train_end
    ].copy()

    valid_wind = train_wind.dropna()

    if valid_wind.empty:
        raise ValueError(
            "Training period contains no valid wind observations."
        )

    X_rows: list[pd.Series] = []
    y_values: list[float] = []

    timestamps = train_wind.index

    horizon_delta = pd.Timedelta(
        minutes=15 * horizon_steps
    )

    for timestamp in timestamps:

        target_timestamp = (
            timestamp
            + horizon_delta
        )

        # Target must remain inside training data.
        if target_timestamp > train_end:
            continue

        # Target must exist in authoritative timeline.
        if target_timestamp not in train_wind.index:
            continue

        target = train_wind.loc[
            target_timestamp
        ]

        if pd.isna(target):
            continue

        features = build_feature_row(
            history=train_wind,
            timestamp=timestamp,
        )

        if features is None:
            continue

        X_rows.append(features)
        y_values.append(float(target))

    if not X_rows:

        raise ValueError(
            "No valid direct-horizon training samples were "
            f"created for horizon={horizon_steps}."
        )

    X = pd.DataFrame(
        X_rows
    )

    y = pd.Series(
        y_values,
        dtype=float,
    )

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Training features contain NaN or infinite values."
        )

    if not np.isfinite(
        y.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Training targets contain NaN or infinite values."
        )

    return X, y


# =============================================================================
# VALIDATION SAMPLE GENERATION
# =============================================================================

def build_validation_samples(
    wind: pd.Series,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon_steps: int,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    """
    Construct direct validation samples.

    Each validation sample uses:

        X(t) -> actual y(t+h)

    Predictions are never inserted into history.

    Only validation origins inside the requested test period are used.

    Returns
    -------
    X
    y
    skipped_missing_origins
    skipped_insufficient_history
    """

    X_rows: list[pd.Series] = []
    y_values: list[float] = []

    skipped_missing_origins = 0
    skipped_insufficient_history = 0

    horizon_delta = pd.Timedelta(
        minutes=15 * horizon_steps
    )

    origins = wind.loc[
        (wind.index >= test_start)
        & (wind.index <= test_end)
    ].index

    for timestamp in origins:

        # ---------------------------------------------------------------------
        # Forecast origin must have an actual observation.
        # ---------------------------------------------------------------------

        origin_value = wind.loc[timestamp]

        if pd.isna(origin_value):

            skipped_missing_origins += 1
            continue

        # ---------------------------------------------------------------------
        # Target timestamp.
        # ---------------------------------------------------------------------

        target_timestamp = (
            timestamp
            + horizon_delta
        )

        if target_timestamp not in wind.index:

            skipped_insufficient_history += 1
            continue

        actual = wind.loc[
            target_timestamp
        ]

        if pd.isna(actual):

            skipped_missing_origins += 1
            continue

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Feature construction uses ONLY observations at or before t.
        #
        # This prevents future leakage.
        # ---------------------------------------------------------------------

        features = build_feature_row(
            history=wind,
            timestamp=timestamp,
        )

        if features is None:

            skipped_insufficient_history += 1
            continue

        X_rows.append(features)
        y_values.append(float(actual))

    if not X_rows:

        raise ValueError(
            "No valid validation samples were produced for "
            f"horizon={horizon_steps}. "
            "Check the validation boundaries and feature history."
        )

    X = pd.DataFrame(
        X_rows
    )

    y = pd.Series(
        y_values,
        dtype=float,
    )

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Validation features contain NaN or infinite values."
        )

    if not np.isfinite(
        y.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Validation targets contain NaN or infinite values."
        )

    return (
        X,
        y,
        skipped_missing_origins,
        skipped_insufficient_history,
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_mae(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    if len(actual_array) == 0:
        raise ValueError(
            "Cannot calculate MAE with zero samples."
        )

    return float(
        np.mean(
            np.abs(
                actual_array
                - predicted_array
            )
        )
    )


def calculate_rmse(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    if len(actual_array) == 0:
        raise ValueError(
            "Cannot calculate RMSE with zero samples."
        )

    return float(
        np.sqrt(
            np.mean(
                (
                    actual_array
                    - predicted_array
                ) ** 2
            )
        )
    )


def calculate_nmae(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """
    Normalised MAE as:

        NMAE = MAE / mean(actual) * 100

    Wind-generation mean is used as the normalisation base.
    """

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    if len(actual_array) == 0:
        raise ValueError(
            "Cannot calculate NMAE with zero samples."
        )

    denominator = float(
        np.mean(actual_array)
    )

    if denominator <= 0:

        raise ValueError(
            "Cannot calculate NMAE because mean actual "
            "wind generation is not positive."
        )

    mae = calculate_mae(
        actual_array,
        predicted_array,
    )

    return float(
        mae
        / denominator
        * 100.0
    )


# =============================================================================
# VALIDATION WINDOW GENERATION
# =============================================================================

def build_validation_windows(
    wind: pd.Series,
) -> list[
    tuple[
        int,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:
    """
    Build expanding-window rolling validation splits.

    Minimum training fraction:
        60%

    Each validation window:
        5% of total timeline

    Windows expand chronologically.
    """

    n = len(wind)

    minimum_train_size = int(
        np.floor(
            n * TRAIN_FRACTION
        )
    )

    test_size = int(
        np.floor(
            n * TEST_FRACTION
        )
    )

    if test_size < 1:
        raise ValueError(
            "Calculated test window contains zero observations."
        )

    windows = []

    for window_number in range(
        1,
        VALIDATION_WINDOWS + 1,
    ):

        train_end_position = (
            minimum_train_size
            + (
                window_number - 1
            )
            * test_size
            - 1
        )

        test_start_position = (
            train_end_position + 1
        )

        test_end_position = (
            test_start_position
            + test_size
            - 1
        )

        if test_end_position >= n:

            break

        train_start = wind.index[0]

        train_end = wind.index[
            train_end_position
        ]

        test_start = wind.index[
            test_start_position
        ]

        test_end = wind.index[
            test_end_position
        ]

        windows.append(
            (
                window_number,
                train_start,
                train_end,
                test_start,
                test_end,
            )
        )

    if len(windows) < VALIDATION_WINDOWS:

        raise ValueError(
            f"Only {len(windows)} validation windows could be "
            f"constructed; required {VALIDATION_WINDOWS}."
        )

    return windows


# =============================================================================
# DIRECT HORIZON EVALUATION
# =============================================================================

def evaluate_direct_horizon(
    wind: pd.Series,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon_name: str,
    horizon_steps: int,
) -> dict[str, object]:
    """
    Train and evaluate one direct horizon model.

    Architecture:

        historical X(t)
              |
              v
        Linear Regression
              |
              v
        y(t + horizon)

    No recursive prediction.
    """

    X_train, y_train = build_direct_training_data(
        wind=wind,
        train_end=train_end,
        horizon_steps=horizon_steps,
    )

    X_test, y_test, skipped_missing, skipped_history = (
        build_validation_samples(
            wind=wind,
            test_start=test_start,
            test_end=test_end,
            horizon_steps=horizon_steps,
        )
    )

    # -------------------------------------------------------------------------
    # Match feature order exactly.
    # -------------------------------------------------------------------------

    X_test = X_test[
        X_train.columns
    ]

    # -------------------------------------------------------------------------
    # Train model.
    # -------------------------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X_train.to_numpy(
            dtype=float
        ),
        y_train.to_numpy(
            dtype=float
        ),
    )

    # -------------------------------------------------------------------------
    # Predict.
    #
    # Predictions are independent direct forecasts.
    # They are NOT fed back into history.
    # -------------------------------------------------------------------------

    predictions = model.predict(
        X_test.to_numpy(
            dtype=float
        )
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    # Wind generation cannot physically be negative.
    predictions = np.maximum(
        predictions,
        0.0,
    )

    actual = y_test.to_numpy(
        dtype=float
    )

    mae = calculate_mae(
        actual,
        predictions,
    )

    rmse = calculate_rmse(
        actual,
        predictions,
    )

    nmae = calculate_nmae(
        actual,
        predictions,
    )

    return {
        "horizon": horizon_name,
        "horizon_steps": horizon_steps,
        "samples": len(actual),
        "mae_mw": mae,
        "rmse_mw": rmse,
        "nmae_percent": nmae,
        "skipped_missing_origins": (
            skipped_missing
        ),
        "skipped_insufficient_history": (
            skipped_history
        ),
        "training_samples": len(y_train),
    }


# =============================================================================
# ROBUSTNESS SUMMARY
# =============================================================================

def build_robustness_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for horizon, group in results.groupby(
        "horizon",
        sort=False,
    ):

        values = group[
            "nmae_percent"
        ].to_numpy(
            dtype=float
        )

        mean_nmae = float(
            np.mean(values)
        )

        std_nmae = float(
            np.std(
                values,
                ddof=0,
            )
        )

        worst_nmae = float(
            np.max(values)
        )

        robustness_score = (
            mean_nmae
            + std_nmae
            + worst_nmae
        )

        rows.append(
            {
                "horizon": horizon,
                "windows_evaluated": len(values),
                "mean_nmae_percent": mean_nmae,
                "std_nmae_percent": std_nmae,
                "worst_nmae_percent": worst_nmae,
                "robustness_score": robustness_score,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID FEATURE-ENGINEERED LINEAR REGRESSION "
        "DIRECT-HORIZON ROLLING TIME-SERIES VALIDATION"
    )
    print("=" * 80)

    print(
        "Primary metric: NMAE (%)"
    )

    print(
        "Lower NMAE is better."
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    print(
        "\nLoading authoritative EirGrid wind data..."
    )

    wind = load_authoritative_wind_data()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    print(
        f"\nMinimum training fraction: "
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
        "\nForecast architecture:"
    )

    print(
        "  DIRECT horizon-specific models"
    )

    print(
        "  Predictions are NEVER fed back into history"
    )

    # -------------------------------------------------------------------------
    # Validation windows
    # -------------------------------------------------------------------------

    windows = build_validation_windows(
        wind
    )

    print(
        "\nValidation windows:"
    )

    for (
        window_number,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in windows:

        print(
            f"  Window {window_number}: "
            f"train={train_start} -> {train_end} | "
            f"test={test_start} -> {test_end}"
        )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    all_results: list[dict[str, object]] = []

    for (
        window_number,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in windows:

        print(
            f"\nEvaluating validation window "
            f"{window_number}..."
        )

        for (
            horizon_name,
            horizon_steps,
        ) in HORIZONS.items():

            print(
                f"  Evaluating {horizon_name} "
                f"({horizon_steps} steps)... ",
                end="",
                flush=True,
            )

            result = evaluate_direct_horizon(
                wind=wind,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                horizon_name=horizon_name,
                horizon_steps=horizon_steps,
            )

            result[
                "window"
            ] = window_number

            result[
                "train_start"
            ] = train_start

            result[
                "train_end"
            ] = train_end

            result[
                "test_start"
            ] = test_start

            result[
                "test_end"
            ] = test_end

            all_results.append(
                result
            )

            print(
                f"NMAE="
                f"{result['nmae_percent']:.4f}% "
                f"(samples="
                f"{result['samples']}, "
                f"skipped_missing="
                f"{result['skipped_missing_origins']}, "
                f"skipped_history="
                f"{result['skipped_insufficient_history']})"
            )

    # -------------------------------------------------------------------------
    # Results dataframe
    # -------------------------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results = results[
        [
            "window",
            "horizon",
            "horizon_steps",
            "samples",
            "mae_mw",
            "rmse_mw",
            "nmae_percent",
            "skipped_missing_origins",
            "skipped_insufficient_history",
            "training_samples",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        ]
    ]

    # -------------------------------------------------------------------------
    # Print detailed results
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "ROLLING VALIDATION RESULTS"
    )
    print("=" * 80)

    display_columns = [
        "window",
        "horizon",
        "horizon_steps",
        "samples",
        "mae_mw",
        "rmse_mw",
        "nmae_percent",
        "skipped_missing_origins",
        "skipped_insufficient_history",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Robustness summary
    # -------------------------------------------------------------------------

    summary = build_robustness_summary(
        results
    )

    print("\n")
    print("=" * 80)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Overall assessment
    # -------------------------------------------------------------------------

    mean_nmae = float(
        results[
            "nmae_percent"
        ].mean()
    )

    mean_std = float(
        summary[
            "std_nmae_percent"
        ].mean()
    )

    worst_nmae = float(
        results[
            "nmae_percent"
        ].max()
    )

    print("\n")
    print("=" * 80)
    print(
        "MODEL VALIDATION ASSESSMENT"
    )
    print("=" * 80)

    print(
        f"Mean NMAE across horizons: "
        f"{mean_nmae:.4f}%"
    )

    print(
        f"Mean cross-window NMAE std: "
        f"{mean_std:.4f}%"
    )

    print(
        f"Worst observed horizon NMAE: "
        f"{worst_nmae:.4f}%"
    )

    if mean_std <= 2.0:

        assessment = (
            "Forecast performance shows low "
            "cross-window variation."
        )

    elif mean_std <= 5.0:

        assessment = (
            "Forecast performance shows moderate "
            "cross-window variation."
        )

    else:

        assessment = (
            "Forecast performance shows substantial "
            "cross-window variation."
        )

    print(
        f"\nAssessment: {assessment}"
    )

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Save detailed results
    # -------------------------------------------------------------------------

    results.to_csv(
        DETAILED_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "OUTPUT"
    )
    print("=" * 80)

    print(
        f"Saved detailed results: "
        f"{DETAILED_OUTPUT}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_OUTPUT}"
    )

    print(
        "\nFeature-engineered Linear Regression "
        "direct-horizon rolling validation complete."
    )


if __name__ == "__main__":
    main()