"""
EirGrid persistence forecast rolling time-series validation.

Purpose
-------
Evaluate the persistence wind-power forecasting baseline using
rolling-origin time-series validation.

Primary metric:
    NMAE (%)

Secondary metrics:
    MAE (MW)
    RMSE (MW)
    cross-window standard deviation
    worst-case NMAE

Important methodological rules
------------------------------
1. No random train/test splitting.
2. No interpolation of missing observations.
3. Missing observations are removed.
4. Training data always precedes validation data.
5. Each validation window is evaluated independently.
6. Persistence forecast:
       forecast(t + h) = actual(t)
7. Forecasts are clipped to non-negative MW.
8. Uses the shared forecast_evaluation.py metric utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .forecast_evaluation import forecast_metrics


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "persistence_rolling_validation.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "persistence_rolling_validation_summary.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "datetime",
    "date_time",
    "date",
    "time",
    "Time",
    "Timestamp",
    "DateTime",
]

WIND_COLUMN_CANDIDATES = [
    "wind_generation_mw",
    "wind_generation",
    "wind_power_mw",
    "wind_power",
    "wind_mw",
    "wind",
    "Wind Generation (MW)",
    "Wind Generation",
    "Wind",
    "wind_generation_MW",
    "wind_power_MW",
]

EXPECTED_FREQUENCY = "15min"

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "1hour": 4,
    "2hour": 8,
    "4hour": 16,
}

MIN_TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.05
VALIDATION_WINDOWS = 5


# ============================================================================
# DATA DISCOVERY
# ============================================================================

def _normalise_column_name(name: str) -> str:
    """Normalise a column name for flexible matching."""

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _find_timestamp_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Find the most likely timestamp column."""

    normalised = {
        _normalise_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in TIMESTAMP_CANDIDATES:
        key = _normalise_column_name(candidate)

        if key in normalised:
            return normalised[key]

    # Fallback: inspect columns for timestamp-like names.
    for column in dataframe.columns:

        name = _normalise_column_name(column)

        if any(
            token in name
            for token in (
                "timestamp",
                "datetime",
                "date_time",
            )
        ):
            return column

    return None


def _find_wind_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Find the most likely wind-generation column."""

    normalised = {
        _normalise_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in WIND_COLUMN_CANDIDATES:
        key = _normalise_column_name(candidate)

        if key in normalised:
            return normalised[key]

    # Fallback based on semantic names.
    candidates: list[str] = []

    for column in dataframe.columns:

        name = _normalise_column_name(column)

        has_wind = "wind" in name
        has_power = any(
            token in name
            for token in (
                "generation",
                "power",
                "mw",
            )
        )

        if has_wind and has_power:
            candidates.append(column)

    if candidates:
        return candidates[0]

    return None


def _inspect_csv(
    path: Path,
) -> tuple[str | None, str | None]:
    """Inspect a CSV and return timestamp/wind columns if suitable."""

    try:
        dataframe = pd.read_csv(
            path,
            nrows=100,
        )
    except Exception:
        return None, None

    timestamp_column = _find_timestamp_column(
        dataframe
    )

    wind_column = _find_wind_column(
        dataframe
    )

    return timestamp_column, wind_column


def find_wind_data() -> Path:
    """
    Locate the EirGrid wind dataset.

    Searches data/raw and data/processed recursively.
    """

    search_roots = [
        RAW_DIR,
        PROCESSED_DIR,
    ]

    candidates: list[Path] = []

    for root in search_roots:

        if not root.exists():
            continue

        candidates.extend(
            root.rglob("*.csv")
        )

    # First pass: identify files containing both
    # timestamp and wind columns.
    valid_candidates: list[Path] = []

    for path in candidates:

        timestamp_column, wind_column = _inspect_csv(
            path
        )

        if (
            timestamp_column is not None
            and wind_column is not None
        ):
            valid_candidates.append(path)

    if not valid_candidates:

        raise FileNotFoundError(
            "Could not locate the EirGrid wind dataset.\n\n"
            "Searched recursively under:\n"
            f"  {RAW_DIR}\n"
            f"  {PROCESSED_DIR}\n\n"
            "The CSV must contain a timestamp/datetime "
            "column and a wind-generation/power column."
        )

    # Prefer files with EirGrid/wind terminology.
    def score(path: Path) -> int:

        name = path.name.lower()

        score_value = 0

        if "eirgrid" in name:
            score_value += 100

        if "wind" in name:
            score_value += 50

        if "generation" in name:
            score_value += 25

        if "forecast" in name:
            score_value -= 20

        return score_value

    valid_candidates.sort(
        key=score,
        reverse=True,
    )

    selected = valid_candidates[0]

    print(
        f"Selected wind dataset: {selected}"
    )

    if len(valid_candidates) > 1:

        print(
            "Other compatible CSV files detected:"
        )

        for path in valid_candidates[1:5]:
            print(
                f"  - {path}"
            )

    return selected


# ============================================================================
# DATA LOADING
# ============================================================================

def load_wind_series(
    path: Path,
) -> pd.Series:
    """
    Load and validate the EirGrid wind series.

    Missing observations are removed.
    No interpolation or synthetic values are created.
    """

    dataframe = pd.read_csv(path)

    timestamp_column = _find_timestamp_column(
        dataframe
    )

    wind_column = _find_wind_column(
        dataframe
    )

    if timestamp_column is None:

        raise ValueError(
            f"{path.name} does not contain a recognised "
            "timestamp/datetime column."
        )

    if wind_column is None:

        raise ValueError(
            f"{path.name} does not contain a recognised "
            "wind-generation/power column."
        )

    print(
        f"Timestamp column: {timestamp_column}"
    )

    print(
        f"Wind column: {wind_column}"
    )

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    wind = pd.to_numeric(
        dataframe[wind_column],
        errors="coerce",
    )

    series = pd.Series(
        wind.to_numpy(dtype=float),
        index=timestamps,
        name="wind_mw",
    )

    # Remove invalid timestamps.
    invalid_timestamp = series.index.isna()

    if invalid_timestamp.any():

        print(
            "WARNING: "
            f"Found {int(invalid_timestamp.sum())} "
            "invalid timestamps."
        )

        series = series.loc[
            ~invalid_timestamp
        ]

    # Sort chronologically.
    series = series.sort_index()

    # Remove duplicate timestamps.
    duplicate_mask = series.index.duplicated(
        keep="first"
    )

    if duplicate_mask.any():

        print(
            "WARNING: "
            f"Found {int(duplicate_mask.sum())} "
            "duplicate timestamps."
        )

        series = series.loc[
            ~duplicate_mask
        ]

    # Wind values cannot be negative.
    negative_mask = series < 0

    if negative_mask.any():

        raise ValueError(
            "Wind dataset contains negative "
            "generation values."
        )

    missing_mask = series.isna()

    missing_count = int(
        missing_mask.sum()
    )

    original_count = len(series)

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

        series = series.loc[
            ~missing_mask
        ]

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
            f"{len(series)}"
        )

    else:

        print(
            f"Observations used: "
            f"{len(series)}"
        )

    if len(series) < 100:

        raise ValueError(
            "Too few valid wind observations "
            "for rolling validation."
        )

    return series


# ============================================================================
# FREQUENCY VALIDATION
# ============================================================================

def validate_frequency(
    series: pd.Series,
) -> None:
    """
    Validate the expected 15-minute data frequency.

    Missing timestamps are reported but not filled.
    """

    deltas = series.index.to_series().diff().dropna()

    expected_delta = pd.Timedelta(
        minutes=15
    )

    irregular = (
        deltas != expected_delta
    )

    irregular_count = int(
        irregular.sum()
    )

    if irregular_count > 0:

        print()
        print(
            "WARNING: Detected "
            f"{irregular_count} irregular timestamp gaps."
        )

        print(
            "No missing timestamps will be "
            "synthetically inserted."
        )

    else:

        print(
            "Frequency check: 15-minute regular series."
        )


# ============================================================================
# VALIDATION WINDOWS
# ============================================================================

def build_validation_windows(
    series: pd.Series,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Construct expanding-window validation periods.

    Training begins at the first observation.

    Each subsequent validation window contains approximately
    TEST_FRACTION of the available observations.
    """

    n = len(series)

    minimum_training_size = int(
        np.floor(
            n * MIN_TRAIN_FRACTION
        )
    )

    test_size = int(
        np.floor(
            n * TEST_FRACTION
        )
    )

    if test_size < 1:

        raise ValueError(
            "Validation test window is empty."
        )

    windows: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ] = []

    for window_number in range(
        VALIDATION_WINDOWS
    ):

        train_end_position = (
            minimum_training_size
            + window_number * test_size
        )

        test_start_position = (
            train_end_position
        )

        test_end_position = (
            test_start_position
            + test_size
        )

        if test_end_position > n:

            break

        train_end_timestamp = series.index[
            train_end_position - 1
        ]

        test_start_timestamp = series.index[
            test_start_position
        ]

        test_end_timestamp = series.index[
            test_end_position - 1
        ]

        windows.append(
            (
                train_end_timestamp,
                test_end_timestamp,
            )
        )

    return windows


# ============================================================================
# PERSISTENCE FORECAST
# ============================================================================

def persistence_forecast(
    history: np.ndarray,
    horizon_steps: int,
) -> float:
    """
    Persistence forecast.

    For every future horizon:

        forecast(t+h) = latest observed value at t

    Only the most recent observed value is used.
    """

    if len(history) == 0:

        raise ValueError(
            "History cannot be empty."
        )

    if horizon_steps < 1:

        raise ValueError(
            "horizon_steps must be >= 1."
        )

    value = float(
        history[-1]
    )

    return max(
        0.0,
        value,
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_window(
    series: pd.Series,
    train_end_timestamp: pd.Timestamp,
    test_end_timestamp: pd.Timestamp,
    horizon_name: str,
    horizon_steps: int,
    window_number: int,
) -> dict[str, object]:
    """
    Evaluate one persistence forecasting horizon
    inside one rolling validation window.
    """

    train = series.loc[
        series.index
        <= train_end_timestamp
    ]

    test = series.loc[
        (
            series.index
            > train_end_timestamp
        )
        & (
            series.index
            <= test_end_timestamp
        )
    ]

    if len(train) == 0:

        raise ValueError(
            "Training set is empty."
        )

    if len(test) < horizon_steps:

        raise ValueError(
            f"Window {window_number}, "
            f"{horizon_name}: insufficient "
            "test observations."
        )

    actual_values: list[float] = []
    forecast_values: list[float] = []

    # ------------------------------------------------------------------
    # Rolling-origin persistence evaluation.
    #
    # At each prediction origin:
    #
    #     forecast(t+h) = actual(t)
    #
    # The actual observation at t is known.
    # Future observations are never used to create the forecast.
    # ------------------------------------------------------------------

    test_values = test.to_numpy(
        dtype=float
    )

    train_values = train.to_numpy(
        dtype=float
    )

    combined = np.concatenate(
        [
            train_values,
            test_values,
        ]
    )

    train_length = len(train_values)
    total_length = len(combined)

    for origin_position in range(
        train_length,
        total_length - horizon_steps + 1
    ):

        latest_observation = float(
            combined[
                origin_position - 1
            ]
        )

        target_position = (
            origin_position
            + horizon_steps
            - 1
        )

        actual = float(
            combined[target_position]
        )

        forecast = max(
            0.0,
            latest_observation,
        )

        actual_values.append(
            actual
        )

        forecast_values.append(
            forecast
        )

    if not actual_values:

        raise ValueError(
            f"Window {window_number}, "
            f"{horizon_name}: no valid "
            "forecast/evaluation pairs."
        )

    metrics = forecast_metrics(
        actual_values,
        forecast_values,
    )

    return {
        "window": window_number,
        "horizon": horizon_name,
        "horizon_steps": horizon_steps,
        "samples": len(actual_values),
        "mae_mw": metrics["mae_mw"],
        "rmse_mw": metrics["rmse_mw"],
        "nmae_percent": metrics[
            "nmae_percent"
        ],
    }


# ============================================================================
# ROBUSTNESS SUMMARY
# ============================================================================

def build_robustness_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build cross-window robustness statistics."""

    rows: list[dict[str, object]] = []

    for horizon in HORIZONS:

        subset = results.loc[
            results["horizon"] == horizon
        ].copy()

        if subset.empty:
            continue

        mean_nmae = float(
            subset["nmae_percent"].mean()
        )

        std_nmae = float(
            subset["nmae_percent"].std(
                ddof=0
            )
        )

        worst_nmae = float(
            subset["nmae_percent"].max()
        )

        robustness_score = (
            mean_nmae
            + std_nmae
            + 0.25 * worst_nmae
        )

        rows.append(
            {
                "horizon": horizon,
                "windows_evaluated": len(
                    subset
                ),
                "mean_nmae_percent": mean_nmae,
                "std_nmae_percent": std_nmae,
                "worst_nmae_percent": worst_nmae,
                "robustness_score": robustness_score,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID PERSISTENCE FORECAST "
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

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    print(
        "Loading EirGrid wind data..."
    )

    data_path = find_wind_data()

    wind = load_wind_series(
        data_path
    )

    validate_frequency(
        wind
    )

    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    minimum_training_fraction = (
        MIN_TRAIN_FRACTION * 100
    )

    test_fraction = (
        TEST_FRACTION * 100
    )

    print(
        f"Minimum training fraction: "
        f"{minimum_training_fraction:.0f}%"
    )

    print(
        f"Validation windows: "
        f"{VALIDATION_WINDOWS}"
    )

    print(
        f"Test fraction/window: "
        f"{test_fraction:.0f}%"
    )

    print()

    windows = build_validation_windows(
        wind
    )

    if len(windows) < VALIDATION_WINDOWS:

        print(
            "WARNING: Only "
            f"{len(windows)} validation windows "
            "could be constructed."
        )

    print(
        "Validation windows:"
    )

    for index, (
        train_end,
        test_end,
    ) in enumerate(
        windows,
        start=1,
    ):

        test_start_position = wind.index.get_loc(
            train_end
        ) + 1

        test_start = wind.index[
            test_start_position
        ]

        print(
            f"  Window {index}: "
            f"train={wind.index[0]} "
            f"-> {train_end} | "
            f"test={test_start} "
            f"-> {test_end}"
        )

    print()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    all_results: list[
        dict[str, object]
    ] = []

    for window_number, (
        train_end,
        test_end,
    ) in enumerate(
        windows,
        start=1,
    ):

        print(
            f"Evaluating validation window "
            f"{window_number}..."
        )

        for horizon_name, horizon_steps in HORIZONS.items():

            print(
                f"  Evaluating {horizon_name} "
                f"({horizon_steps} steps)...",
                end=" ",
            )

            result = evaluate_window(
                series=wind,
                train_end_timestamp=train_end,
                test_end_timestamp=test_end,
                horizon_name=horizon_name,
                horizon_steps=horizon_steps,
                window_number=window_number,
            )

            all_results.append(
                result
            )

            print(
                f"NMAE="
                f"{result['nmae_percent']:.4f}%"
            )

    results = pd.DataFrame(
        all_results
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ROLLING VALIDATION RESULTS"
    )
    print("=" * 80)
    print()

    print(
        results.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}"
        )
    )

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    summary = build_robustness_summary(
        results
    )

    print()
    print("=" * 80)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 80)
    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}"
        )
    )

    # ------------------------------------------------------------------
    # Overall assessment
    # ------------------------------------------------------------------

    mean_nmae = float(
        results["nmae_percent"].mean()
    )

    mean_window_std = float(
        summary[
            "std_nmae_percent"
        ].mean()
    )

    worst_nmae = float(
        results["nmae_percent"].max()
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
        f"{mean_nmae:.4f}%"
    )

    print(
        f"Mean cross-window NMAE std: "
        f"{mean_window_std:.4f}%"
    )

    print(
        f"Worst observed horizon NMAE: "
        f"{worst_nmae:.4f}%"
    )

    if mean_window_std <= 2.0:

        print()
        print(
            "Assessment: Forecast performance "
            "appears relatively stable across "
            "validation windows."
        )

    else:

        print()
        print(
            "Assessment: Forecast performance "
            "shows meaningful variation across "
            "validation windows."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
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
        "Persistence rolling validation complete."
    )


if __name__ == "__main__":
    main()