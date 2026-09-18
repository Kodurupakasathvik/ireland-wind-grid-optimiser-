"""
EirGrid Production Wind Forecast
================================

Production forecast runner for the Ireland Wind Grid Optimiser.

Purpose
-------
Generate the production wind forecast using the horizon-specific
winning models selected by the rolling-validation pipeline.

Current production horizons
---------------------------
    15 minutes
    30 minutes
    1 hour
    2 hours
    4 hours

Horizon-specific production models
-----------------------------------
Read from:

    data/processed/forecast_model_selection.csv

The production runner does NOT hard-code a global model.

Expected production mapping
---------------------------
    15min  -> Direct Multi-Horizon LR
    30min  -> Linear Regression
    1hour  -> Linear Regression
    2hour  -> Linear Regression
    4hour  -> Linear Regression

Data handling
-------------
Missing or non-numeric wind observations are removed.

NO interpolation is performed.
NO synthetic observations are introduced.

Forecast horizons:
    1  -> 15 minutes
    2  -> 30 minutes
    4  -> 1 hour
    8  -> 2 hours
    16 -> 4 hours

Outputs
-------
    data/processed/production_wind_forecast.csv
    data/processed/production_wind_forecast_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data.wind_data_loader import load_wind_data

from scripts.forecasting.linear_regression_forecast import (
    LinearRegressionWindForecaster,
)

from scripts.forecasting.direct_multi_horizon_linear_regression import (
    DirectMultiHorizonLinearRegressionWindForecaster,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

HORIZON_SELECTION_FILE = (
    PROCESSED
    / "forecast_model_selection.csv"
)

OUTPUT_FILE = (
    PROCESSED
    / "production_wind_forecast.csv"
)

SUMMARY_FILE = (
    PROCESSED
    / "production_wind_forecast_summary.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

WIND_COLUMN = "wind_generation_mw"

LAGS = 4

HORIZONS = {
    "15min": 1,
    "30min": 2,
    "1hour": 4,
    "2hour": 8,
    "4hour": 16,
}


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_wind_series(
    wind: pd.Series,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Validate and clean the wind-generation series.

    Important production-data policy
    ---------------------------------
    Missing and non-numeric observations are removed.

    They are NOT interpolated.

    No synthetic observations are introduced.
    """

    if not isinstance(
        wind,
        pd.Series,
    ):
        raise TypeError(
            "Wind data must be provided as a pandas Series."
        )

    if wind.empty:
        raise ValueError(
            "Wind dataset cannot be empty."
        )

    if not isinstance(
        wind.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Wind series must use a DatetimeIndex."
        )

    if wind.index.has_duplicates:
        raise ValueError(
            "Wind dataset contains duplicate timestamps."
        )

    if not wind.index.is_monotonic_increasing:
        print(
            "WARNING: Wind timestamps are not chronologically sorted."
        )

        print(
            "Sorting timestamps before validation."
        )

        wind = wind.sort_index()

    original_count = len(wind)

    numeric_wind = pd.to_numeric(
        wind,
        errors="coerce",
    )

    non_numeric_count = int(
        numeric_wind.isna().sum()
        - wind.isna().sum()
    )

    non_numeric_count = max(
        0,
        non_numeric_count,
    )

    missing_count = int(
        wind.isna().sum()
    )

    invalid_mask = (
        numeric_wind.isna()
    )

    removed_count = int(
        invalid_mask.sum()
    )

    if removed_count > 0:

        print()
        print(
            f"WARNING: Found {removed_count} "
            "missing/non-numeric wind observations."
        )

        print(
            "Removing invalid observations."
        )

        print(
            "No interpolation or synthetic values "
            "will be introduced."
        )

        numeric_wind = numeric_wind[
            ~invalid_mask
        ].copy()

    if numeric_wind.empty:
        raise ValueError(
            "No valid wind observations remain after "
            "removing missing/non-numeric values."
        )

    negative_count = int(
        (numeric_wind < 0).sum()
    )

    if negative_count > 0:

        raise ValueError(
            "Wind generation contains "
            f"{negative_count} negative observations. "
            "Negative generation is physically invalid."
        )

    numeric_wind = numeric_wind.astype(
        float
    )

    statistics = {
        "original_observations": original_count,
        "missing_observations": missing_count,
        "non_numeric_observations": non_numeric_count,
        "removed_invalid_observations": removed_count,
        "observations_used": len(numeric_wind),
    }

    return (
        numeric_wind,
        statistics,
    )


# =============================================================================
# HORIZON-SPECIFIC MODEL SELECTION
# =============================================================================

def load_horizon_model_map() -> dict[str, str]:
    """
    Load the horizon-specific best-model selection.

    Returns
    -------
    dict[str, str]
        Mapping from horizon label to winning model name.
    """

    if not HORIZON_SELECTION_FILE.exists():
        raise FileNotFoundError(
            "Horizon-specific model-selection file was not found:\n"
            f"  {HORIZON_SELECTION_FILE}\n\n"
            "Run the model-selection pipeline first:\n"
            "  python -m scripts.forecasting.forecast_model_selector"
        )

    dataframe = pd.read_csv(
        HORIZON_SELECTION_FILE
    )

    if dataframe.empty:
        raise ValueError(
            "Horizon-specific model-selection file is empty:\n"
            f"  {HORIZON_SELECTION_FILE}"
        )

    required_columns = {
        "horizon",
        "best_model",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Horizon-specific model-selection file is missing "
            f"required columns: {sorted(missing_columns)}"
        )

    dataframe["horizon"] = (
        dataframe["horizon"]
        .astype(str)
        .str.strip()
    )

    dataframe["best_model"] = (
        dataframe["best_model"]
        .astype(str)
        .str.strip()
    )

    return dict(
        zip(
            dataframe["horizon"],
            dataframe["best_model"],
        )
    )


# =============================================================================
# FORECAST GENERATION
# =============================================================================

def generate_production_forecast(
    wind: pd.Series,
    horizon_model_map: dict[str, str],
) -> pd.DataFrame:
    """
    Generate production forecasts using the winning model
    for each horizon.

    Parameters
    ----------
    wind:
        Cleaned historical wind-generation series.

    horizon_model_map:
        Mapping from horizon label to model name, loaded from
        forecast_model_selection.csv.

    Returns
    -------
    pd.DataFrame
        Forecast table with one row per horizon.
    """

    if wind.empty:
        raise ValueError(
            "Cannot generate a forecast from an empty wind series."
        )

    forecast_origin = wind.index[-1]

    rows: list[dict[str, Any]] = []

    for horizon_name, horizon_steps in HORIZONS.items():

        print(
            f"Generating {horizon_name} forecast "
            f"({horizon_steps} step(s))..."
        )

        model_name = horizon_model_map.get(horizon_name)

        if not model_name:
            raise ValueError(
                f"No winning model found for horizon "
                f"'{horizon_name}' in:\n"
                f"  {HORIZON_SELECTION_FILE}"
            )

        print(
            f"  Horizon-specific model: {model_name}"
        )

        if model_name == "Direct Multi-Horizon LR":

            model = DirectMultiHorizonLinearRegressionWindForecaster(
                horizons=(horizon_steps,),
            )

            model.fit(wind)

            forecast_value = model.predict_horizon(
                wind,
                horizon_steps,
            )

        elif model_name == "Linear Regression":

            model = LinearRegressionWindForecaster(
                lags=LAGS,
                horizon=horizon_steps,
            )

            model.fit(wind)

            forecasts = model.predict(wind)

            if len(forecasts) != horizon_steps:
                raise RuntimeError(
                    f"Expected {horizon_steps} forecasts for "
                    f"{horizon_name}, but received "
                    f"{len(forecasts)}."
                )

            forecast_value = float(forecasts[-1])

        else:
            raise ValueError(
                f"Unsupported horizon-specific model "
                f"'{model_name}' for horizon '{horizon_name}'."
            )

        forecast_timestamp = (
            forecast_origin
            + pd.Timedelta(
                minutes=15 * horizon_steps
            )
        )

        rows.append(
            {
                "forecast_origin": forecast_origin,
                "horizon": horizon_name,
                "horizon_steps": horizon_steps,
                "forecast_timestamp": forecast_timestamp,
                "forecast_wind_generation_mw": forecast_value,
                "model": model_name,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    forecast: pd.DataFrame,
    cleaning_statistics: dict[str, int],
) -> pd.DataFrame:
    """
    Build a production forecast summary table.
    """

    rows: list[dict[str, Any]] = []

    for _, row in forecast.iterrows():

        rows.append(
            {
                "forecast_origin": row[
                    "forecast_origin"
                ],
                "horizon": row[
                    "horizon"
                ],
                "horizon_steps": row[
                    "horizon_steps"
                ],
                "forecast_timestamp": row[
                    "forecast_timestamp"
                ],
                "forecast_wind_generation_mw": row[
                    "forecast_wind_generation_mw"
                ],
                "model": row[
                    "model"
                ],
                "lags": LAGS,
                "observations_used": (
                    cleaning_statistics[
                        "observations_used"
                    ]
                ),
                "removed_invalid_observations": (
                    cleaning_statistics[
                        "removed_invalid_observations"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "EIRGRID PRODUCTION WIND FORECAST"
    )
    print("=" * 80)
    print()

    # =========================================================================
    # LOAD HORIZON-SPECIFIC MODEL SELECTION
    # =========================================================================

    print(
        "Loading horizon-specific model selection..."
    )

    horizon_model_map = load_horizon_model_map()

    print(
        "Horizon-specific production models:"
    )

    for horizon_name in HORIZONS.keys():

        print(
            f"  {horizon_name}: "
            f"{horizon_model_map[horizon_name]}"
        )

    print()

    # =========================================================================
    # LOAD WIND DATA
    # =========================================================================

    print(
        "Loading EirGrid wind data..."
    )

    dataframe = load_wind_data()

    if WIND_COLUMN not in dataframe.columns:

        raise ValueError(
            f"Expected column '{WIND_COLUMN}' "
            "was not found in the wind dataset."
        )

    wind = dataframe[
        WIND_COLUMN
    ].copy()

    # =========================================================================
    # VALIDATE / CLEAN
    # =========================================================================

    wind, cleaning_statistics = (
        validate_wind_series(
            wind
        )
    )

    print()
    print(
        "WIND DATA VALIDATION"
    )
    print("-" * 80)

    print(
        f"Original observations: "
        f"{cleaning_statistics['original_observations']}"
    )

    print(
        f"Missing observations: "
        f"{cleaning_statistics['missing_observations']}"
    )

    print(
        f"Non-numeric observations: "
        f"{cleaning_statistics['non_numeric_observations']}"
    )

    print(
        f"Invalid observations removed: "
        f"{cleaning_statistics['removed_invalid_observations']}"
    )

    print(
        f"Observations used: "
        f"{cleaning_statistics['observations_used']}"
    )

    print(
        "Interpolation: NONE"
    )

    print(
        "Synthetic observations: NONE"
    )

    print()

    # =========================================================================
    # FORECAST ORIGIN
    # =========================================================================

    forecast_origin = wind.index[-1]

    print(
        f"Forecast origin: "
        f"{forecast_origin}"
    )

    print(
        f"Latest observed wind generation: "
        f"{wind.iloc[-1]:.4f} MW"
    )

    print()

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================

    print(
        "PRODUCTION MODEL CONFIGURATION"
    )
    print("-" * 80)

    print(
        "Forecast horizons:"
    )

    for horizon_name, horizon_steps in HORIZONS.items():

        print(
            f"  {horizon_name}: "
            f"{horizon_steps} step(s) "
            f"-> {horizon_model_map[horizon_name]}"
        )

    print()

    # =========================================================================
    # GENERATE FORECAST
    # =========================================================================

    forecast = generate_production_forecast(
        wind=wind,
        horizon_model_map=horizon_model_map,
    )

    if forecast.empty:

        raise RuntimeError(
            "No production forecasts were generated."
        )

    # =========================================================================
    # DISPLAY FORECAST
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "PRODUCTION WIND FORECAST"
    )
    print("=" * 80)
    print()

    display_columns = [
        "horizon",
        "horizon_steps",
        "forecast_timestamp",
        "forecast_wind_generation_mw",
        "model",
    ]

    display_forecast = forecast[
        display_columns
    ].copy()

    print(
        display_forecast.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # =========================================================================
    # BUILD SUMMARY
    # =========================================================================

    summary = build_summary(
        forecast=forecast,
        cleaning_statistics=cleaning_statistics,
    )

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "PRODUCTION FORECAST OUTPUT"
    )
    print("=" * 80)
    print()

    print(
        f"Saved forecast: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_FILE}"
    )

    print()

    print(
        "Production models by horizon:"
    )

    for horizon_name in HORIZONS.keys():

        print(
            f"  {horizon_name}: "
            f"{horizon_model_map[horizon_name]}"
        )

    print()

    print(
        f"Forecast origin: "
        f"{forecast_origin}"
    )

    print(
        f"Horizons generated: "
        f"{len(forecast)}"
    )

    print()

    print("=" * 80)
    print(
        "PRODUCTION FORECAST COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()