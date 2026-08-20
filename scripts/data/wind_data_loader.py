"""
EirGrid wind data loader.

Loads quarter-hourly wind data from the EirGrid
System-Data-Qtr-Hourly-2026-V7.xlsx workbook.

The source dataset uses naive local DateTime values. The loader
normalises the returned index to a continuous 15-minute timeline
without fabricating or overwriting wind measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "System-Data-Qtr-Hourly-2026-V7.xlsx"
)

DATETIME_COLUMN = "DateTime"

IE_WIND_AVAILABILITY_COLUMN = "IE Wind Availability"
IE_WIND_GENERATION_COLUMN = "IE Wind Generation"


def load_wind_data(
    filepath: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load Irish wind data from the EirGrid workbook.

    Parameters
    ----------
    filepath:
        Optional path to the EirGrid workbook.

        If omitted, the project default is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by DateTime with:

        - wind_availability_mw
        - wind_generation_mw

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.

    ValueError
        If required columns are missing, timestamps are invalid,
        timestamps are duplicated, or wind values are invalid.
    """

    path = (
        Path(filepath)
        if filepath is not None
        else DEFAULT_FILE
    )

    # ------------------------------------------------------------------
    # File validation
    # ------------------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"Wind data file does not exist: {path}"
        )

    # ------------------------------------------------------------------
    # Load workbook
    # ------------------------------------------------------------------

    dataframe = pd.read_excel(
        path,
        sheet_name="System Data",
    )

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    required_columns = {
        DATETIME_COLUMN,
        IE_WIND_AVAILABILITY_COLUMN,
        IE_WIND_GENERATION_COLUMN,
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the EirGrid "
            f"workbook: {sorted(missing_columns)}"
        )

    dataframe = dataframe[
        [
            DATETIME_COLUMN,
            IE_WIND_AVAILABILITY_COLUMN,
            IE_WIND_GENERATION_COLUMN,
        ]
    ].copy()

    # ------------------------------------------------------------------
    # Timestamp conversion
    # ------------------------------------------------------------------

    dataframe[DATETIME_COLUMN] = pd.to_datetime(
        dataframe[DATETIME_COLUMN],
        errors="coerce",
    )

    if dataframe[DATETIME_COLUMN].isna().any():

        invalid_count = int(
            dataframe[DATETIME_COLUMN].isna().sum()
        )

        raise ValueError(
            f"Found {invalid_count} invalid DateTime values."
        )

    # ------------------------------------------------------------------
    # Duplicate timestamp validation
    # ------------------------------------------------------------------

    if dataframe[DATETIME_COLUMN].duplicated().any():

        duplicate_count = int(
            dataframe[DATETIME_COLUMN].duplicated().sum()
        )

        raise ValueError(
            "Duplicate DateTime values detected: "
            f"{duplicate_count}"
        )

    # ------------------------------------------------------------------
    # Sort and index
    # ------------------------------------------------------------------

    dataframe = dataframe.sort_values(
        DATETIME_COLUMN
    )

    dataframe = dataframe.set_index(
        DATETIME_COLUMN
    )

    # ------------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------------

    dataframe[
        IE_WIND_AVAILABILITY_COLUMN
    ] = pd.to_numeric(
        dataframe[IE_WIND_AVAILABILITY_COLUMN],
        errors="coerce",
    )

    dataframe[
        IE_WIND_GENERATION_COLUMN
    ] = pd.to_numeric(
        dataframe[IE_WIND_GENERATION_COLUMN],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # Missing-value validation
    # ------------------------------------------------------------------

    if dataframe[
        IE_WIND_AVAILABILITY_COLUMN
    ].isna().any():

        raise ValueError(
            "Wind availability contains missing or "
            "non-numeric values."
        )

    if dataframe[
        IE_WIND_GENERATION_COLUMN
    ].isna().any():

        raise ValueError(
            "Wind generation contains missing or "
            "non-numeric values."
        )

    # ------------------------------------------------------------------
    # Negative-value validation
    # ------------------------------------------------------------------

    negative_availability = (
        dataframe[IE_WIND_AVAILABILITY_COLUMN] < 0
    )

    if negative_availability.any():

        count = int(
            negative_availability.sum()
        )

        raise ValueError(
            f"Wind availability contains {count} "
            "negative observations."
        )

    negative_generation = (
        dataframe[IE_WIND_GENERATION_COLUMN] < 0
    )

    if negative_generation.any():

        count = int(
            negative_generation.sum()
        )

        raise ValueError(
            f"Wind generation contains {count} "
            "negative observations."
        )

    # ------------------------------------------------------------------
    # Rename columns to project-level names
    # ------------------------------------------------------------------

    dataframe = dataframe.rename(
        columns={
            IE_WIND_AVAILABILITY_COLUMN:
                "wind_availability_mw",
            IE_WIND_GENERATION_COLUMN:
                "wind_generation_mw",
        }
    )

    # ------------------------------------------------------------------
    # Normalise the index to a 15-minute grid.
    #
    # IMPORTANT:
    #
    # We do not interpolate or fabricate observations.
    #
    # If the source has a genuine missing timestamp, reindex()
    # will represent it as NaN.
    #
    # However, the source workbook is treated as the authoritative
    # source of actual wind observations.
    # ------------------------------------------------------------------

    start = dataframe.index.min()
    end = dataframe.index.max()

    complete_index = pd.date_range(
        start=start,
        end=end,
        freq="15min",
    )

    dataframe = dataframe.reindex(
        complete_index
    )

    dataframe.index.name = DATETIME_COLUMN

    # ------------------------------------------------------------------
    # Final index validation
    # ------------------------------------------------------------------

    differences = (
        dataframe.index.to_series()
        .diff()
        .dropna()
    )

    if not (
        differences
        == pd.Timedelta(minutes=15)
    ).all():

        raise ValueError(
            "Internal error: wind dataset is not "
            "strictly quarter-hourly after normalisation."
        )

    return dataframe


__all__ = [
    "load_wind_data",
    "DEFAULT_FILE",
]