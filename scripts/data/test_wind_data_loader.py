"""
Tests for the EirGrid wind data loader.
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.data.wind_data_loader import load_wind_data


REAL_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "System-Data-Qtr-Hourly-2026-V7.xlsx"
)


def test_real_wind_dataset_loads():
    """
    The real EirGrid workbook should load successfully.
    """

    dataframe = load_wind_data(REAL_FILE)

    assert not dataframe.empty

    assert (
        "wind_availability_mw"
        in dataframe.columns
    )

    assert (
        "wind_generation_mw"
        in dataframe.columns
    )


def test_real_wind_dataset_has_15_minute_resolution():
    """
    The returned dataset must have a strict 15-minute index.
    """

    dataframe = load_wind_data(REAL_FILE)

    differences = (
        dataframe.index.to_series()
        .diff()
        .dropna()
    )

    assert (
        differences
        == pd.Timedelta(minutes=15)
    ).all()


def test_real_wind_generation_is_non_negative():
    """
    Real wind generation must never be negative.

    NaN values are excluded here because the test concerns the
    validity of actual observations.
    """

    dataframe = load_wind_data(REAL_FILE)

    valid_generation = (
        dataframe["wind_generation_mw"]
        .dropna()
    )

    assert (
        valid_generation >= 0
    ).all()


def test_real_wind_timestamps_are_unique():
    """
    The normalised index must contain no duplicates.
    """

    dataframe = load_wind_data(REAL_FILE)

    assert dataframe.index.is_unique


def test_real_wind_filtering():
    """
    The returned dataset should support normal datetime
    filtering.
    """

    dataframe = load_wind_data(REAL_FILE)

    filtered = dataframe.loc[
        "2026-01-01":"2026-01-02"
    ]

    assert not filtered.empty

    assert (
        filtered.index.min()
        >= pd.Timestamp("2026-01-01")
    )

    assert (
        filtered.index.max()
        <= pd.Timestamp("2026-01-02 23:45")
    )


def test_missing_file_is_rejected():
    """
    A missing workbook must raise FileNotFoundError.
    """

    missing_file = (
        REAL_FILE.parent
        / "does_not_exist.xlsx"
    )

    with pytest.raises(FileNotFoundError):
        load_wind_data(missing_file)


def test_dst_transition_is_handled():
    """
    The source workbook contains a daylight-saving-time
    transition around 29 March 2026.

    The loader must return a valid quarter-hourly timeline
    without duplicate timestamps.

    We do not assume that the four 02:00-02:45 timestamps
    must contain NaN. The source workbook uses naive local
    timestamps, and the actual source observations remain
    authoritative.
    """

    dataframe = load_wind_data(REAL_FILE)

    dst_period = dataframe.loc[
        "2026-03-29 01:00":
        "2026-03-29 04:00"
    ]

    assert not dst_period.empty

    differences = (
        dst_period.index.to_series()
        .diff()
        .dropna()
    )

    assert (
        differences
        == pd.Timedelta(minutes=15)
    ).all()

    assert dst_period.index.is_unique