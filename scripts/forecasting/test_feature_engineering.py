import numpy as np
import pandas as pd
import pytest

from scripts.forecasting.feature_engineering import (
    create_wind_features,
)


def make_wind_data(periods=100):
    index = pd.date_range(
        "2026-01-01 00:00",
        periods=periods,
        freq="15min",
    )

    return pd.DataFrame(
        {
            "wind_generation_mw": np.arange(
                periods,
                dtype=float,
            )
        },
        index=index,
    )


def test_feature_engineering_creates_expected_columns():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=1,
    )

    expected_columns = {
        "wind_current_mw",
        "wind_lag_1_steps_mw",
        "wind_lag_2_steps_mw",
        "wind_lag_4_steps_mw",
        "wind_lag_8_steps_mw",
        "wind_lag_16_steps_mw",
        "wind_rolling_4_steps_mean_mw",
        "wind_rolling_4_steps_std_mw",
        "wind_rolling_16_steps_mean_mw",
        "wind_rolling_16_steps_std_mw",
        "time_sin",
        "time_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "target_wind_1_steps_ahead_mw",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_lag_features_are_correct():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=1,
    )

    timestamp = pd.Timestamp(
        "2026-01-01 04:00"
    )

    row = result.loc[timestamp]

    assert row["wind_current_mw"] == 16.0
    assert row["wind_lag_1_steps_mw"] == 15.0
    assert row["wind_lag_2_steps_mw"] == 14.0
    assert row["wind_lag_4_steps_mw"] == 12.0
    assert row["wind_lag_8_steps_mw"] == 8.0
    assert row["wind_lag_16_steps_mw"] == 0.0


def test_target_is_future_observation():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=1,
    )

    timestamp = pd.Timestamp(
        "2026-01-01 04:00"
    )

    assert (
        result.loc[
            timestamp,
            "target_wind_1_steps_ahead_mw",
        ]
        == 17.0
    )


def test_multiple_step_target_is_correct():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=4,
    )

    timestamp = pd.Timestamp(
        "2026-01-01 04:00"
    )

    assert (
        result.loc[
            timestamp,
            "target_wind_4_steps_ahead_mw",
        ]
        == 20.0
    )


def test_rolling_features_use_only_past_data():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=1,
    )

    timestamp = pd.Timestamp(
        "2026-01-01 04:00"
    )

    # At 04:00 the previous four observations are:
    #
    # 03:45 -> 15
    # 03:30 -> 14
    # 03:15 -> 13
    # 03:00 -> 12
    #
    # Mean = 13.5

    assert (
        result.loc[
            timestamp,
            "wind_rolling_4_steps_mean_mw",
        ]
        == 13.5
    )


def test_time_features_are_bounded():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=1,
    )

    for column in [
        "time_sin",
        "time_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]:
        assert (
            result[column].abs() <= 1.0
        ).all()


def test_negative_wind_is_rejected():

    dataframe = make_wind_data()

    dataframe.loc[
        dataframe.index[20],
        "wind_generation_mw",
    ] = -1.0

    with pytest.raises(ValueError):
        create_wind_features(dataframe)


def test_missing_wind_is_rejected():

    dataframe = make_wind_data()

    dataframe.loc[
        dataframe.index[20],
        "wind_generation_mw",
    ] = np.nan

    with pytest.raises(ValueError):
        create_wind_features(dataframe)


def test_unsorted_index_is_rejected():

    dataframe = make_wind_data()

    dataframe = dataframe.iloc[::-1]

    with pytest.raises(ValueError):
        create_wind_features(dataframe)


def test_duplicate_timestamp_is_rejected():

    dataframe = make_wind_data()

    duplicate = dataframe.iloc[[10]]

    dataframe = pd.concat(
        [dataframe, duplicate]
    )

    with pytest.raises(ValueError):
        create_wind_features(dataframe)


def test_empty_dataframe_is_rejected():

    dataframe = pd.DataFrame(
        {
            "wind_generation_mw": []
        },
        index=pd.DatetimeIndex([]),
    )

    with pytest.raises(ValueError):
        create_wind_features(dataframe)


def test_invalid_horizon_is_rejected():

    dataframe = make_wind_data()

    with pytest.raises(ValueError):
        create_wind_features(
            dataframe,
            target_horizon=0,
        )


def test_feature_rows_are_chronological():

    dataframe = make_wind_data()

    result = create_wind_features(
        dataframe,
        target_horizon=1,
    )

    assert result.index.is_monotonic_increasing
    assert not result.index.has_duplicates