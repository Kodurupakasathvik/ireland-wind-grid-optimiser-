import pytest

from scripts.forecasting.forecast_evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    normalized_mae,
    normalized_mae_percent,
    forecast_metrics,
)


def test_mae():

    actual = [100.0, 200.0, 300.0]
    forecast = [110.0, 180.0, 330.0]

    result = mean_absolute_error(
        actual,
        forecast,
    )

    assert result == pytest.approx(
        20.0
    )


def test_rmse():

    actual = [100.0, 200.0, 300.0]
    forecast = [110.0, 180.0, 330.0]

    result = root_mean_squared_error(
        actual,
        forecast,
    )

    expected = (
        (100.0 + 400.0 + 900.0) / 3
    ) ** 0.5

    assert result == pytest.approx(
        expected
    )


def test_normalized_mae():

    actual = [100.0, 200.0, 300.0]
    forecast = [110.0, 180.0, 330.0]

    result = normalized_mae(
        actual,
        forecast,
    )

    assert result == pytest.approx(
        20.0 / 200.0
    )


def test_normalized_mae_percent():

    actual = [100.0, 200.0, 300.0]
    forecast = [110.0, 180.0, 330.0]

    result = normalized_mae_percent(
        actual,
        forecast,
    )

    assert result == pytest.approx(
        10.0
    )


def test_perfect_forecast():

    actual = [100.0, 200.0, 300.0]

    result = forecast_metrics(
        actual,
        actual,
    )

    assert result["mae_mw"] == pytest.approx(0.0)
    assert result["rmse_mw"] == pytest.approx(0.0)
    assert result["nmae"] == pytest.approx(0.0)
    assert result["nmae_percent"] == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected():

    with pytest.raises(ValueError):

        mean_absolute_error(
            [100.0, 200.0],
            [100.0],
        )


def test_negative_actual_is_rejected():

    with pytest.raises(ValueError):

        mean_absolute_error(
            [-100.0],
            [100.0],
        )


def test_negative_forecast_is_rejected():

    with pytest.raises(ValueError):

        mean_absolute_error(
            [100.0],
            [-100.0],
        )


def test_empty_input_is_rejected():

    with pytest.raises(ValueError):

        mean_absolute_error(
            [],
            [],
        )


def test_zero_mean_actual_is_rejected_for_normalized_metric():

    with pytest.raises(ValueError):

        normalized_mae(
            [0.0, 0.0],
            [0.0, 0.0],
        )