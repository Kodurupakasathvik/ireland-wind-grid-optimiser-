"""
Tests for forecast-to-grid integration.
"""

import pytest

from scripts.pipeline.forecast_to_grid import (
    run_forecast_to_grid,
)

from scripts.optimisation.linear_model import (
    LinearLine,
    LinearNetwork,
)


def build_test_network(
    line_1_limit=600.0,
    line_2_limit=600.0,
):
    """
    Build the small three-bus network used for the MVP
    forecast-to-grid integration tests.
    """

    lines = [
        LinearLine(
            line_id="line_1",
            from_bus="bus_1",
            to_bus="bus_2",
            susceptance=10.0,
            limit_mw=line_1_limit,
        ),
        LinearLine(
            line_id="line_2",
            from_bus="bus_2",
            to_bus="bus_3",
            susceptance=10.0,
            limit_mw=line_2_limit,
        ),
    ]

    return LinearNetwork(
        buses=[
            "bus_1",
            "bus_2",
            "bus_3",
        ],
        lines=lines,
        reference_bus="bus_3",
    )


def build_common_inputs(
    line_1_limit=600.0,
    line_2_limit=600.0,
):
    """
    Common simplified test data.
    """

    return {
        "snapshot": "TEST_HIGH_WIND",

        "demand_mw": {
            "bus_3": 500.0,
        },

        "wind_capacity_mw": {
            "wind_1": 600.0,
        },

        "lines": {
            "line_1": {
                "from_bus": "bus_1",
                "to_bus": "bus_2",
                "limit_mw": line_1_limit,
            },
            "line_2": {
                "from_bus": "bus_2",
                "to_bus": "bus_3",
                "limit_mw": line_2_limit,
            },
        },

        "wind_bus": {
            "wind_1": "bus_1",
        },
    }


def test_forecast_feeds_available_wind_to_optimizer():
    """
    Latest observed wind = 500 MW.

    Persistence forecast:
        available wind = 500 MW

    With sufficient transmission capacity:
        accepted = 500 MW
        curtailed = 0 MW
    """

    network = build_test_network()

    inputs = build_common_inputs()

    result = run_forecast_to_grid(
        observed_wind_mw=[
            400.0,
            450.0,
            500.0,
        ],
        snapshot=inputs["snapshot"],
        demand_mw=inputs["demand_mw"],
        wind_capacity_mw=inputs["wind_capacity_mw"],
        lines=inputs["lines"],
        wind_bus=inputs["wind_bus"],
        network=network,
    )

    assert result.available_wind_total_mw == pytest.approx(
        500.0
    )

    assert result.accepted_wind_total_mw == pytest.approx(
        500.0
    )

    assert result.curtailment_total_mw == pytest.approx(
        0.0
    )


def test_forecast_is_curtailed_by_grid_constraint():
    """
    Latest observed wind = 500 MW.

    Transmission capacity = 300 MW.

    Expected:

        available = 500 MW
        accepted  = 300 MW
        curtailed = 200 MW
    """

    network = build_test_network(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    inputs = build_common_inputs(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    result = run_forecast_to_grid(
        observed_wind_mw=[
            400.0,
            450.0,
            500.0,
        ],
        snapshot=inputs["snapshot"],
        demand_mw=inputs["demand_mw"],
        wind_capacity_mw=inputs["wind_capacity_mw"],
        lines=inputs["lines"],
        wind_bus=inputs["wind_bus"],
        network=network,
    )

    assert result.available_wind_total_mw == pytest.approx(
        500.0
    )

    assert result.accepted_wind_total_mw == pytest.approx(
        300.0
    )

    assert result.curtailment_total_mw == pytest.approx(
        200.0
    )

    assert result.curtailment_percentage == pytest.approx(
        40.0
    )


def test_zero_wind_flows_through_pipeline():
    """
    Zero observed wind should produce zero available wind,
    zero accepted wind and zero curtailment.
    """

    network = build_test_network()

    inputs = build_common_inputs()

    result = run_forecast_to_grid(
        observed_wind_mw=[
            0.0,
        ],
        snapshot=inputs["snapshot"],
        demand_mw=inputs["demand_mw"],
        wind_capacity_mw=inputs["wind_capacity_mw"],
        lines=inputs["lines"],
        wind_bus=inputs["wind_bus"],
        network=network,
    )

    assert result.available_wind_total_mw == pytest.approx(
        0.0
    )

    assert result.accepted_wind_total_mw == pytest.approx(
        0.0
    )

    assert result.curtailment_total_mw == pytest.approx(
        0.0
    )