"""
Tests for the linear wind-curtailment optimiser.
"""

import pytest

from scripts.optimisation.linear_model import (
    LinearLine,
    LinearNetwork,
)

from scripts.optimisation.optimiser_types import (
    OptimiserInput,
)

from scripts.optimisation.wind_curtailment_optimizer import (
    WindCurtailmentOptimizer,
)


def build_test_network(
    line_1_limit=300.0,
    line_2_limit=300.0,
):
    """
    Build the 3-bus MVP network.

    Wind:
        Bus 1

    Transmission:
        Bus 1 ---- Line 1 ---- Bus 2 ---- Line 2 ---- Bus 3

    Load:
        Bus 3

    Bus 3 is the reference/slack bus.
    """

    buses = [
        "bus_1",
        "bus_2",
        "bus_3",
    ]

    lines = [
        LinearLine(
            line_id="line_1",
            from_bus="bus_1",
            to_bus="bus_2",
            susceptance=1.0,
            limit_mw=line_1_limit,
        ),
        LinearLine(
            line_id="line_2",
            from_bus="bus_2",
            to_bus="bus_3",
            susceptance=1.0,
            limit_mw=line_2_limit,
        ),
    ]

    return LinearNetwork(
        buses=buses,
        lines=lines,
        reference_bus="bus_3",
    )


def build_input(
    available_wind=500.0,
    demand=500.0,
):
    """
    Create a simple operating condition.
    """

    return OptimiserInput(
        snapshot="DUMMY_01",

        demand_mw={
            "bus_3": demand,
        },

        available_wind_mw={
            "wind_1": available_wind,
        },

        wind_capacity_mw={
            "wind_1": 500.0,
        },

        lines={
            "line_1": {
                "from_bus": "bus_1",
                "to_bus": "bus_2",
                "limit_mw": 300.0,
            },
            "line_2": {
                "from_bus": "bus_2",
                "to_bus": "bus_3",
                "limit_mw": 300.0,
            },
        },

        scenario="existing",
    )


def test_optimizer_accepts_available_wind_when_no_congestion():
    """
    If the network can accommodate all available wind,
    accepted wind should equal available wind.
    """

    network = build_test_network(
        line_1_limit=600.0,
        line_2_limit=600.0,
    )

    optimiser = WindCurtailmentOptimizer(
        network
    )

    data = build_input(
        available_wind=500.0,
        demand=500.0,
    )

    result = optimiser.solve(
        data,
        wind_bus={
            "wind_1": "bus_1",
        },
    )

    assert result.status == "optimal"

    assert result.available_wind_total_mw == pytest.approx(
        500.0
    )

    assert result.accepted_wind_total_mw == pytest.approx(
        500.0
    )

    assert result.curtailment_total_mw == pytest.approx(
        0.0
    )

    assert result.curtailment_percentage == pytest.approx(
        0.0
    )


def test_optimizer_curtails_wind_when_line_is_constrained():
    """
    A 300 MW line should limit accepted wind to approximately
    300 MW when 500 MW is available.
    """

    network = build_test_network(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    optimiser = WindCurtailmentOptimizer(
        network
    )

    data = build_input(
        available_wind=500.0,
        demand=500.0,
    )

    result = optimiser.solve(
        data,
        wind_bus={
            "wind_1": "bus_1",
        },
    )

    assert result.status == "optimal"

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


def test_generator_level_curtailment_is_reported():
    """
    Generator-level accepted and curtailed wind should be returned.
    """

    network = build_test_network(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    optimiser = WindCurtailmentOptimizer(
        network
    )

    data = build_input(
        available_wind=500.0,
        demand=500.0,
    )

    result = optimiser.solve(
        data,
        wind_bus={
            "wind_1": "bus_1",
        },
    )

    assert result.accepted_wind_by_generator_mw[
        "wind_1"
    ] == pytest.approx(300.0)

    assert result.curtailed_wind_by_generator_mw[
        "wind_1"
    ] == pytest.approx(200.0)


def test_negative_available_wind_is_rejected():
    """
    Negative available wind is physically invalid.
    """

    network = build_test_network()

    optimiser = WindCurtailmentOptimizer(
        network
    )

    data = build_input(
        available_wind=-10.0,
        demand=500.0,
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):

        optimiser.solve(
            data,
            wind_bus={
                "wind_1": "bus_1",
            },
        )


def test_available_wind_cannot_exceed_capacity():
    """
    Available wind cannot exceed installed capacity.
    """

    network = build_test_network()

    optimiser = WindCurtailmentOptimizer(
        network
    )

    data = build_input(
        available_wind=600.0,
        demand=500.0,
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed installed capacity",
    ):

        optimiser.solve(
            data,
            wind_bus={
                "wind_1": "bus_1",
            },
        )