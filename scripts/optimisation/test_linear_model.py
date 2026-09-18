"""
Tests for the MVP linear transmission network.
"""

import pytest

from scripts.optimisation.linear_model import (
    LinearLine,
    LinearNetwork,
)


def build_three_bus_network(
    line_1_limit=300.0,
    line_2_limit=300.0,
):
    """
    Build the simple 3-bus test network.

    Bus 1 ---- Line 1 ---- Bus 2 ---- Line 2 ---- Bus 3
                                                     |
                                                    Load
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


def test_network_initialises():
    """The 3-bus network should initialise correctly."""

    network = build_three_bus_network()

    assert len(network.buses) == 3
    assert len(network.lines) == 2
    assert network.reference_bus == "bus_3"


def test_balanced_injections_produce_line_flows():
    """
    100 MW generated at bus 1 and 100 MW consumed at bus 3
    should produce 100 MW through both lines.
    """

    network = build_three_bus_network()

    injections = {
        "bus_1": 100.0,
        "bus_2": 0.0,
        "bus_3": -100.0,
    }

    flows = network.calculate_line_flows(injections)

    assert flows["line_1"] == pytest.approx(100.0)
    assert flows["line_2"] == pytest.approx(100.0)


def test_thermal_limit_is_respected():
    """
    100 MW flow should be acceptable on a 300 MW line.
    """

    network = build_three_bus_network(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    injections = {
        "bus_1": 100.0,
        "bus_2": 0.0,
        "bus_3": -100.0,
    }

    flows = network.calculate_line_flows(injections)

    checks = network.check_thermal_limits(flows)

    assert checks["line_1"] is True
    assert checks["line_2"] is True


def test_overloaded_line_is_detected():
    """
    400 MW flow should exceed a 300 MW line limit.
    """

    network = build_three_bus_network(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    injections = {
        "bus_1": 400.0,
        "bus_2": 0.0,
        "bus_3": -400.0,
    }

    flows = network.calculate_line_flows(injections)

    overloaded = network.get_overloaded_lines(flows)

    assert "line_1" in overloaded
    assert "line_2" in overloaded


def test_unbalanced_injections_are_rejected():
    """
    The linear network must reject unbalanced injections.
    """

    network = build_three_bus_network()

    injections = {
        "bus_1": 100.0,
        "bus_2": 0.0,
        "bus_3": -50.0,
    }

    with pytest.raises(ValueError, match="must balance to zero"):
        network.calculate_line_flows(injections)