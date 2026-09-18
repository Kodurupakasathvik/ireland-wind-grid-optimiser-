"""
Tests for the optimiser input/output data structures.

These tests verify that the optimisation interface can correctly
store the information required by the forecasting and optimisation
pipeline.
"""

from scripts.optimisation.optimiser_types import (
    OptimiserInput,
    OptimiserResult,
)


def test_optimiser_input():
    """
    Verify that OptimiserInput correctly stores an operating condition.
    """

    data = OptimiserInput(
        snapshot="DUMMY_01",
        demand_mw={
            "bus_3": 400.0,
        },
        available_wind_mw={
            "wind_1": 250.0,
            "wind_2": 250.0,
        },
        wind_capacity_mw={
            "wind_1": 300.0,
            "wind_2": 300.0,
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
    )

    # Operating condition
    assert data.snapshot == "DUMMY_01"

    # Available wind
    assert sum(data.available_wind_mw.values()) == 500.0

    # Installed capacity
    assert sum(data.wind_capacity_mw.values()) == 600.0

    # Demand
    assert data.demand_mw["bus_3"] == 400.0

    # Scenario default
    assert data.scenario == "existing"

    # Transmission data
    assert len(data.lines) == 2
    assert data.lines["line_1"]["limit_mw"] == 300.0


def test_optimiser_result():
    """
    Verify that OptimiserResult correctly stores optimisation results.
    """

    result = OptimiserResult(
        status="optimal",
        available_wind_total_mw=500.0,
        accepted_wind_total_mw=300.0,
        curtailment_total_mw=200.0,
        curtailment_percentage=40.0,
    )

    # Solver status
    assert result.status == "optimal"

    # Main wind quantities
    assert result.available_wind_total_mw == 500.0
    assert result.accepted_wind_total_mw == 300.0
    assert result.curtailment_total_mw == 200.0
    assert result.curtailment_percentage == 40.0

    # Scenario default
    assert result.scenario == "existing"


def test_curtailment_relationship():
    """
    Verify the fundamental project relationship:

        Curtailment = Available Wind - Accepted Wind
    """

    available_wind = 500.0
    accepted_wind = 300.0

    curtailment = available_wind - accepted_wind

    assert curtailment == 200.0


def test_zero_wind_case():
    """
    Verify that zero available wind produces zero curtailment.
    """

    result = OptimiserResult(
        status="optimal",
        available_wind_total_mw=0.0,
        accepted_wind_total_mw=0.0,
        curtailment_total_mw=0.0,
        curtailment_percentage=0.0,
    )

    assert result.available_wind_total_mw == 0.0
    assert result.accepted_wind_total_mw == 0.0
    assert result.curtailment_total_mw == 0.0
    assert result.curtailment_percentage == 0.0