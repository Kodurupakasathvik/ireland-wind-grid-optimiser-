"""
Tests for the complete MVP wind-to-grid pipeline.

The test intentionally uses a small synthetic network.

Pipeline tested:

    observed wind
        ↓
    persistence forecast
        ↓
    available wind
        ↓
    linear grid
        ↓
    wind curtailment optimiser
        ↓
    curtailment
        ↓
    AC validation
"""

from scripts.pipeline.mvp_pipeline import run_mvp_pipeline

from scripts.optimisation.linear_model import (
    LinearLine,
    LinearNetwork,
)

from scripts.optimisation.wind_curtailment_optimizer import (
    WindCurtailmentOptimizer,
)

from scripts.validation.ac_validator import (
    ACValidationResult,
    ACValidator,
)


def build_test_network(
    line_1_limit: float = 600.0,
    line_2_limit: float = 600.0,
):
    """
    Build a simple 3-bus network.

    bus_1:
        wind generator

    bus_2:
        intermediate transmission bus

    bus_3:
        demand

    Reference bus:
        bus_3
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


def build_ac_result(
    secure: bool = True,
):
    """
    Build a synthetic AC validation result.

    The actual Irish PyPSA AC model will be connected later.

    For this MVP integration test, the result represents
    the output that would come from the AC validation layer.
    """

    if secure:
        return ACValidationResult(
            converged=True,
            minimum_voltage_pu=0.98,
            weak_bus="bus_2",
            weak_bus_voltage_pu=0.98,
            maximum_line_loading_percent=80.0,
            overloaded_lines=[],
            scenario="existing",
            snapshot="MVP_TEST",
        )

    return ACValidationResult(
        converged=True,
        minimum_voltage_pu=0.92,
        weak_bus="bus_2",
        weak_bus_voltage_pu=0.92,
        maximum_line_loading_percent=80.0,
        overloaded_lines=[],
        scenario="existing",
        snapshot="MVP_TEST",
    )


def test_complete_mvp_pipeline_without_congestion():
    """
    With sufficient transmission capacity:

        observed wind       = 500 MW
        forecast wind       = 500 MW
        available wind     = 500 MW
        accepted wind      = 500 MW
        curtailment        = 0 MW

    The AC validation result is secure.
    """

    network = build_test_network(
        line_1_limit=600.0,
        line_2_limit=600.0,
    )

    optimiser = WindCurtailmentOptimizer(
        network
    )

    validator = ACValidator(
        voltage_min_pu=0.95,
        line_loading_limit_percent=100.0,
    )

    ac_result = build_ac_result(
        secure=True
    )

    result = run_mvp_pipeline(
        observed_wind_mw=500.0,
        installed_wind_capacity_mw=600.0,
        demand_mw={
            "bus_3": 500.0,
        },
        wind_bus={
            "wind_1": "bus_1",
        },
        network=network,
        optimiser=optimiser,
        ac_validator=validator,
        ac_result=ac_result,
    )

    assert result.observed_wind_mw == 500.0
    assert result.forecast_wind_mw == 500.0
    assert result.available_wind_mw == 500.0

    assert result.accepted_wind_mw == 500.0
    assert result.curtailment_mw == 0.0
    assert result.curtailment_percentage == 0.0

    assert result.ac_converged is True
    assert result.physically_secure is True


def test_complete_mvp_pipeline_with_congestion():
    """
    With a 300 MW transmission corridor:

        observed wind       = 500 MW
        forecast wind       = 500 MW
        available wind     = 500 MW
        accepted wind      = 300 MW
        curtailment        = 200 MW
        curtailment        = 40%

    """

    network = build_test_network(
        line_1_limit=300.0,
        line_2_limit=300.0,
    )

    optimiser = WindCurtailmentOptimizer(
        network
    )

    validator = ACValidator(
        voltage_min_pu=0.95,
        line_loading_limit_percent=100.0,
    )

    ac_result = build_ac_result(
        secure=True
    )

    result = run_mvp_pipeline(
        observed_wind_mw=500.0,
        installed_wind_capacity_mw=600.0,
        demand_mw={
            "bus_3": 500.0,
        },
        wind_bus={
            "wind_1": "bus_1",
        },
        network=network,
        optimiser=optimiser,
        ac_validator=validator,
        ac_result=ac_result,
    )

    assert result.available_wind_mw == 500.0
    assert result.accepted_wind_mw == 300.0
    assert result.curtailment_mw == 200.0

    assert result.curtailment_percentage == 40.0

    assert result.ac_converged is True
    assert result.physically_secure is True


def test_mvp_pipeline_reports_ac_security_failure():
    """
    The optimisation can complete while the AC validation
    subsequently identifies an insecure operating point.

    This confirms that AC validation is a validation layer,
    not the optimisation itself.
    """

    network = build_test_network(
        line_1_limit=600.0,
        line_2_limit=600.0,
    )

    optimiser = WindCurtailmentOptimizer(
        network
    )

    validator = ACValidator(
        voltage_min_pu=0.95,
        line_loading_limit_percent=100.0,
    )

    ac_result = build_ac_result(
        secure=False
    )

    result = run_mvp_pipeline(
        observed_wind_mw=500.0,
        installed_wind_capacity_mw=600.0,
        demand_mw={
            "bus_3": 500.0,
        },
        wind_bus={
            "wind_1": "bus_1",
        },
        network=network,
        optimiser=optimiser,
        ac_validator=validator,
        ac_result=ac_result,
    )

    assert result.ac_converged is True
    assert result.physically_secure is False