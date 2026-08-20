"""
End-to-end MVP pipeline for Ireland Wind Grid Optimiser.

Pipeline:

    observed wind
        ↓
    persistence forecast
        ↓
    available wind
        ↓
    linear grid model
        ↓
    wind curtailment optimiser
        ↓
    accepted wind
        ↓
    curtailment
        ↓
    AC validation

This module is deliberately simple.

It is the first complete end-to-end pipeline and is intended
to prove that all major project components can communicate
before individual components are upgraded.
"""

from dataclasses import dataclass
from typing import Dict

from scripts.forecasting.persistence_forecast import (
    persistence_forecast,
)

from scripts.optimisation.optimiser_types import (
    OptimiserInput,
    OptimiserResult,
)

from scripts.optimisation.wind_curtailment_optimizer import (
    WindCurtailmentOptimizer,
)

from scripts.optimisation.linear_model import (
    LinearNetwork,
)

from scripts.validation.ac_validator import (
    ACValidator,
    ACValidationResult,
)


@dataclass
class MVPPipelineResult:
    """
    Final result from one complete MVP pipeline run.
    """

    # Wind quantities
    observed_wind_mw: float
    forecast_wind_mw: float
    available_wind_mw: float
    accepted_wind_mw: float
    curtailment_mw: float
    curtailment_percentage: float

    # Optimiser
    optimiser_status: str
    optimiser_message: str

    # AC validation
    ac_converged: bool
    physically_secure: bool
    minimum_voltage_pu: float
    maximum_line_loading_percent: float

    # Scenario
    scenario: str
    snapshot: str

    # Human-readable status
    message: str


def run_mvp_pipeline(
    observed_wind_mw: float,
    installed_wind_capacity_mw: float,
    demand_mw: Dict[str, float],
    wind_bus: Dict[str, str],
    network: LinearNetwork,
    optimiser: WindCurtailmentOptimizer,
    ac_validator: ACValidator,
    ac_result: ACValidationResult,
    snapshot: str = "MVP_TEST",
    scenario: str = "existing",
) -> MVPPipelineResult:
    """
    Run the complete MVP wind-to-grid pipeline.

    Parameters
    ----------
    observed_wind_mw:
        Most recent observed total wind generation.

    installed_wind_capacity_mw:
        Installed wind capacity represented by the wind generator(s).

    demand_mw:
        System demand by bus.

    wind_bus:
        Mapping from wind-generator ID to network bus.

    network:
        Simplified linear transmission network.

    optimiser:
        Wind curtailment optimiser.

    ac_validator:
        AC security validator.

    ac_result:
        AC validation result produced by the existing validation layer.

    snapshot:
        Operating snapshot identifier.

    scenario:
        Network scenario identifier.

    Returns
    -------
    MVPPipelineResult
        Complete end-to-end pipeline result.
    """

    # ------------------------------------------------------------
    # 1. Validate basic inputs
    # ------------------------------------------------------------

    if observed_wind_mw < 0:
        raise ValueError(
            "Observed wind generation cannot be negative."
        )

    if installed_wind_capacity_mw <= 0:
        raise ValueError(
            "Installed wind capacity must be positive."
        )

    if observed_wind_mw > installed_wind_capacity_mw:
        raise ValueError(
            "Observed wind generation cannot exceed "
            "installed wind capacity."
        )

    # ------------------------------------------------------------
    # 2. Persistence forecast
    # ------------------------------------------------------------

    forecast = persistence_forecast(
        [observed_wind_mw],
        horizon=1,
    )

    forecast_wind_mw = float(forecast[0])

    # ------------------------------------------------------------
    # 3. Convert forecast into available wind
    #
    # Physical constraint:
    #
    #     0 <= available wind <= installed capacity
    # ------------------------------------------------------------

    available_wind_mw = min(
        max(forecast_wind_mw, 0.0),
        installed_wind_capacity_mw,
    )

    # ------------------------------------------------------------
    # 4. Build optimiser input
    # ------------------------------------------------------------

    wind_capacity = {
        generator_id: installed_wind_capacity_mw
        for generator_id in wind_bus
    }

    available_wind = {
        generator_id: available_wind_mw
        for generator_id in wind_bus
    }

    optimiser_input = OptimiserInput(
        snapshot=snapshot,
        demand_mw=demand_mw,
        available_wind_mw=available_wind,
        wind_capacity_mw=wind_capacity,
        lines={
            line.line_id: {
                "from_bus": line.from_bus,
                "to_bus": line.to_bus,
                "limit_mw": line.limit_mw,
            }
            for line in network.lines
        },
        scenario=scenario,
    )

    # ------------------------------------------------------------
    # 5. Run grid-aware wind optimiser
    # ------------------------------------------------------------

    optimiser_result: OptimiserResult = optimiser.solve(
        optimiser_input,
        wind_bus=wind_bus,
    )

    # ------------------------------------------------------------
    # 6. Extract curtailment result
    # ------------------------------------------------------------

    accepted_wind_mw = (
        optimiser_result.accepted_wind_total_mw
    )

    curtailment_mw = (
        optimiser_result.curtailment_total_mw
    )

    curtailment_percentage = (
        optimiser_result.curtailment_percentage
    )

    # ------------------------------------------------------------
    # 7. AC validation
    #
    # The AC model is a validation layer.
    # It does not replace the optimiser.
    # ------------------------------------------------------------

    validated_ac = ac_validator.validate_result(
        ac_result
    )

    # ------------------------------------------------------------
    # 8. Final pipeline status
    # ------------------------------------------------------------

    if validated_ac.physically_secure:
        message = (
            "MVP pipeline completed successfully: "
            "optimised dispatch passed AC security validation."
        )
    else:
        message = (
            "MVP pipeline completed, but the optimised "
            "dispatch failed AC security validation."
        )

    return MVPPipelineResult(
        observed_wind_mw=observed_wind_mw,
        forecast_wind_mw=forecast_wind_mw,
        available_wind_mw=available_wind_mw,
        accepted_wind_mw=accepted_wind_mw,
        curtailment_mw=curtailment_mw,
        curtailment_percentage=curtailment_percentage,
        optimiser_status=optimiser_result.status,
        optimiser_message=optimiser_result.message,
        ac_converged=validated_ac.converged,
        physically_secure=validated_ac.physically_secure,
        minimum_voltage_pu=validated_ac.minimum_voltage_pu,
        maximum_line_loading_percent=(
            validated_ac.maximum_line_loading_percent
        ),
        scenario=scenario,
        snapshot=snapshot,
        message=message,
    )