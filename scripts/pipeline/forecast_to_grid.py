"""
Forecast-to-grid integration.

Connects:

    observed wind
        ↓
    persistence forecast
        ↓
    available wind
        ↓
    grid-aware optimiser
        ↓
    accepted wind
        ↓
    curtailment

This module intentionally contains only the integration logic.
Forecasting and optimisation remain separate components.
"""

from typing import Dict, Sequence

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


def run_forecast_to_grid(
    *,
    observed_wind_mw: Sequence[float],
    snapshot: str,
    demand_mw: Dict[str, float],
    wind_capacity_mw: Dict[str, float],
    lines: Dict[str, Dict[str, object]],
    wind_bus: Dict[str, str],
    network,
    scenario: str = "existing",
) -> OptimiserResult:
    """
    Run the first forecast-to-grid pipeline.

    The latest observed wind value is used as the persistence
    forecast for the next operating period.

    Parameters
    ----------
    observed_wind_mw:
        Historical wind-generation observations in MW.

    snapshot:
        Operating-condition identifier.

    demand_mw:
        Demand by bus.

    wind_capacity_mw:
        Installed wind capacity by generator ID.

    lines:
        Transmission-line definitions used by the optimiser.

    wind_bus:
        Mapping from wind-generator ID to network bus.

    network:
        LinearNetwork instance used by the optimiser.

    scenario:
        Network scenario name.

    Returns
    -------
    OptimiserResult
        Result from the grid-aware wind-curtailment optimiser.
    """

    # ----------------------------------------------------------
    # 1. Forecast available wind
    # ----------------------------------------------------------

    forecast = persistence_forecast(
        observed_wind_mw,
        horizon=1,
    )

    forecast_available_wind_mw = float(
        forecast[0]
    )

    # ----------------------------------------------------------
    # 2. Map forecast to each wind generator
    #
    # For the MVP, a single observed system-wide wind value
    # is distributed proportionally to installed wind capacity.
    #
    # This is deliberately simple.
    # Generator-level forecasting will be introduced later.
    # ----------------------------------------------------------

    total_capacity_mw = sum(
        wind_capacity_mw.values()
    )

    if total_capacity_mw <= 0:
        raise ValueError(
            "Total installed wind capacity must be greater than zero."
        )

    available_wind_mw = {}

    for generator_id, capacity_mw in wind_capacity_mw.items():

        share = (
            float(capacity_mw)
            / total_capacity_mw
        )

        available_wind_mw[generator_id] = (
            forecast_available_wind_mw
            * share
        )

    # ----------------------------------------------------------
    # 3. Create optimiser input
    # ----------------------------------------------------------

    optimiser_input = OptimiserInput(
        snapshot=snapshot,
        demand_mw=demand_mw,
        available_wind_mw=available_wind_mw,
        wind_capacity_mw=wind_capacity_mw,
        lines=lines,
        scenario=scenario,
    )

    # ----------------------------------------------------------
    # 4. Run grid-aware optimiser
    # ----------------------------------------------------------

    optimiser = WindCurtailmentOptimizer(
        network
    )

    result = optimiser.solve(
        optimiser_input,
        wind_bus=wind_bus,
    )

    return result