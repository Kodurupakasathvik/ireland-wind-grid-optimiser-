"""
Forecast-to-grid integration.

Pipeline:

    observed wind
        ↓
    forecasting model
        ↓
    forecast available wind
        ↓
    generator-level wind allocation
        ↓
    grid-aware LP optimiser
        ↓
    accepted wind
        ↓
    curtailment

This module intentionally keeps forecasting and optimisation
as separate components.

The forecasting model is supplied to this module rather than
being hard-coded. This allows the validated production model
selection to evolve without changing the optimiser.

All power quantities are expressed in MW.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from scripts.optimisation.optimiser_types import (
    OptimiserInput,
    OptimiserResult,
)

from scripts.optimisation.wind_curtailment_optimizer import (
    WindCurtailmentOptimizer,
)


# ----------------------------------------------------------------------
# Type alias
# ----------------------------------------------------------------------

ForecastFunction = Callable[
    [Sequence[float]],
    float,
]


def persistence_forecast_function(
    observed_wind_mw: Sequence[float],
) -> float:
    """
    Simple persistence forecast.

    This fallback is intentionally kept local to this integration
    module so that the production pipeline does not require the
    persistence model.

    Forecast:

        P(t+1) = P(t)

    Parameters
    ----------
    observed_wind_mw:
        Historical observed wind-generation values in MW.

    Returns
    -------
    float
        One-step forecast in MW.
    """

    if not observed_wind_mw:
        raise ValueError(
            "observed_wind_mw cannot be empty."
        )

    last_value = float(
        observed_wind_mw[-1]
    )

    if not np.isfinite(last_value):
        raise ValueError(
            "The latest observed wind value must be finite."
        )

    if last_value < 0:
        raise ValueError(
            "Wind generation cannot be negative."
        )

    return last_value


class ForecastToGrid:
    """
    Forecast-to-grid integration layer.

    Responsibilities
    ----------------
    1. Validate observed wind.
    2. Obtain a wind forecast from the supplied forecasting model.
    3. Limit forecast wind to installed capacity.
    4. Distribute system-level forecast wind across wind generators.
    5. Construct OptimiserInput.
    6. Run WindCurtailmentOptimizer.

    The class does NOT:
        - train forecasting models,
        - perform AC power flow,
        - modify the PyPSA network,
        - perform physical AC validation.

    Those responsibilities remain in their respective modules.
    """

    def __init__(
        self,
        network,
        forecast_function: ForecastFunction | None = None,
    ) -> None:
        """
        Parameters
        ----------
        network:
            LinearNetwork instance used by the LP optimiser.

        forecast_function:
            Callable receiving observed wind history and returning
            one forecast value in MW.

            If omitted, a persistence fallback is used.

            Production use should provide the validated forecasting
            model explicitly.
        """

        if network is None:
            raise ValueError(
                "network cannot be None."
            )

        self.network = network

        self.forecast_function = (
            forecast_function
            if forecast_function is not None
            else persistence_forecast_function
        )

        if not callable(
            self.forecast_function
        ):
            raise ValueError(
                "forecast_function must be callable."
            )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def run(
        self,
        *,
        observed_wind_mw: Sequence[float],
        snapshot: str,
        demand_mw: Dict[str, float],
        wind_capacity_mw: Dict[str, float],
        wind_bus: Dict[str, str],
        scenario: str = "existing",
        conventional_generation_mw: Dict[str, float] | None = None,
        interconnector_dispatch_mw: Dict[str, float] | None = None,
        interconnectors: Dict[str, Dict[str, float]] | None = None,
    ) -> OptimiserResult:
        """
        Run the complete forecast-to-LP-optimisation pipeline.

        Parameters
        ----------
        observed_wind_mw:
            Historical/system-level observed wind generation.

        snapshot:
            Operating-condition identifier.

        demand_mw:
            Demand by network bus.

        wind_capacity_mw:
            Installed wind capacity by generator ID.

        wind_bus:
            Mapping:

                wind_generator_id -> bus_id

        scenario:
            Network scenario identifier.

        conventional_generation_mw:
            Existing fixed conventional generation by bus.

        interconnector_dispatch_mw:
            Fixed interconnector dispatch by bus.

        interconnectors:
            Optional interconnector definitions/limits.

        Returns
        -------
        OptimiserResult
            Result produced by WindCurtailmentOptimizer.
        """

        # --------------------------------------------------------------
        # 1. Validate integration inputs
        # --------------------------------------------------------------

        self._validate_inputs(
            observed_wind_mw=observed_wind_mw,
            snapshot=snapshot,
            demand_mw=demand_mw,
            wind_capacity_mw=wind_capacity_mw,
            wind_bus=wind_bus,
        )

        # --------------------------------------------------------------
        # 2. Forecast wind
        # --------------------------------------------------------------

        forecast_wind_mw = float(
            self.forecast_function(
                observed_wind_mw
            )
        )

        if not np.isfinite(
            forecast_wind_mw
        ):
            raise ValueError(
                "Forecast wind generation must be finite."
            )

        if forecast_wind_mw < 0:
            raise ValueError(
                "Forecast wind generation cannot be negative."
            )

        # --------------------------------------------------------------
        # 3. Physical capacity limit
        #
        # The forecast represents available wind.
        #
        # It cannot exceed the total installed wind capacity.
        # --------------------------------------------------------------

        total_capacity_mw = float(
            sum(
                wind_capacity_mw.values()
            )
        )

        available_total_mw = min(
            forecast_wind_mw,
            total_capacity_mw,
        )

        # --------------------------------------------------------------
        # 4. Allocate system-level forecast to generators
        #
        # MVP assumption:
        #
        #     generator share =
        #         generator capacity /
        #         total installed capacity
        #
        # Therefore:
        #
        #     available_generator =
        #         system forecast × capacity share
        #
        # Generator-level forecasting can replace this later.
        # --------------------------------------------------------------

        available_wind_mw = (
            self._allocate_forecast_to_generators(
                total_forecast_mw=available_total_mw,
                wind_capacity_mw=wind_capacity_mw,
            )
        )

        # --------------------------------------------------------------
        # 5. Build line definitions from LinearNetwork
        # --------------------------------------------------------------

        lines = {
            line.line_id: {
                "from_bus": line.from_bus,
                "to_bus": line.to_bus,
                "limit_mw": line.limit_mw,
            }
            for line in self.network.lines
        }

        # --------------------------------------------------------------
        # 6. Build OptimiserInput
        # --------------------------------------------------------------

        optimiser_input = OptimiserInput(
            snapshot=snapshot,
            demand_mw=dict(demand_mw),
            available_wind_mw=available_wind_mw,
            wind_capacity_mw=dict(wind_capacity_mw),
            lines=lines,
            interconnectors=(
                dict(interconnectors)
                if interconnectors is not None
                else {}
            ),
            conventional_generation_mw=(
                dict(conventional_generation_mw)
                if conventional_generation_mw is not None
                else {}
            ),
            interconnector_dispatch_mw=(
                dict(interconnector_dispatch_mw)
                if interconnector_dispatch_mw is not None
                else {}
            ),
            scenario=scenario,
        )

        # --------------------------------------------------------------
        # 7. Run LP optimiser
        # --------------------------------------------------------------

        optimiser = WindCurtailmentOptimizer(
            self.network
        )

        result = optimiser.solve(
            optimiser_input,
            wind_bus=wind_bus,
        )

        return result

    # ==================================================================
    # FORECAST ALLOCATION
    # ==================================================================

    @staticmethod
    def _allocate_forecast_to_generators(
        *,
        total_forecast_mw: float,
        wind_capacity_mw: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Allocate system-level forecast wind to individual generators.

        Allocation is proportional to installed capacity.

        Example
        -------
        If:

            Generator A = 600 MW
            Generator B = 400 MW

        and:

            System forecast = 500 MW

        then:

            A = 300 MW
            B = 200 MW

        This is an MVP allocation method.

        It will later be replaceable by generator-level forecasts.
        """

        if total_forecast_mw < 0:
            raise ValueError(
                "total_forecast_mw cannot be negative."
            )

        total_capacity_mw = float(
            sum(
                wind_capacity_mw.values()
            )
        )

        if total_capacity_mw <= 0:
            raise ValueError(
                "Total installed wind capacity must be "
                "greater than zero."
            )

        available = {}

        for generator_id, capacity_mw in (
            wind_capacity_mw.items()
        ):

            capacity = float(
                capacity_mw
            )

            share = (
                capacity
                / total_capacity_mw
            )

            available[generator_id] = (
                total_forecast_mw
                * share
            )

        return available

    # ==================================================================
    # INPUT VALIDATION
    # ==================================================================

    def _validate_inputs(
        self,
        *,
        observed_wind_mw: Sequence[float],
        snapshot: str,
        demand_mw: Dict[str, float],
        wind_capacity_mw: Dict[str, float],
        wind_bus: Dict[str, str],
    ) -> None:
        """
        Validate all inputs before forecasting or optimisation.
        """

        # --------------------------------------------------------------
        # Observations
        # --------------------------------------------------------------

        if observed_wind_mw is None:
            raise ValueError(
                "observed_wind_mw cannot be None."
            )

        if len(observed_wind_mw) == 0:
            raise ValueError(
                "observed_wind_mw cannot be empty."
            )

        for value in observed_wind_mw:

            numeric_value = float(value)

            if not np.isfinite(
                numeric_value
            ):
                raise ValueError(
                    "observed_wind_mw contains a "
                    "non-finite value."
                )

            if numeric_value < 0:
                raise ValueError(
                    "observed_wind_mw cannot contain "
                    "negative values."
                )

        # --------------------------------------------------------------
        # Snapshot
        # --------------------------------------------------------------

        if not isinstance(
            snapshot,
            str,
        ) or not snapshot.strip():

            raise ValueError(
                "snapshot must be a non-empty string."
            )

        # --------------------------------------------------------------
        # Demand
        # --------------------------------------------------------------

        if demand_mw is None:
            raise ValueError(
                "demand_mw cannot be None."
            )

        for bus, demand in demand_mw.items():

            if bus not in self.network.buses:
                raise ValueError(
                    f"Demand references unknown bus "
                    f"'{bus}'."
                )

            value = float(demand)

            if not np.isfinite(value):
                raise ValueError(
                    f"Demand at bus '{bus}' must be finite."
                )

            if value < 0:
                raise ValueError(
                    f"Demand at bus '{bus}' cannot be negative."
                )

        # --------------------------------------------------------------
        # Wind capacity
        # --------------------------------------------------------------

        if not wind_capacity_mw:
            raise ValueError(
                "wind_capacity_mw cannot be empty."
            )

        for generator_id, capacity in (
            wind_capacity_mw.items()
        ):

            value = float(capacity)

            if not np.isfinite(value):
                raise ValueError(
                    f"Wind capacity for generator "
                    f"'{generator_id}' must be finite."
                )

            if value < 0:
                raise ValueError(
                    f"Wind capacity for generator "
                    f"'{generator_id}' cannot be negative."
                )

        # --------------------------------------------------------------
        # Wind-bus mapping
        # --------------------------------------------------------------

        for generator_id in wind_capacity_mw:

            if generator_id not in wind_bus:
                raise ValueError(
                    f"No bus mapping supplied for wind "
                    f"generator '{generator_id}'."
                )

            bus = wind_bus[
                generator_id
            ]

            if bus not in self.network.buses:
                raise ValueError(
                    f"Wind generator '{generator_id}' "
                    f"references unknown bus '{bus}'."
                )


# ----------------------------------------------------------------------
# Backwards-compatible functional interface
# ----------------------------------------------------------------------

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
    forecast_function: ForecastFunction | None = None,
    conventional_generation_mw: Dict[str, float] | None = None,
    interconnector_dispatch_mw: Dict[str, float] | None = None,
    interconnectors: Dict[str, Dict[str, float]] | None = None,
) -> OptimiserResult:
    """
    Functional wrapper around ForecastToGrid.

    The 'lines' argument is retained for compatibility with the
    previous forecast_to_grid.py interface.

    The authoritative line definitions are taken from the supplied
    LinearNetwork because WindCurtailmentOptimizer operates on that
    network object.
    """

    # --------------------------------------------------------------
    # The previous API accepted 'lines' separately.
    #
    # The LinearNetwork already contains the authoritative lines,
    # so we deliberately do not reconstruct another network here.
    #
    # This prevents two potentially inconsistent line definitions
    # from existing inside the same optimisation run.
    # --------------------------------------------------------------

    del lines

    pipeline = ForecastToGrid(
        network=network,
        forecast_function=forecast_function,
    )

    return pipeline.run(
        observed_wind_mw=observed_wind_mw,
        snapshot=snapshot,
        demand_mw=demand_mw,
        wind_capacity_mw=wind_capacity_mw,
        wind_bus=wind_bus,
        scenario=scenario,
        conventional_generation_mw=(
            conventional_generation_mw
        ),
        interconnector_dispatch_mw=(
            interconnector_dispatch_mw
        ),
        interconnectors=interconnectors,
    )


__all__ = [
    "ForecastFunction",
    "ForecastToGrid",
    "persistence_forecast_function",
    "run_forecast_to_grid",
]