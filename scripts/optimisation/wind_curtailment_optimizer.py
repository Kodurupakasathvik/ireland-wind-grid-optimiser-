"""
Linear wind-curtailment optimiser.

Purpose
-------
Determine the maximum amount of available wind generation that can be
accepted by a simplified transmission network while respecting line
thermal limits.

This is the first MVP optimisation engine.

The model uses a reference bus as the balancing bus. The reference bus
absorbs the net system imbalance so that the linear network model always
receives a balanced injection vector.

Mathematical problem
--------------------

For each wind generator i:

    0 <= P_accepted_i <= P_available_i

Objective:

    maximise sum(P_accepted_i)

Equivalent:

    minimise total wind curtailment

Network:

    P = B * theta

    F_l = b_l * (theta_from - theta_to)

Thermal constraints:

    -F_limit_l <= F_l <= F_limit_l
"""

from typing import Dict

import numpy as np
from scipy.optimize import linprog

from scripts.optimisation.linear_model import LinearNetwork
from scripts.optimisation.optimiser_types import (
    OptimiserInput,
    OptimiserResult,
)


class WindCurtailmentOptimizer:
    """
    Linear programming optimiser for wind acceptance.

    MVP assumptions:

    - one reference/slack bus
    - fixed demand
    - fixed conventional generation
    - fixed interconnector dispatch
    - variable wind generation
    - linear transmission constraints
    - reference bus provides balancing

    The optimiser maximises total accepted wind.
    """

    def __init__(
        self,
        network: LinearNetwork,
    ) -> None:

        self.network = network
        self.wind_generators = []
        self.wind_bus = {}

    def solve(
        self,
        optimiser_input: OptimiserInput,
        wind_bus: Dict[str, str],
    ) -> OptimiserResult:
        """
        Solve the wind-curtailment optimisation problem.

        Parameters
        ----------
        optimiser_input:
            Available wind, demand, network information and scenario.

        wind_bus:
            Mapping:

                wind_generator_id -> bus_id

        Returns
        -------
        OptimiserResult
        """

        self._validate_inputs(
            optimiser_input,
            wind_bus,
        )

        self.wind_generators = list(
            optimiser_input.available_wind_mw.keys()
        )

        self.wind_bus = wind_bus

        # ------------------------------------------------------------
        # Zero-wind case
        # ------------------------------------------------------------

        if not self.wind_generators:

            return OptimiserResult(
                status="optimal",
                available_wind_total_mw=0.0,
                accepted_wind_total_mw=0.0,
                curtailment_total_mw=0.0,
                curtailment_percentage=0.0,
                scenario=optimiser_input.scenario,
                objective_value_mw=0.0,
                message="No wind generators were provided.",
            )

        n_wind = len(self.wind_generators)

        # ------------------------------------------------------------
        # Objective
        #
        # scipy.optimize.linprog minimises.
        #
        # We want:
        #
        #     maximise sum(accepted wind)
        #
        # Therefore:
        #
        #     minimise -sum(accepted wind)
        # ------------------------------------------------------------

        objective = -np.ones(
            n_wind,
            dtype=float,
        )

        # ------------------------------------------------------------
        # Variable bounds
        #
        # 0 <= accepted wind <= available wind
        # ------------------------------------------------------------

        bounds = [
            (
                0.0,
                optimiser_input.available_wind_mw[
                    wind_id
                ],
            )
            for wind_id in self.wind_generators
        ]

        # ------------------------------------------------------------
        # Fixed injections
        #
        # These contain demand, conventional generation and fixed
        # interconnector dispatch.
        #
        # The reference bus is used to balance these injections.
        # ------------------------------------------------------------

        fixed_injections = self._build_fixed_injections(
            optimiser_input
        )

        # ------------------------------------------------------------
        # Flow sensitivities
        #
        # For each wind generator:
        #
        #     +1 MW at wind bus
        #     -1 MW at reference bus
        #
        # This gives the change in each line flow caused by 1 MW
        # of accepted wind.
        # ------------------------------------------------------------

        flow_sensitivities = (
            self._build_flow_sensitivities()
        )

        # ------------------------------------------------------------
        # Fixed network flows
        # ------------------------------------------------------------

        fixed_flows = (
            self.network.calculate_line_flows(
                fixed_injections
            )
        )

        # ------------------------------------------------------------
        # Thermal constraints
        #
        # For every line:
        #
        #     fixed_flow + sensitivity * wind <= limit
        #
        #     fixed_flow + sensitivity * wind >= -limit
        # ------------------------------------------------------------

        a_ub = []
        b_ub = []

        for line in self.network.lines:

            line_id = line.line_id

            sensitivity = (
                flow_sensitivities[line_id]
            )

            fixed_flow = fixed_flows[line_id]

            # Upper limit
            a_ub.append(sensitivity)

            b_ub.append(
                line.limit_mw - fixed_flow
            )

            # Lower limit
            a_ub.append(-sensitivity)

            b_ub.append(
                line.limit_mw + fixed_flow
            )

        # ------------------------------------------------------------
        # Solve linear program
        # ------------------------------------------------------------

        result = linprog(
            c=objective,
            A_ub=np.array(a_ub),
            b_ub=np.array(b_ub),
            bounds=bounds,
            method="highs",
        )

        # ------------------------------------------------------------
        # Solver failure
        # ------------------------------------------------------------

        if not result.success:

            available_total = sum(
                optimiser_input.available_wind_mw.values()
            )

            return OptimiserResult(
                status="infeasible",
                available_wind_total_mw=available_total,
                accepted_wind_total_mw=0.0,
                curtailment_total_mw=available_total,
                curtailment_percentage=(
                    100.0
                    if available_total > 0
                    else 0.0
                ),
                scenario=optimiser_input.scenario,
                message=result.message,
            )

        # ------------------------------------------------------------
        # Accepted wind
        # ------------------------------------------------------------

        accepted = {
            wind_id: float(result.x[index])
            for index, wind_id in enumerate(
                self.wind_generators
            )
        }

        available_total = sum(
            optimiser_input.available_wind_mw.values()
        )

        accepted_total = sum(
            accepted.values()
        )

        # ------------------------------------------------------------
        # Curtailment
        # ------------------------------------------------------------

        curtailed = {
            wind_id: max(
                0.0,
                optimiser_input.available_wind_mw[
                    wind_id
                ]
                - accepted[wind_id],
            )
            for wind_id in self.wind_generators
        }

        curtailment_total = (
            available_total
            - accepted_total
        )

        curtailment_percentage = (
            100.0 * curtailment_total / available_total
            if available_total > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Final network injections
        #
        # Start with fixed injections, add accepted wind, then
        # rebalance the reference bus.
        # ------------------------------------------------------------

        final_injections = dict(
            fixed_injections
        )

        for wind_id, accepted_mw in accepted.items():

            bus = wind_bus[wind_id]

            final_injections[bus] = (
                final_injections.get(bus, 0.0)
                + accepted_mw
            )

        final_injections = (
            self._rebalance_reference_bus(
                final_injections
            )
        )

        # ------------------------------------------------------------
        # Final line flows
        # ------------------------------------------------------------

        line_flows = (
            self.network.calculate_line_flows(
                final_injections
            )
        )

        # ------------------------------------------------------------
        # Binding constraints
        # ------------------------------------------------------------

        binding_constraints = []

        for line in self.network.lines:

            flow = abs(
                line_flows[line.line_id]
            )

            if flow >= line.limit_mw - 1e-6:

                binding_constraints.append(
                    line.line_id
                )

        return OptimiserResult(
            status="optimal",
            available_wind_total_mw=available_total,
            accepted_wind_total_mw=accepted_total,
            curtailment_total_mw=curtailment_total,
            curtailment_percentage=curtailment_percentage,
            accepted_wind_by_generator_mw=accepted,
            curtailed_wind_by_generator_mw=curtailed,
            line_flows_mw=line_flows,
            binding_constraints=binding_constraints,
            scenario=optimiser_input.scenario,
            objective_value_mw=accepted_total,
            solve_time_seconds=None,
            message="Optimisation completed successfully.",
        )

    def _build_fixed_injections(
        self,
        optimiser_input: OptimiserInput,
    ) -> Dict[str, float]:
        """
        Build fixed injections excluding variable wind.

        Positive:
            generation

        Negative:
            demand

        The reference bus is then adjusted so that total injection
        equals zero.
        """

        injections = {
            bus: 0.0
            for bus in self.network.buses
        }

        # Conventional generation
        for bus, generation in (
            optimiser_input.conventional_generation_mw.items()
        ):

            injections[bus] = (
                injections.get(bus, 0.0)
                + generation
            )

        # Demand
        for bus, demand in (
            optimiser_input.demand_mw.items()
        ):

            injections[bus] = (
                injections.get(bus, 0.0)
                - demand
            )

        # Fixed interconnector dispatch
        for bus, dispatch in (
            optimiser_input.interconnector_dispatch_mw.items()
        ):

            injections[bus] = (
                injections.get(bus, 0.0)
                + dispatch
            )

        return self._rebalance_reference_bus(
            injections
        )

    def _rebalance_reference_bus(
        self,
        injections: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Adjust the reference bus so that total network injection
        equals zero.

        The reference bus acts as the balancing/slack bus in the
        MVP linear model.
        """

        balanced = {
            bus: float(injections.get(bus, 0.0))
            for bus in self.network.buses
        }

        non_reference_total = sum(
            value
            for bus, value in balanced.items()
            if bus != self.network.reference_bus
        )

        balanced[
            self.network.reference_bus
        ] = -non_reference_total

        return balanced

    def _build_flow_sensitivities(self):
        """
        Build line-flow sensitivity to each wind generator.

        For each wind generator:

            +1 MW at wind bus
            -1 MW at reference bus

        This creates a balanced perturbation.
        """

        sensitivities = {
            line.line_id: []
            for line in self.network.lines
        }

        for wind_id in self.wind_generators:

            bus = self.wind_bus[wind_id]

            injections = {
                network_bus: 0.0
                for network_bus in self.network.buses
            }

            injections[bus] += 1.0

            injections[
                self.network.reference_bus
            ] -= 1.0

            flows = (
                self.network.calculate_line_flows(
                    injections
                )
            )

            for line in self.network.lines:

                sensitivities[
                    line.line_id
                ].append(
                    flows[line.line_id]
                )

        return {
            line_id: np.array(
                coefficients,
                dtype=float,
            )
            for line_id, coefficients
            in sensitivities.items()
        }

    def _validate_inputs(
        self,
        optimiser_input: OptimiserInput,
        wind_bus: Dict[str, str],
    ) -> None:
        """
        Validate optimiser inputs before solving.
        """

        # ------------------------------------------------------------
        # Wind validation
        # ------------------------------------------------------------

        for wind_id in (
            optimiser_input.available_wind_mw
        ):

            if wind_id not in wind_bus:

                raise ValueError(
                    f"No bus mapping supplied for "
                    f"wind generator '{wind_id}'."
                )

            bus = wind_bus[wind_id]

            if bus not in self.network.buses:

                raise ValueError(
                    f"Wind generator '{wind_id}' "
                    f"references unknown bus '{bus}'."
                )

            available = (
                optimiser_input.available_wind_mw[
                    wind_id
                ]
            )

            capacity = (
                optimiser_input.wind_capacity_mw[
                    wind_id
                ]
            )

            if available < 0:

                raise ValueError(
                    f"Available wind for '{wind_id}' "
                    f"cannot be negative."
                )

            if capacity < 0:

                raise ValueError(
                    f"Installed capacity for '{wind_id}' "
                    f"cannot be negative."
                )

            if available > capacity + 1e-9:

                raise ValueError(
                    f"Available wind for '{wind_id}' "
                    f"cannot exceed installed capacity."
                )

        # ------------------------------------------------------------
        # Bus validation
        # ------------------------------------------------------------

        for bus in optimiser_input.demand_mw:

            if bus not in self.network.buses:

                raise ValueError(
                    f"Demand references unknown bus '{bus}'."
                )

        for bus in (
            optimiser_input.conventional_generation_mw
        ):

            if bus not in self.network.buses:

                raise ValueError(
                    "Conventional generation references "
                    f"unknown bus '{bus}'."
                )

        for bus in (
            optimiser_input.interconnector_dispatch_mw
        ):

            if bus not in self.network.buses:

                raise ValueError(
                    "Interconnector dispatch references "
                    f"unknown bus '{bus}'."
                )