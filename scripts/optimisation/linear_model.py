"""
Linear transmission network model for the MVP optimiser.

This module intentionally implements a small, transparent DC/PTDF-style
network model.

It is NOT the Irish 58-bus model.

Purpose:
    1. Represent a simple transmission network.
    2. Convert bus injections into line flows.
    3. Provide thermal limits for the optimiser.
    4. Provide a deterministic test system before connecting real data.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class LinearLine:
    """
    Representation of one transmission line.

    Parameters
    ----------
    line_id:
        Unique line identifier.

    from_bus:
        Sending-end bus.

    to_bus:
        Receiving-end bus.

    susceptance:
        Simplified line susceptance used by the DC power-flow model.

    limit_mw:
        Absolute thermal flow limit in MW.
    """

    line_id: str
    from_bus: str
    to_bus: str
    susceptance: float
    limit_mw: float


class LinearNetwork:
    """
    Small linear/DC transmission network.

    The network solves:

        P = Bbus * theta

    and then:

        F_line = b * (theta_from - theta_to)

    One bus is selected as the reference/slack bus.
    """

    def __init__(
        self,
        buses: List[str],
        lines: List[LinearLine],
        reference_bus: str,
    ) -> None:

        if reference_bus not in buses:
            raise ValueError(
                f"Reference bus '{reference_bus}' is not in the network."
            )

        self.buses = list(buses)
        self.lines = list(lines)
        self.reference_bus = reference_bus

        self.bus_index = {
            bus: index
            for index, bus in enumerate(self.buses)
        }

        self._validate_lines()

        self.bbus = self._build_bbus()

    def _validate_lines(self) -> None:
        """Validate that all line buses exist and limits are positive."""

        for line in self.lines:

            if line.from_bus not in self.bus_index:
                raise ValueError(
                    f"Unknown from_bus '{line.from_bus}' "
                    f"for line '{line.line_id}'."
                )

            if line.to_bus not in self.bus_index:
                raise ValueError(
                    f"Unknown to_bus '{line.to_bus}' "
                    f"for line '{line.line_id}'."
                )

            if line.limit_mw <= 0:
                raise ValueError(
                    f"Line '{line.line_id}' must have "
                    f"a positive thermal limit."
                )

    def _build_bbus(self) -> np.ndarray:
        """
        Construct the bus susceptance matrix.

        For a line between i and j with susceptance b:

            B[i,i] += b
            B[j,j] += b
            B[i,j] -= b
            B[j,i] -= b
        """

        n_buses = len(self.buses)

        bbus = np.zeros(
            (n_buses, n_buses),
            dtype=float,
        )

        for line in self.lines:

            i = self.bus_index[line.from_bus]
            j = self.bus_index[line.to_bus]
            b = line.susceptance

            bbus[i, i] += b
            bbus[j, j] += b

            bbus[i, j] -= b
            bbus[j, i] -= b

        return bbus

    def solve_angles(
        self,
        injections_mw: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Solve bus voltage angles for a balanced injection vector.

        Positive injection:
            generation into the grid.

        Negative injection:
            demand/load.

        The total injection must equal zero.
        """

        unknown_buses = [
            bus
            for bus in self.buses
            if bus != self.reference_bus
        ]

        p = np.array(
            [
                injections_mw.get(bus, 0.0)
                for bus in self.buses
            ],
            dtype=float,
        )

        total_injection = float(np.sum(p))

        if not np.isclose(
            total_injection,
            0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Network injections must balance to zero. "
                f"Total injection = {total_injection:.6f} MW."
            )

        unknown_indices = [
            self.bus_index[bus]
            for bus in unknown_buses
        ]

        reduced_bbus = self.bbus[
            np.ix_(
                unknown_indices,
                unknown_indices,
            )
        ]

        reduced_p = p[unknown_indices]

        try:
            theta_unknown = np.linalg.solve(
                reduced_bbus,
                reduced_p,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Unable to solve linear network. "
                "The network may be disconnected or singular."
            ) from exc

        theta = np.zeros(
            len(self.buses),
            dtype=float,
        )

        theta[unknown_indices] = theta_unknown

        return {
            bus: float(theta[self.bus_index[bus]])
            for bus in self.buses
        }

    def calculate_line_flows(
        self,
        injections_mw: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate linear transmission flows.

        Flow convention:

            positive = from_bus -> to_bus
            negative = to_bus -> from_bus
        """

        theta = self.solve_angles(injections_mw)

        flows = {}

        for line in self.lines:

            flow = (
                line.susceptance
                * (
                    theta[line.from_bus]
                    - theta[line.to_bus]
                )
            )

            flows[line.line_id] = float(flow)

        return flows

    def check_thermal_limits(
        self,
        line_flows_mw: Dict[str, float],
        tolerance_mw: float = 1e-6,
    ) -> Dict[str, bool]:
        """
        Check whether each line remains within its thermal limit.
        """

        limits = {
            line.line_id: line.limit_mw
            for line in self.lines
        }

        return {
            line_id: abs(flow)
            <= limits[line_id] + tolerance_mw
            for line_id, flow in line_flows_mw.items()
        }

    def get_overloaded_lines(
        self,
        line_flows_mw: Dict[str, float],
        tolerance_mw: float = 1e-6,
    ) -> List[str]:
        """
        Return line IDs whose absolute flow exceeds their limit.
        """

        limits = {
            line.line_id: line.limit_mw
            for line in self.lines
        }

        return [
            line_id
            for line_id, flow in line_flows_mw.items()
            if abs(flow) > limits[line_id] + tolerance_mw
        ]