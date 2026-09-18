"""
Optimiser input/output data structures.

This module defines the interface between:

    forecast -> optimisation -> validation

The optimisation mathematics will be implemented in later modules.

All power quantities are expressed in MW unless otherwise stated.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OptimiserInput:
    """
    Input data for one optimisation operating condition.
    """

    # Operating condition
    snapshot: str

    # System demand by bus
    demand_mw: Dict[str, float]

    # Forecast/available wind by wind-generator ID
    available_wind_mw: Dict[str, float]

    # Installed wind capacity by wind-generator ID
    wind_capacity_mw: Dict[str, float]

    # Transmission lines
    #
    # Example:
    # {
    #     "line_1": {
    #         "from_bus": "bus_1",
    #         "to_bus": "bus_2",
    #         "limit_mw": 300.0
    #     }
    # }
    lines: Dict[str, Dict[str, object]]

    # Interconnector limits
    interconnectors: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    # Existing non-wind generation
    conventional_generation_mw: Dict[str, float] = field(
        default_factory=dict
    )

    # Optional fixed interconnector dispatch
    interconnector_dispatch_mw: Dict[str, float] = field(
        default_factory=dict
    )

    # Optimisation scenario
    scenario: str = "existing"


@dataclass
class OptimiserResult:
    """
    Output produced by the wind-curtailment optimiser.
    """

    # Solver information
    status: str

    # Main wind quantities
    available_wind_total_mw: float
    accepted_wind_total_mw: float
    curtailment_total_mw: float
    curtailment_percentage: float

    # Generator-level results
    accepted_wind_by_generator_mw: Dict[str, float] = field(
        default_factory=dict
    )

    curtailed_wind_by_generator_mw: Dict[str, float] = field(
        default_factory=dict
    )

    # Network results
    line_flows_mw: Dict[str, float] = field(
        default_factory=dict
    )

    # Constraints at or near their limits
    binding_constraints: List[str] = field(
        default_factory=list
    )

    # Scenario used
    scenario: str = "existing"

    # Optional solver diagnostics
    objective_value_mw: Optional[float] = None
    solve_time_seconds: Optional[float] = None

    # Human-readable message
    message: str = ""