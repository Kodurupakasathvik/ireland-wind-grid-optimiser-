"""
Production forecast -> grid optimiser -> AC validation
for ALL production horizons.

Pipeline:
    1. Read production_wind_forecast.csv
    2. For each horizon:
        - convert system forecast to available wind
        - build simplified LinearNetwork
        - run WindCurtailmentOptimizer
        - write accepted wind back to PyPSA
        - run AC power flow
        - classify SECURE/INSECURE
    3. Save combined curtailment + security tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pypsa

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
from scripts.validation.ac_validator import (
    ACValidationResult,
    validate_pypsa_dispatch,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

FORECAST_FILE = PROCESSED / "production_wind_forecast.csv"
SCENARIO_FILE = PROCESSED / "selected_operating_scenarios.csv"
HOSTING_FILE = PROCESSED / "wind_hosting_capacity.csv"
PYPSA_NETWORK = PROCESSED / "eirgrid_scenarios.nc"
BASE_NETWORK = PROCESSED / "ireland_base_scenario.nc"

OUTPUT_SECURITY = PROCESSED / "production_multi_horizon_security.csv"
OUTPUT_CURTAILMENT = PROCESSED / "production_multi_horizon_curtailment.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose one existing operating scenario for demand allocation
SCENARIO_NAME = "S1_NORMAL"

# Reference/slack bus from the PyPSA network
REFERENCE_BUS = "way/88462768-220"

# Wind generator IDs from eirgrid_scenarios.nc
WIND_GENERATORS = {
    "eirgrid_wind_way/88462768-220": "way/88462768-220",
    "eirgrid_wind_way/104388595-220": "way/104388595-220",
    "eirgrid_wind_way/516651650-220": "way/516651650-220",
    "eirgrid_wind_way/88144450-220": "way/88144450-220",
    "eirgrid_wind_way/254158424-220": "way/254158424-220",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_wind_capacities_mw() -> Dict[str, float]:
    """
    Map each wind generator ID to installed capacity using
    wind_hosting_capacity.csv.
    """

    if not HOSTING_FILE.exists():
        raise FileNotFoundError(
            f"Wind hosting capacity file not found: {HOSTING_FILE}"
        )

    hosting = pd.read_csv(HOSTING_FILE)

    capacity_by_bus = dict(
        zip(
            hosting["bus"],
            hosting["hosting_capacity_mw"],
        )
    )

    capacities = {}

    for gen_id, bus in WIND_GENERATORS.items():

        capacity = capacity_by_bus.get(bus)

        if capacity is None or capacity <= 0:
            print(
                f"WARNING: No hosting capacity for {bus}. "
                f"Using default 500.0 MW."
            )
            capacity = 500.0

        capacities[gen_id] = float(capacity)

    return capacities


def allocate_available_wind(
    total_available_mw: float,
    wind_capacity_mw: Dict[str, float],
) -> Dict[str, float]:
    """
    Allocate system forecast proportionally to installed capacity.
    """

    total_capacity = sum(wind_capacity_mw.values())

    if total_capacity <= 0:
        raise ValueError("Total wind capacity must be positive.")

    allocated = {}

    for gen_id, capacity in wind_capacity_mw.items():
        share = capacity / total_capacity
        allocated[gen_id] = total_available_mw * share

    return allocated


def load_demand_mw(
    scenario_name: str = SCENARIO_NAME,
) -> Dict[str, float]:
    """
    Read per-bus demand directly from the PyPSA scenario snapshot.
    """

    if not PYPSA_NETWORK.exists():
        raise FileNotFoundError(
            f"PyPSA network not found: {PYPSA_NETWORK}"
        )

    n = pypsa.Network(str(PYPSA_NETWORK))

    loads_p = n.loads_t.p_set.loc[scenario_name]

    demand_mw = {}

    for load_name, p in loads_p.items():
        bus = n.loads.at[load_name, "bus"]
        demand_mw[bus] = demand_mw.get(bus, 0.0) + float(p)

    return demand_mw


def build_balanced_conventional_generation(
    demand_mw: Dict[str, float],
) -> Dict[str, float]:
    """
    Create a balanced conventional-generation profile equal to demand
    at each bus. This ensures zero net fixed injections before wind,
    making the LP optimiser feasible.
    """

    return dict(demand_mw)


def build_linear_network_from_pypsa() -> LinearNetwork:
    """
    Build a simplified LinearNetwork using PyPSA lines AND transformers.
    """

    if not PYPSA_NETWORK.exists():
        raise FileNotFoundError(
            f"PyPSA network not found: {PYPSA_NETWORK}"
        )

    n = pypsa.Network(str(PYPSA_NETWORK))

    buses = list(n.buses.index)
    lines = []

    # AC lines
    for line_id, line in n.lines.iterrows():
        x = float(line["x"])
        susceptance = 1.0 / x if x > 0 else 1.0

        lines.append(
            LinearLine(
                line_id=line_id,
                from_bus=line["bus0"],
                to_bus=line["bus1"],
                susceptance=susceptance,
                limit_mw=float(line["s_nom"]),
            )
        )

    # Transformers as equivalent lines
    for trafo_id, trafo in n.transformers.iterrows():
        x = float(trafo["x"])
        susceptance = 1.0 / x if x > 0 else 1.0

        lines.append(
            LinearLine(
                line_id=f"trafo_{trafo_id}",
                from_bus=trafo["bus0"],
                to_bus=trafo["bus1"],
                susceptance=susceptance,
                limit_mw=float(trafo["s_nom"]),
            )
        )

    return LinearNetwork(
        buses=buses,
        lines=lines,
        reference_bus=REFERENCE_BUS,
    )


def run_ac_validation(
    accepted_wind_mw: Dict[str, float],
    demand_mw: Dict[str, float],
    wind_capacity_mw: Dict[str, float],
) -> ACValidationResult:
    """Run the shared nonlinear AC security-validation adapter."""

    return validate_pypsa_dispatch(
        PYPSA_NETWORK,
        snapshot="PRODUCTION",
        scenario="production",
        demand_mw=demand_mw,
        accepted_wind_mw=accepted_wind_mw,
        wind_capacity_mw=wind_capacity_mw,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("PRODUCTION FORECAST -> OPTIMISER -> AC VALIDATION")
    print("           MULTI-HORIZON RUN")
    print("=" * 80)
    print()

    # Load production forecast all horizons
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Production forecast not found: {FORECAST_FILE}"
        )

    forecast_df = pd.read_csv(FORECAST_FILE)

    horizons = forecast_df["horizon"].unique().tolist()

    print(f"Horizons found: {horizons}\n")

    # Load static data once
    print("Loading wind capacities...")
    wind_capacity_mw = load_wind_capacities_mw()
    print(
        f"Total installed wind capacity: "
        f"{sum(wind_capacity_mw.values()):.2f} MW\n"
    )

    print("Loading demand...")
    demand_mw = load_demand_mw()
    print(
        f"Total demand: {sum(demand_mw.values()):.2f} MW\n"
    )

    print("Building balanced conventional generation...")
    conventional_generation_mw = build_balanced_conventional_generation(
        demand_mw
    )
    print(
        f"Total conventional generation: "
        f"{sum(conventional_generation_mw.values()):.2f} MW\n"
    )

    print("Building linear network...")
    linear_network = build_linear_network_from_pypsa()
    print(
        f"Linear network: {len(linear_network.buses)} buses, "
        f"{len(linear_network.lines)} lines\n"
    )

    # Collect results
    all_results: List[dict] = []
    all_security: List[dict] = []

    for horizon in horizons:

        print("\n" + "=" * 80)
        print(f"PROCESSING HORIZON: {horizon}")
        print("=" * 80)

        # Get forecast value for this horizon
        row = forecast_df[forecast_df["horizon"] == horizon].iloc[0]
        total_available_mw = float(row["forecast_wind_generation_mw"])

        print(f"Forecast available wind: {total_available_mw:.4f} MW")

        # Allocate to generators
        available_wind_mw = allocate_available_wind(
            total_available_mw,
            wind_capacity_mw,
        )

        # Build optimiser input
        optimiser_input = OptimiserInput(
            snapshot="PRODUCTION",
            demand_mw=demand_mw,
            available_wind_mw=available_wind_mw,
            wind_capacity_mw=wind_capacity_mw,
            conventional_generation_mw=conventional_generation_mw,
            lines={},
            scenario="production",
        )

        # Run LP optimiser
        optimiser = WindCurtailmentOptimizer(linear_network)
        optimiser_result = optimiser.solve(
            optimiser_input,
            wind_bus=WIND_GENERATORS,
        )

        print(f"Optimiser status: {optimiser_result.status}")
        print(
            f"Accepted wind: "
            f"{optimiser_result.accepted_wind_total_mw:.4f} MW"
        )
        print(
            f"Curtailment: "
            f"{optimiser_result.curtailment_total_mw:.4f} MW "
            f"({optimiser_result.curtailment_percentage:.2f}%)"
        )

        # AC validation
        ac_result = run_ac_validation(
            accepted_wind_mw=(
                optimiser_result.accepted_wind_by_generator_mw
            ),
            demand_mw=demand_mw,
            wind_capacity_mw=wind_capacity_mw,
        )

        print(f"AC converged: {ac_result.converged}")
        print(f"Min voltage: {ac_result.minimum_voltage_pu:.4f} pu")
        print(
            f"Max line loading: "
            f"{ac_result.maximum_line_loading_percent:.2f}%"
        )
        print(
            f"Max transformer loading: "
            f"{ac_result.maximum_transformer_loading_percent:.2f}%"
        )
        print(f"Secure: {ac_result.physically_secure}")
        print(f"Message: {ac_result.message}")

        # Store curtailment result
        all_results.append({
            "horizon": horizon,
            "forecast_wind_mw": total_available_mw,
            "available_wind_total_mw": (
                optimiser_result.available_wind_total_mw
            ),
            "accepted_wind_total_mw": (
                optimiser_result.accepted_wind_total_mw
            ),
            "curtailment_total_mw": (
                optimiser_result.curtailment_total_mw
            ),
            "curtailment_percentage": (
                optimiser_result.curtailment_percentage
            ),
            "status": optimiser_result.status,
            "physically_secure": ac_result.physically_secure,
        })

        # Store security result
        all_security.append({
            "horizon": horizon,
            "converged": ac_result.converged,
            "minimum_voltage_pu": ac_result.minimum_voltage_pu,
            "maximum_line_loading_percent": (
                ac_result.maximum_line_loading_percent
            ),
            "maximum_transformer_loading_percent": (
                ac_result.maximum_transformer_loading_percent
            ),
            "overloaded_lines": ",".join(
                ac_result.overloaded_lines
            ),
            "overloaded_transformers": ",".join(
                ac_result.overloaded_transformers
            ),
            "physically_secure": ac_result.physically_secure,
            "message": ac_result.message,
        })

    # Save combined results
    curtailment_df = pd.DataFrame(all_results)
    security_df = pd.DataFrame(all_security)

    curtailment_df.to_csv(OUTPUT_CURTAILMENT, index=False)
    security_df.to_csv(OUTPUT_SECURITY, index=False)

    print("\n" + "=" * 80)
    print("MULTI-HORIZON RESULTS")
    print("=" * 80)
    print("\nCurtailment table:")
    print(curtailment_df.to_string(index=False))

    print("\nSecurity table:")
    print(security_df.to_string(index=False))

    print("\nSaved outputs:")
    print(f"  {OUTPUT_CURTAILMENT}")
    print(f"  {OUTPUT_SECURITY}")

    print("\nPRODUCTION MULTI-HORIZON PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
