"""
Scenario comparison / historical validation.

Compare the production forecast pipeline result against
actual historical wind availability from selected_operating_scenarios.csv.

For each scenario:
    - use actual IE_Wind_Availability_MW
    - use actual IE_Demand_MW
    - run LP optimiser + AC validation
    - record curtailment and security

A forecast row is included for the 15-minute production forecast.
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
from scripts.optimisation.optimiser_types import OptimiserInput
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

OUTPUT_COMPARISON = PROCESSED / "scenario_comparison_forecast_vs_actual.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

REFERENCE_BUS = "way/88462768-220"

WIND_GENERATORS = {
    "eirgrid_wind_way/88462768-220": "way/88462768-220",
    "eirgrid_wind_way/104388595-220": "way/104388595-220",
    "eirgrid_wind_way/516651650-220": "way/516651650-220",
    "eirgrid_wind_way/88144450-220": "way/88144450-220",
    "eirgrid_wind_way/254158424-220": "way/254158424-220",
}


# =============================================================================
# HELPERS
# =============================================================================

def load_wind_capacities_mw() -> Dict[str, float]:
    hosting = pd.read_csv(HOSTING_FILE)
    capacity_by_bus = dict(zip(hosting["bus"], hosting["hosting_capacity_mw"]))
    capacities = {}
    for gen_id, bus in WIND_GENERATORS.items():
        cap = capacity_by_bus.get(bus, 500.0)
        capacities[gen_id] = float(cap if cap > 0 else 500.0)
    return capacities


def allocate_available_wind(total_available_mw, wind_capacity_mw):
    total_capacity = sum(wind_capacity_mw.values())
    if total_capacity <= 0:
        raise ValueError("Total wind capacity must be positive.")
    return {
        gen_id: total_available_mw * (capacity / total_capacity)
        for gen_id, capacity in wind_capacity_mw.items()
    }


def load_demand_mw(scenario_name: str) -> Dict[str, float]:
    n = pypsa.Network(str(PYPSA_NETWORK))
    if scenario_name not in n.snapshots:
        raise ValueError(
            f"Scenario '{scenario_name}' not found in PyPSA network."
        )
    loads_p = n.loads_t.p_set.loc[scenario_name]
    demand_mw = {}
    for load_name, p in loads_p.items():
        bus = n.loads.at[load_name, "bus"]
        demand_mw[bus] = demand_mw.get(bus, 0.0) + float(p)
    return demand_mw


def build_linear_network_from_pypsa() -> LinearNetwork:
    n = pypsa.Network(str(PYPSA_NETWORK))
    buses = list(n.buses.index)
    lines = []

    for line_id, line in n.lines.iterrows():
        x = float(line["x"])
        b = 1.0 / x if x > 0 else 1.0
        lines.append(
            LinearLine(
                line_id=line_id,
                from_bus=line["bus0"],
                to_bus=line["bus1"],
                susceptance=b,
                limit_mw=float(line["s_nom"]),
            )
        )

    for trafo_id, trafo in n.transformers.iterrows():
        x = float(trafo["x"])
        b = 1.0 / x if x > 0 else 1.0
        lines.append(
            LinearLine(
                line_id=f"trafo_{trafo_id}",
                from_bus=trafo["bus0"],
                to_bus=trafo["bus1"],
                susceptance=b,
                limit_mw=float(trafo["s_nom"]),
            )
        )

    return LinearNetwork(buses=buses, lines=lines, reference_bus=REFERENCE_BUS)


def run_ac_validation(
    accepted_wind_mw: Dict[str, float],
    demand_mw: Dict[str, float],
    wind_capacity_mw: Dict[str, float],
) -> ACValidationResult:
    return validate_pypsa_dispatch(
        PYPSA_NETWORK,
        snapshot="COMPARISON",
        scenario="comparison",
        demand_mw=demand_mw,
        accepted_wind_mw=accepted_wind_mw,
        wind_capacity_mw=wind_capacity_mw,
    )


def run_optimiser_for_wind(
    available_wind_total_mw: float,
    demand_mw: Dict[str, float],
    wind_capacity_mw: Dict[str, float],
    network: LinearNetwork,
):
    available_wind_mw = allocate_available_wind(
        available_wind_total_mw, wind_capacity_mw
    )
    conventional_generation_mw = dict(demand_mw)

    optimiser_input = OptimiserInput(
        snapshot="COMPARISON",
        demand_mw=demand_mw,
        available_wind_mw=available_wind_mw,
        wind_capacity_mw=wind_capacity_mw,
        conventional_generation_mw=conventional_generation_mw,
        lines={},
        scenario="comparison",
    )

    optimiser = WindCurtailmentOptimizer(network)
    optimiser_result = optimiser.solve(optimiser_input, wind_bus=WIND_GENERATORS)

    ac_result = run_ac_validation(
        optimiser_result.accepted_wind_by_generator_mw,
        demand_mw,
        wind_capacity_mw,
    )

    return optimiser_result, ac_result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO COMPARISON: FORECAST vs ACTUAL")
    print("=" * 80)

    # Load static data
    wind_capacity_mw = load_wind_capacities_mw()
    network = build_linear_network_from_pypsa()

    print("Wind capacity loaded.")
    print("Linear network built.")

    # Load forecast value for 15min
    forecast_df = pd.read_csv(FORECAST_FILE)
    forecast_row = forecast_df[forecast_df["horizon"] == "15min"].iloc[0]
    forecast_available_mw = float(forecast_row["forecast_wind_generation_mw"])

    # Use S1_NORMAL demand for forecast comparison
    forecast_demand_mw = load_demand_mw("S1_NORMAL")

    print("\nRunning forecast row...")
    forecast_opt, forecast_ac = run_optimiser_for_wind(
        forecast_available_mw,
        forecast_demand_mw,
        wind_capacity_mw,
        network,
    )

    results = [{
        "source": "forecast_15min",
        "horizon": "15min",
        "available_wind_mw": forecast_available_mw,
        "demand_mw": sum(forecast_demand_mw.values()),
        "accepted_wind_mw": forecast_opt.accepted_wind_total_mw,
        "curtailment_mw": forecast_opt.curtailment_total_mw,
        "curtailment_percent": forecast_opt.curtailment_percentage,
        "optimiser_status": forecast_opt.status,
        "ac_converged": forecast_ac.converged,
        "minimum_voltage_pu": forecast_ac.minimum_voltage_pu,
        "maximum_line_loading_percent": forecast_ac.maximum_line_loading_percent,
        "maximum_transformer_loading_percent": (
            forecast_ac.maximum_transformer_loading_percent
        ),
        "physically_secure": forecast_ac.physically_secure,
    }]

    # Load actual scenario data
    scenarios = pd.read_csv(SCENARIO_FILE)

    # Only scenarios present in PyPSA network
    pypsa_scenarios = pypsa.Network(str(PYPSA_NETWORK)).snapshots.tolist()

    for _, scenario in scenarios.iterrows():
        scenario_name = scenario["Scenario"]
        if scenario_name not in pypsa_scenarios:
            continue

        actual_wind = float(scenario["IE_Wind_Availability_MW"])
        actual_demand_mw = load_demand_mw(scenario_name)

        print(f"\nRunning scenario {scenario_name}...")
        opt, ac = run_optimiser_for_wind(
            actual_wind,
            actual_demand_mw,
            wind_capacity_mw,
            network,
        )

        results.append({
            "source": "actual_scenario",
            "horizon": scenario_name,
            "available_wind_mw": actual_wind,
            "demand_mw": sum(actual_demand_mw.values()),
            "accepted_wind_mw": opt.accepted_wind_total_mw,
            "curtailment_mw": opt.curtailment_total_mw,
            "curtailment_percent": opt.curtailment_percentage,
            "optimiser_status": opt.status,
            "ac_converged": ac.converged,
            "minimum_voltage_pu": ac.minimum_voltage_pu,
            "maximum_line_loading_percent": ac.maximum_line_loading_percent,
            "maximum_transformer_loading_percent": (
                ac.maximum_transformer_loading_percent
            ),
            "physically_secure": ac.physically_secure,
        })

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(OUTPUT_COMPARISON, index=False)

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_COMPARISON)
    print("\nSCENARIO COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
