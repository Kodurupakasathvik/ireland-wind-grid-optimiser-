"""
Historical counterfactual curtailment analysis.

For each real EirGrid operating scenario:
    - Use actual IE_Wind_Availability_MW and IE_Demand_MW
    - Run the LP wind-curtailment optimiser
    - Compute optimised curtailment
    - Compare against actual Wind_Not_Generated_MW
    - Screen the LP dispatch with nonlinear AC power flow
    - Credit energy, economic, and CO₂ values only when the dispatch passes
      the configured AC voltage and thermal-security checks

Methodological fix applied:
    conventional_generation_mw = {}
    so that the optimiser sees real demand at each non-reference bus,
    and the reference bus supplies that demand. No artificial
    conventional generation is assumed.

Assumptions:
    - Time interval = 15 minutes (0.25 h)
    - Wholesale price = 50 €/MWh (base case)
    - CO₂ emission factor = 0.4 tCO₂/MWh (base case)
    - Sensitivity analysis for price: 30, 50, 75, 100 €/MWh
    - Sensitivity analysis for CO₂ factor: 0.4, 0.5 tCO₂/MWh

Output:
    data/processed/counterfactual_analysis.csv

Important interpretation
------------------------
Each row represents one selected 15-minute operating snapshot.  The LP
curtailment reduction is reported separately from the AC-security-screened
result.  A dispatch that is AC-insecure or non-converged receives no energy,
economic, or emissions credit.
"""

from pathlib import Path
from typing import Dict

import numpy as np
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

SCENARIO_FILE = PROCESSED / "selected_operating_scenarios.csv"
HOSTING_FILE = PROCESSED / "wind_hosting_capacity.csv"
PYPSA_NETWORK = PROCESSED / "eirgrid_scenarios.nc"

OUTPUT_COUNTERFACTUAL = PROCESSED / "counterfactual_analysis.csv"


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

# Base assumptions
WHOLESALE_PRICE_EUR_PER_MWH = 50.0
CO2_EMISSION_FACTOR_TONNES_PER_MWH = 0.4
TIME_INTERVAL_HOURS = 0.25  # 15 minutes

# Sensitivity values
PRICE_SENSITIVITY = [30.0, 50.0, 75.0, 100.0]
CO2_SENSITIVITY = [0.4, 0.5]


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
        raise ValueError(f"Scenario '{scenario_name}' not in PyPSA snapshots.")
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


def run_optimiser_for_wind(
    available_wind_total_mw: float,
    demand_mw: Dict[str, float],
    wind_capacity_mw: Dict[str, float],
    network: LinearNetwork,
):
    available_wind_mw = allocate_available_wind(
        available_wind_total_mw, wind_capacity_mw
    )

    # Corrected: no artificial conventional generation.
    # The reference bus balances demand.
    conventional_generation_mw = {}

    optimiser_input = OptimiserInput(
        snapshot="COUNTERFACTUAL",
        demand_mw=demand_mw,
        available_wind_mw=available_wind_mw,
        wind_capacity_mw=wind_capacity_mw,
        conventional_generation_mw=conventional_generation_mw,
        lines={},
        scenario="counterfactual",
    )

    optimiser = WindCurtailmentOptimizer(network)
    optimiser_result = optimiser.solve(
        optimiser_input, wind_bus=WIND_GENERATORS
    )

    return optimiser_result


def security_screened_counterfactual_metrics(
    *,
    actual_not_generated_mw: float,
    optimised_curtailment_mw: float,
    ac_result: ACValidationResult,
    interval_hours: float,
) -> Dict[str, float | bool | str]:
    """Separate LP potential from AC-security-creditable results.

    LP feasibility is not evidence of physical deliverability.  The latter
    requires a converged AC result that passes both voltage and thermal
    checks.  Values with an economic or emissions interpretation are set to
    zero unless this gate passes; the uncredited LP potential remains visible
    for diagnosis only.
    """

    lp_potential_mw = max(
        0.0,
        actual_not_generated_mw - optimised_curtailment_mw,
    )
    lp_potential_energy_mwh = lp_potential_mw * interval_hours

    if ac_result.physically_secure:
        return {
            "lp_potential_curtailment_reduction_mw": lp_potential_mw,
            "lp_potential_energy_mwh": lp_potential_energy_mwh,
            "security_screened_curtailment_reduction_mw": lp_potential_mw,
            "security_screened_energy_mwh": lp_potential_energy_mwh,
            "eligible_for_value_credit": True,
            "security_assessment": "AC_SECURITY_PASSED",
        }

    assessment = (
        "AC_NOT_CONVERGED_NOT_CREDITED"
        if not ac_result.converged
        else "AC_INSECURE_NOT_CREDITED"
    )
    return {
        "lp_potential_curtailment_reduction_mw": lp_potential_mw,
        "lp_potential_energy_mwh": lp_potential_energy_mwh,
        "security_screened_curtailment_reduction_mw": 0.0,
        "security_screened_energy_mwh": 0.0,
        "eligible_for_value_credit": False,
        "security_assessment": assessment,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("HISTORICAL COUNTERFACTUAL CURTAILMENT ANALYSIS")
    print("=" * 80)

    # Load data
    wind_capacity_mw = load_wind_capacities_mw()
    network = build_linear_network_from_pypsa()

    scenarios = pd.read_csv(SCENARIO_FILE)

    # Only scenarios present in PyPSA snapshots
    n_snapshots = pypsa.Network(str(PYPSA_NETWORK)).snapshots.tolist()
    scenarios = scenarios[scenarios["Scenario"].isin(n_snapshots)]

    print(f"Scenarios to evaluate: {len(scenarios)}")
    print(f"Base wholesale price assumption: {WHOLESALE_PRICE_EUR_PER_MWH} €/MWh")
    print(f"Base CO₂ emission factor: {CO2_EMISSION_FACTOR_TONNES_PER_MWH} tCO₂/MWh")
    print(f"Time interval: {TIME_INTERVAL_HOURS} h\n")

    results = []

    for _, row in scenarios.iterrows():
        scenario_name = row["Scenario"]
        actual_wind_available = float(row["IE_Wind_Availability_MW"])
        actual_wind_generated = float(row["IE_Wind_Generation_MW"])
        actual_not_generated = float(row["Wind_Not_Generated_MW"])

        demand_mw = load_demand_mw(scenario_name)

        opt_result = run_optimiser_for_wind(
            actual_wind_available,
            demand_mw,
            wind_capacity_mw,
            network,
        )

        ac_result = validate_pypsa_dispatch(
            PYPSA_NETWORK,
            snapshot="COUNTERFACTUAL",
            scenario=scenario_name,
            demand_mw=demand_mw,
            accepted_wind_mw=opt_result.accepted_wind_by_generator_mw,
            wind_capacity_mw=wind_capacity_mw,
        )
        security_metrics = security_screened_counterfactual_metrics(
            actual_not_generated_mw=actual_not_generated,
            optimised_curtailment_mw=opt_result.curtailment_total_mw,
            ac_result=ac_result,
            interval_hours=TIME_INTERVAL_HOURS,
        )

        common_record = {
            "scenario": scenario_name,
            "evaluation_unit": "single_15_minute_snapshot",
            "interval_hours": TIME_INTERVAL_HOURS,
            "available_wind_mw": actual_wind_available,
            "actual_generated_mw": actual_wind_generated,
            "actual_not_generated_mw": actual_not_generated,
            "optimised_accepted_mw": opt_result.accepted_wind_total_mw,
            "optimised_curtailment_mw": opt_result.curtailment_total_mw,
            "ac_converged": ac_result.converged,
            "minimum_voltage_pu": ac_result.minimum_voltage_pu,
            "maximum_line_loading_percent": ac_result.maximum_line_loading_percent,
            "maximum_transformer_loading_percent": (
                ac_result.maximum_transformer_loading_percent
            ),
            "physically_secure": ac_result.physically_secure,
            "ac_message": ac_result.message,
            **security_metrics,
        }

        sensitivity_cases = [("base", WHOLESALE_PRICE_EUR_PER_MWH, CO2_EMISSION_FACTOR_TONNES_PER_MWH)]
        sensitivity_cases += [
            ("price", price, CO2_EMISSION_FACTOR_TONNES_PER_MWH)
            for price in PRICE_SENSITIVITY
            if price != WHOLESALE_PRICE_EUR_PER_MWH
        ]
        sensitivity_cases += [
            ("co2", WHOLESALE_PRICE_EUR_PER_MWH, co2_factor)
            for co2_factor in CO2_SENSITIVITY
            if co2_factor != CO2_EMISSION_FACTOR_TONNES_PER_MWH
        ]

        for sensitivity_case, price, co2_factor in sensitivity_cases:
            credited_energy_mwh = security_metrics[
                "security_screened_energy_mwh"
            ]
            results.append({
                **common_record,
                "indicative_value_eur": credited_energy_mwh * price,
                "indicative_co2_tonnes": credited_energy_mwh * co2_factor,
                "price_assumption_eur_per_mwh": price,
                "co2_factor_assumption_tonnes_per_mwh": co2_factor,
                "sensitivity_case": sensitivity_case,
            })

        print(f"Scenario {scenario_name}:")
        print(f"  LP potential reduction: {security_metrics['lp_potential_curtailment_reduction_mw']:.2f} MW")
        print(f"  AC status: {security_metrics['security_assessment']}")
        print(f"  AC-security-screened energy: {security_metrics['security_screened_energy_mwh']:.2f} MWh")
        print(f"  Indicative value / CO₂ are credited only after AC security passes.\n")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_COUNTERFACTUAL, index=False)

    # =========================================================================
    # SUMMARY (base case only)
    # =========================================================================

    base_df = results_df[results_df["sensitivity_case"] == "base"]

    print("=" * 80)
    print("COUNTERFACTUAL ANALYSIS SUMMARY (BASE ASSUMPTIONS)")
    print("=" * 80)
    print(base_df[[
        "scenario",
        "lp_potential_curtailment_reduction_mw",
        "physically_secure",
        "security_screened_energy_mwh",
        "indicative_value_eur",
        "indicative_co2_tonnes",
    ]].to_string(index=False))

    aggregate_energy = base_df["security_screened_energy_mwh"].sum()
    aggregate_value = base_df["indicative_value_eur"].sum()
    aggregate_co2 = base_df["indicative_co2_tonnes"].sum()

    print(
        "\nAggregate across selected independent 15-minute snapshot intervals "
        f"(not an annual total): {aggregate_energy:.2f} MWh"
    )
    print(f"Aggregate AC-security-screened indicative value: {aggregate_value:.2f} €")
    print(f"Aggregate AC-security-screened indicative CO₂: {aggregate_co2:.4f} tCO₂")

    print("\nSensitivity results saved in full CSV.")
    print(f"Results saved to: {OUTPUT_COUNTERFACTUAL}")
    print("\nCOUNTERFACTUAL ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
