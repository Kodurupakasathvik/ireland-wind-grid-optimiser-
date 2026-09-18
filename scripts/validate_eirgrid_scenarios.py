import pandas as pd
import pypsa
from pathlib import Path
# ============================================================
# IRELAND GRID - EIRGRID SCENARIO VALIDATION
# ============================================================
NETWORK_FILE = Path("data/processed/eirgrid_scenarios.nc")
SCENARIO_FILE = Path("data/processed/selected_operating_scenarios.csv")
print("=" * 70)
print("       IRELAND GRID - EIRGRID SCENARIO VALIDATION")
print("=" * 70)
# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print()
print("Loading PyPSA network:")
print(NETWORK_FILE)
n = pypsa.Network(NETWORK_FILE)
print("OK: Network loaded.")
print()
print("Loading original EirGrid scenario data:")
print(SCENARIO_FILE)
scenarios = pd.read_csv(
    SCENARIO_FILE,
    parse_dates=["DateTime"]
)
print(f"OK: {len(scenarios)} scenarios loaded.")
# ------------------------------------------------------------
# NETWORK SUMMARY
# ------------------------------------------------------------
print()
print("-" * 70)
print("NETWORK")
print("-" * 70)
print(f"Buses        : {len(n.buses)}")
print(f"Loads        : {len(n.loads)}")
print(f"Generators   : {len(n.generators)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print()
print("Snapshots:")
for snapshot in n.snapshots:
    print(f"  {snapshot}")
# ------------------------------------------------------------
# GENERATOR IDENTIFICATION
# ------------------------------------------------------------
wind_generators = [
    name for name in n.generators.index
    if str(name).startswith("eirgrid_wind_")
]
balancing_generators = [
    name for name in n.generators.index
    if str(name).startswith("eirgrid_balancing")
]
print()
print("-" * 70)
print("GENERATORS")
print("-" * 70)
print(f"Wind generators      : {len(wind_generators)}")
print(f"Balancing generators : {len(balancing_generators)}")
# ------------------------------------------------------------
# LOAD IDENTIFICATION
# ------------------------------------------------------------
eirgrid_loads = [
    name for name in n.loads.index
    if str(name).startswith("eirgrid_load_")
]
print(f"Scenario loads       : {len(eirgrid_loads)}")
# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
print()
print("=" * 70)
print("SCENARIO VALIDATION")
print("=" * 70)
results = []
for _, row in scenarios.iterrows():
    scenario = row["Scenario"]
    # --------------------------------------------------------
    # REAL EIRGRID VALUES
    # --------------------------------------------------------
    eirgrid_demand = float(row["IE_Demand_MW"])
    eirgrid_wind = float(row["IE_Wind_Generation_MW"])
    eirgrid_generation = float(row["IE_Generation_MW"])
    # --------------------------------------------------------
    # PYPSA MODEL VALUES
    # --------------------------------------------------------
    model_demand = float(
        n.loads_t.p_set.loc[scenario, eirgrid_loads].sum()
    )
    model_wind = float(
        n.generators_t.p_set.loc[
            scenario,
            wind_generators
        ].sum()
    )
    model_balancing = float(
        n.generators_t.p_set.loc[
            scenario,
            balancing_generators
        ].sum()
    )
    model_generation = model_wind + model_balancing
    # --------------------------------------------------------
    # DIFFERENCES
    # --------------------------------------------------------
    demand_difference = model_demand - eirgrid_demand
    wind_difference = model_wind - eirgrid_wind
    generation_difference = (
        model_generation - eirgrid_generation
    )
    model_balance = model_generation - model_demand
    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------
    if (
        abs(demand_difference) < 0.01
        and abs(wind_difference) < 0.01
        and abs(generation_difference) < 0.01
    ):
        status = "PASS"
    else:
        status = "CHECK"
    print()
    print("-" * 70)
    print(f"SCENARIO: {scenario}")
    print("-" * 70)
    print(f"EirGrid demand          : {eirgrid_demand:10.2f} MW")
    print(f"PyPSA demand            : {model_demand:10.2f} MW")
    print(f"Demand difference       : {demand_difference:10.2f} MW")
    print()
    print(f"EirGrid wind            : {eirgrid_wind:10.2f} MW")
    print(f"PyPSA wind              : {model_wind:10.2f} MW")
    print(f"Wind difference         : {wind_difference:10.2f} MW")
    print()
    print(f"EirGrid generation      : {eirgrid_generation:10.2f} MW")
    print(f"PyPSA wind              : {model_wind:10.2f} MW")
    print(f"PyPSA balancing         : {model_balancing:10.2f} MW")
    print(f"PyPSA total generation  : {model_generation:10.2f} MW")
    print(f"Generation difference   : {generation_difference:10.2f} MW")
    print()
    print(f"PyPSA system balance    : {model_balance:10.2f} MW")
    print(f"STATUS                  : {status}")
    results.append({
        "Scenario": scenario,
        "EirGrid_Demand_MW": eirgrid_demand,
        "PyPSA_Demand_MW": model_demand,
        "Demand_Difference_MW": demand_difference,
        "EirGrid_Wind_MW": eirgrid_wind,
        "PyPSA_Wind_MW": model_wind,
        "Wind_Difference_MW": wind_difference,
        "EirGrid_Generation_MW": eirgrid_generation,
        "PyPSA_Generation_MW": model_generation,
        "Generation_Difference_MW": generation_difference,
        "PyPSA_Balance_MW": model_balance,
        "Status": status,
    })
# ------------------------------------------------------------
# DUPLICATE SCENARIO CHECK
# ------------------------------------------------------------
print()
print("=" * 70)
print("SCENARIO UNIQUENESS CHECK")
print("=" * 70)
duplicates = scenarios[
    scenarios.duplicated(
        subset=[
            "IE_Demand_MW",
            "IE_Wind_Generation_MW",
            "IE_Wind_Availability_MW",
            "IE_Solar_Generation_MW",
            "IE_Hydro_MW",
            "SNSP"
        ],
        keep=False
    )
]
if len(duplicates) == 0:
    print("OK: No identical operating scenarios detected.")
else:
    print(
        f"WARNING: {len(duplicates)} records belong to "
        "duplicate operating conditions."
    )
    print()
    print(
        duplicates[
            [
                "Scenario",
                "DateTime",
                "IE_Demand_MW",
                "IE_Wind_Generation_MW",
                "IE_Wind_Availability_MW",
                "IE_Solar_Generation_MW",
                "IE_Hydro_MW",
                "SNSP"
            ]
        ].to_string(index=False)
    )
# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(
    results_df[
        [
            "Scenario",
            "EirGrid_Demand_MW",
            "PyPSA_Demand_MW",
            "EirGrid_Wind_MW",
            "PyPSA_Wind_MW",
            "EirGrid_Generation_MW",
            "PyPSA_Generation_MW",
            "PyPSA_Balance_MW",
            "Status"
        ]
    ].to_string(index=False)
)
# ------------------------------------------------------------
# SAVE VALIDATION RESULTS
# ------------------------------------------------------------
output_file = Path(
    "data/processed/eirgrid_scenario_validation.csv"
)
results_df.to_csv(
    output_file,
    index=False
)
print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print()
print("Saved:")
print(output_file)
print()
print("IMPORTANT:")
print("PASS means the scenario values were mapped correctly.")
print("CHECK means the mapping requires investigation.")
print()
print("Do NOT perform network optimisation yet.")
print("First resolve any CHECK results and duplicate scenarios.")
print("=" * 70)