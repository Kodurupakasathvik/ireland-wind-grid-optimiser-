import pandas as pd
import pypsa
from pathlib import Path
# ============================================================
# IRELAND GRID - INTERCONNECTED EIRGRID SCENARIO VALIDATION
# ============================================================
NETWORK = Path(
    "data/processed/eirgrid_interconnected_scenarios.nc"
)
SCENARIOS = Path(
    "data/processed/selected_operating_scenarios.csv"
)
OUTPUT = Path(
    "data/processed/eirgrid_interconnected_validation.csv"
)
print("=" * 70)
print("   IRELAND GRID - INTERCONNECTED EIRGRID SCENARIO VALIDATION")
print("=" * 70)
# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
print()
print("Loading network:")
print(NETWORK)
n = pypsa.Network(NETWORK)
print("OK: Network loaded.")
print()
print("Loading original EirGrid scenarios:")
print(SCENARIOS)
df = pd.read_csv(SCENARIOS)
print(f"OK: {len(df)} scenarios loaded.")
# ------------------------------------------------------------
# NETWORK
# ------------------------------------------------------------
print()
print("-" * 70)
print("NETWORK")
print("-" * 70)
print(f"Buses        : {len(n.buses)}")
print(f"Loads        : {len(n.loads)}")
print(f"Generators   : {len(n.generators)}")
print(f"Links        : {len(n.links)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
# ------------------------------------------------------------
# COMPONENT IDENTIFICATION
# ------------------------------------------------------------
load_names = [
    x for x in n.loads.index
    if x.startswith("eirgrid_load_")
]
wind_names = [
    x for x in n.generators.index
    if x.startswith("eirgrid_wind_")
]
non_wind_name = "eirgrid_non_wind_generation"
ewic_name = "ewic_import"
greenlink_name = "greenlink_import"
# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
results = []
print()
print("=" * 70)
print("SCENARIO BALANCE VALIDATION")
print("=" * 70)
for _, row in df.iterrows():
    scenario = row["Scenario"]
    # --------------------------------------------------------
    # EIRGRID VALUES
    # --------------------------------------------------------
    eirgrid_demand = float(
        row["IE_Demand_MW"]
    )
    eirgrid_wind = float(
        row["IE_Wind_Generation_MW"]
    )
    eirgrid_generation = float(
        row["IE_Generation_MW"]
    )
    eirgrid_ewic = float(
        row["EWIC_MW"]
    )
    eirgrid_greenlink = float(
        row["Greenlink_MW"]
    )
    # --------------------------------------------------------
    # PYPSA LOAD
    # --------------------------------------------------------
    pypsa_demand = n.loads_t.p_set.loc[
        scenario,
        load_names
    ].sum()
    # --------------------------------------------------------
    # PYPSA WIND
    # --------------------------------------------------------
    pypsa_wind = n.generators_t.p_set.loc[
        scenario,
        wind_names
    ].sum()
    # --------------------------------------------------------
    # PYPSA NON-WIND
    # --------------------------------------------------------
    pypsa_non_wind = n.generators_t.p_set.loc[
        scenario,
        non_wind_name
    ]
    # --------------------------------------------------------
    # PYPSA INTERCONNECTORS
    # --------------------------------------------------------
    pypsa_ewic = n.generators_t.p_set.loc[
        scenario,
        ewic_name
    ]
    pypsa_greenlink = n.generators_t.p_set.loc[
        scenario,
        greenlink_name
    ]
    pypsa_interconnector = (
        pypsa_ewic
        + pypsa_greenlink
    )
    # --------------------------------------------------------
    # TOTAL INTERNAL GENERATION
    # --------------------------------------------------------
    pypsa_internal_generation = (
        pypsa_wind
        + pypsa_non_wind
    )
    # --------------------------------------------------------
    # SYSTEM SUPPLY INCLUDING INTERCONNECTORS
    # --------------------------------------------------------
    pypsa_total_supply = (
        pypsa_internal_generation
        + pypsa_interconnector
    )
    # --------------------------------------------------------
    # NET BALANCE
    # --------------------------------------------------------
    pypsa_balance = (
        pypsa_total_supply
        - pypsa_demand
    )
    # --------------------------------------------------------
    # EXPECTED BALANCE FROM EIRGRID
    # --------------------------------------------------------
    expected_balance = (
        eirgrid_generation
        + eirgrid_ewic
        + eirgrid_greenlink
        - eirgrid_demand
    )
    # --------------------------------------------------------
    # DIFFERENCES
    # --------------------------------------------------------
    demand_difference = (
        pypsa_demand
        - eirgrid_demand
    )
    wind_difference = (
        pypsa_wind
        - eirgrid_wind
    )
    generation_difference = (
        pypsa_internal_generation
        - eirgrid_generation
    )
    ewic_difference = (
        pypsa_ewic
        - eirgrid_ewic
    )
    greenlink_difference = (
        pypsa_greenlink
        - eirgrid_greenlink
    )
    balance_difference = (
        pypsa_balance
        - expected_balance
    )
    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------
    tolerance = 0.01
    if (
        abs(demand_difference) <= tolerance
        and
        abs(wind_difference) <= tolerance
        and
        abs(generation_difference) <= tolerance
        and
        abs(ewic_difference) <= tolerance
        and
        abs(greenlink_difference) <= tolerance
        and
        abs(balance_difference) <= tolerance
    ):
        status = "PASS"
    else:
        status = "CHECK"
    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print(f"SCENARIO: {scenario}")
    print("-" * 70)
    print(
        f"EirGrid demand          : {eirgrid_demand:10.2f} MW"
    )
    print(
        f"PyPSA demand            : {pypsa_demand:10.2f} MW"
    )
    print(
        f"Demand difference       : {demand_difference:10.2f} MW"
    )
    print()
    print(
        f"EirGrid wind            : {eirgrid_wind:10.2f} MW"
    )
    print(
        f"PyPSA wind              : {pypsa_wind:10.2f} MW"
    )
    print(
        f"Wind difference         : {wind_difference:10.2f} MW"
    )
    print()
    print(
        f"EirGrid generation      : {eirgrid_generation:10.2f} MW"
    )
    print(
        f"PyPSA internal gen      : {pypsa_internal_generation:10.2f} MW"
    )
    print(
        f"Generation difference   : {generation_difference:10.2f} MW"
    )
    print()
    print(
        f"EirGrid EWIC            : {eirgrid_ewic:10.2f} MW"
    )
    print(
        f"PyPSA EWIC              : {pypsa_ewic:10.2f} MW"
    )
    print(
        f"EWIC difference         : {ewic_difference:10.2f} MW"
    )
    print()
    print(
        f"EirGrid Greenlink       : {eirgrid_greenlink:10.2f} MW"
    )
    print(
        f"PyPSA Greenlink         : {pypsa_greenlink:10.2f} MW"
    )
    print(
        f"Greenlink difference    : {greenlink_difference:10.2f} MW"
    )
    print()
    print(
        f"Expected system balance : {expected_balance:10.2f} MW"
    )
    print(
        f"PyPSA system balance    : {pypsa_balance:10.2f} MW"
    )
    print(
        f"Balance difference      : {balance_difference:10.2f} MW"
    )
    print(
        f"STATUS                  : {status}"
    )
    results.append({
        "Scenario": scenario,
        "EirGrid_Demand_MW": eirgrid_demand,
        "PyPSA_Demand_MW": pypsa_demand,
        "EirGrid_Wind_MW": eirgrid_wind,
        "PyPSA_Wind_MW": pypsa_wind,
        "EirGrid_Generation_MW": eirgrid_generation,
        "PyPSA_Internal_Generation_MW":
            pypsa_internal_generation,
        "EirGrid_EWIC_MW": eirgrid_ewic,
        "PyPSA_EWIC_MW": pypsa_ewic,
        "EirGrid_Greenlink_MW":
            eirgrid_greenlink,
        "PyPSA_Greenlink_MW":
            pypsa_greenlink,
        "Expected_Balance_MW":
            expected_balance,
        "PyPSA_Balance_MW":
            pypsa_balance,
        "Balance_Difference_MW":
            balance_difference,
        "Status": status,
    })
# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
result_df = pd.DataFrame(results)
result_df.to_csv(
    OUTPUT,
    index=False
)
print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(
    result_df[
        [
            "Scenario",
            "EirGrid_Demand_MW",
            "EirGrid_Generation_MW",
            "EirGrid_EWIC_MW",
            "EirGrid_Greenlink_MW",
            "Expected_Balance_MW",
            "PyPSA_Balance_MW",
            "Status",
        ]
    ].to_string(index=False)
)
print()
print("Saved:")
print(OUTPUT)
if (result_df["Status"] == "PASS").all():
    print()
    print("=" * 70)
    print("ALL SCENARIOS PASS")
    print("=" * 70)
else:
    print()
    print("=" * 70)
    print("CHECK RESULTS REQUIRE INVESTIGATION")
    print("=" * 70)