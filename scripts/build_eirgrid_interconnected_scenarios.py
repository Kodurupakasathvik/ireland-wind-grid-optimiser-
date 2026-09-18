import pandas as pd
import pypsa
from pathlib import Path
# ============================================================
# IRELAND GRID - EIRGRID INTERCONNECTED OPERATING SCENARIOS
# ============================================================
#
# Purpose:
#   Map real EirGrid system-level operating conditions onto
#   the PyPSA transmission network while explicitly modelling
#   EWIC and Greenlink interconnector flows.
#
# IMPORTANT:
#   EirGrid system-level values are real.
#   Bus-level demand/generation locations remain synthetic.
#
# Convention:
#   Positive interconnector flow  = IMPORT into Ireland
#   Negative interconnector flow  = EXPORT from Ireland
#
# ============================================================
INPUT_NETWORK = Path(
    "data/processed/ireland_network.nc"
)
INPUT_SCENARIOS = Path(
    "data/processed/selected_operating_scenarios.csv"
)
OUTPUT = Path(
    "data/processed/eirgrid_interconnected_scenarios.nc"
)
print("=" * 70)
print("       IRELAND GRID - EIRGRID INTERCONNECTED SCENARIOS")
print("=" * 70)
# ------------------------------------------------------------
# LOAD NETWORK
# ------------------------------------------------------------
print()
print("Loading network:")
print(INPUT_NETWORK)
n = pypsa.Network(INPUT_NETWORK)
print("OK: Network loaded.")
# ------------------------------------------------------------
# LOAD SCENARIOS
# ------------------------------------------------------------
print()
print("Loading EirGrid scenarios:")
print(INPUT_SCENARIOS)
scenarios = pd.read_csv(
    INPUT_SCENARIOS,
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
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
# ------------------------------------------------------------
# SELECT SCENARIO BUSES
# ------------------------------------------------------------
load_buses = [
    "way/88462768-220",
    "way/104388595-220",
    "way/516651650-220",
    "way/88144450-220",
    "way/254158424-220",
    "virtual_relation/4872296:a:1-220",
    "way/264275258-220",
    "virtual_way/1059093587:0-220",
    "way/180637986-220",
    "way/1003262502-220",
]
generation_buses = [
    "way/88462768-220",
    "way/104388595-220",
    "way/516651650-220",
    "way/88144450-220",
    "way/254158424-220",
]
# ------------------------------------------------------------
# VERIFY BUSES
# ------------------------------------------------------------
for bus in load_buses:
    if bus not in n.buses.index:
        raise RuntimeError(
            f"Load bus not found: {bus}"
        )
for bus in generation_buses:
    if bus not in n.buses.index:
        raise RuntimeError(
            f"Generation bus not found: {bus}"
        )
print()
print("OK: All scenario buses exist.")
# ------------------------------------------------------------
# INTERCONNECTOR BUSES
#
# We need two explicit buses representing the external system.
# These buses are not claimed to be actual EirGrid bus locations.
# They are modelling interfaces used to represent net imports/
# exports recorded by EirGrid.
# ------------------------------------------------------------
EWIC_BUS = "external_ewic"
GREENLINK_BUS = "external_greenlink"
if EWIC_BUS not in n.buses.index:
    n.add(
        "Bus",
        EWIC_BUS,
        v_nom=220.0,
        carrier="AC"
    )
if GREENLINK_BUS not in n.buses.index:
    n.add(
        "Bus",
        GREENLINK_BUS,
        v_nom=220.0,
        carrier="AC"
    )
# ------------------------------------------------------------
# REMOVE OPERATING COMPONENTS
# ------------------------------------------------------------
if len(n.loads) > 0:
    n.remove(
        "Load",
        n.loads.index.tolist()
    )
if len(n.generators) > 0:
    n.remove(
        "Generator",
        n.generators.index.tolist()
    )
print()
print("Existing synthetic loads/generators removed.")
# ------------------------------------------------------------
# LOAD WEIGHTS
# ------------------------------------------------------------
load_weights = {
    load_buses[0]: 10,
    load_buses[1]: 9,
    load_buses[2]: 8,
    load_buses[3]: 7,
    load_buses[4]: 6,
    load_buses[5]: 5,
    load_buses[6]: 4,
    load_buses[7]: 3,
    load_buses[8]: 2,
    load_buses[9]: 1,
}
weight_total = sum(
    load_weights.values()
)
# ------------------------------------------------------------
# WIND GENERATION WEIGHTS
# ------------------------------------------------------------
generation_weights = {
    generation_buses[0]: 5,
    generation_buses[1]: 4,
    generation_buses[2]: 3,
    generation_buses[3]: 2,
    generation_buses[4]: 1,
}
generation_weight_total = sum(
    generation_weights.values()
)
# ------------------------------------------------------------
# SNAPSHOTS
# ------------------------------------------------------------
snapshot_names = scenarios[
    "Scenario"
].tolist()
n.set_snapshots(snapshot_names)
# ------------------------------------------------------------
# CREATE LOADS
# ------------------------------------------------------------
for bus in load_buses:
    load_name = f"eirgrid_load_{bus}"
    n.add(
        "Load",
        load_name,
        bus=bus,
    )
# ------------------------------------------------------------
# CREATE WIND GENERATORS
# ------------------------------------------------------------
for bus in generation_buses:
    generator_name = f"eirgrid_wind_{bus}"
    n.add(
        "Generator",
        generator_name,
        bus=bus,
        carrier="wind",
        control="PQ",
    )
# ------------------------------------------------------------
# CREATE NON-WIND GENERATION
#
# EirGrid gives total generation and wind generation.
# The difference is represented as aggregate non-wind
# generation.
#
# This is NOT claimed to represent individual plants.
# ------------------------------------------------------------
n.add(
    "Generator",
    "eirgrid_non_wind_generation",
    bus=generation_buses[0],
    carrier="non_wind",
    control="Slack",
)
# ------------------------------------------------------------
# EXPLICIT SYSTEM BALANCING
#
# The EirGrid system-level measurements do not close exactly:
# generation + interconnectors != demand.
# This generator represents that residual explicitly.
# It is a modelling balance, NOT measured generation.
# ------------------------------------------------------------
n.add(
    "Generator",
    "eirgrid_system_balance",
    bus=generation_buses[0],
    carrier="system_balance",
    control="PQ",
)
# ------------------------------------------------------------
EWIC_IRISH_BUS = generation_buses[4]
GREENLINK_IRISH_BUS = generation_buses[3]
# CREATE INTERCONNECTOR GENERATORS
# Interconnector flows are represented by dedicated Links below.
# No duplicate generator is created here.
# ------------------------------------------------------------
# CONNECT EXTERNAL BUSES TO IRISH NETWORK
#
# These are modelling interfaces, not claims about exact
# physical EirGrid topology.
# ------------------------------------------------------------
n.add("Generator", "ewic_external_grid", bus=EWIC_BUS, carrier="external_grid", control="Slack")
n.add("Generator", "greenlink_external_grid", bus=GREENLINK_BUS, carrier="external_grid", control="Slack")

n.add(
    "Link",
    "EWIC_interface",
    bus0=EWIC_BUS,
    bus1=EWIC_IRISH_BUS,
    p_nom=1000.0,
    p_min_pu=-1.0,
    p_max_pu=1.0,
    efficiency=1.0,
)
n.add(
    "Link",
    "Greenlink_interface",
    bus0=GREENLINK_BUS,
    bus1=GREENLINK_IRISH_BUS,
    p_nom=1000.0,
    p_min_pu=-1.0,
    p_max_pu=1.0,
    efficiency=1.0,
)
# ------------------------------------------------------------
# TIME SERIES
# ------------------------------------------------------------
for _, row in scenarios.iterrows():
    scenario = row["Scenario"]
    total_demand = float(
        row["IE_Demand_MW"]
    )
    wind_generation = float(
        row["IE_Wind_Generation_MW"]
    )
    total_generation = float(
        row["IE_Generation_MW"]
    )
    ewic = float(
        row["EWIC_MW"]
    )
    greenlink = float(
        row["Greenlink_MW"]
    )
    # --------------------------------------------------------
    # LOAD DISTRIBUTION
    # --------------------------------------------------------
    for bus, weight in load_weights.items():
        load_name = (
            f"eirgrid_load_{bus}"
        )
        demand_share = (
            total_demand
            * weight
            / weight_total
        )
        n.loads_t.p_set.loc[
            scenario,
            load_name
        ] = demand_share
    # --------------------------------------------------------
    # WIND DISTRIBUTION
    # --------------------------------------------------------
    for bus, weight in generation_weights.items():
        generator_name = (
            f"eirgrid_wind_{bus}"
        )
        wind_share = (
            wind_generation
            * weight
            / generation_weight_total
        )
        n.generators_t.p_set.loc[
            scenario,
            generator_name
        ] = wind_share
    # --------------------------------------------------------
    # NON-WIND GENERATION
    # --------------------------------------------------------
    non_wind_generation = (
        total_generation
        - wind_generation
    )
    n.generators_t.p_set.loc[
        scenario,
        "eirgrid_non_wind_generation"
    ] = max(
        non_wind_generation,
        0.0
    )
    # --------------------------------------------------------
    # EXPLICIT SYSTEM BALANCE
    #
    # Positive = additional generation required.
    # Negative = generation surplus that must be absorbed.
    # This is a modelling residual, not measured generation.
    # --------------------------------------------------------
    balance_residual = (
        total_generation
        + ewic
        + greenlink
        - total_demand
    )
    n.generators_t.p_set.loc[
        scenario,
        "eirgrid_system_balance"
    ] = -balance_residual
    # --------------------------------------------------------
    # INTERCONNECTOR IMPORTS
    #
    # Positive = import
    # Negative = export
    # --------------------------------------------------------
    n.links_t.p_set.loc[
        scenario,
        "EWIC_interface"
    ] = ewic
    n.links_t.p_set.loc[
        scenario,
        "Greenlink_interface"
    ] = greenlink
# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------
n.export_to_netcdf(
    OUTPUT
)
# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
print()
print("=" * 70)
print("       INTERCONNECTED EIRGRID SCENARIOS CREATED")
print("=" * 70)
print()
print(f"Scenarios       : {len(scenarios)}")
print(f"Loads           : {len(n.loads)}")
print(f"Generators      : {len(n.generators)}")
print(f"Links           : {len(n.links)}")
print(f"Buses           : {len(n.buses)}")
print(f"Lines           : {len(n.lines)}")
print(f"Transformers    : {len(n.transformers)}")
print()
print("Scenarios:")
for scenario in snapshot_names:
    print(f"  {scenario}")
print()
print("Saved:")
print(OUTPUT)
print()
print("IMPORTANT:")
print("EirGrid system-level measurements are real.")
print("Demand/generation bus allocation remains synthetic.")
print("Interconnector flows are explicitly represented.")
print("External interface buses are modelling constructs.")
print()
print("NEXT:")
print("Run an interconnector-aware power-flow validation.")
print("=" * 70)
