import pandas as pd
import pypsa
from pathlib import Path

# ============================================================
# IRELAND GRID - BUILD REAL EIRGRID OPERATING SCENARIOS
# ============================================================

INPUT_NETWORK = Path("data/processed/ireland_network.nc")
INPUT_SCENARIOS = Path("data/processed/selected_operating_scenarios.csv")
OUTPUT = Path("data/processed/eirgrid_scenarios.nc")

print("=" * 70)
print("       IRELAND GRID - REAL EIRGRID OPERATING SCENARIOS")
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
# LOAD EIRGRID SCENARIOS
# ------------------------------------------------------------

print()
print("Loading selected EirGrid scenarios:")
print(INPUT_SCENARIOS)

scenarios = pd.read_csv(
    INPUT_SCENARIOS,
    parse_dates=["DateTime"]
)

print(f"OK: {len(scenarios)} scenarios loaded.")

# ------------------------------------------------------------
# CHECK NETWORK
# ------------------------------------------------------------

print()
print("-" * 70)
print("NETWORK")
print("-" * 70)

print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")

# ------------------------------------------------------------
# SELECT TRANSMISSION BUSES
# ------------------------------------------------------------

transmission_buses = n.buses[
    n.buses["v_nom"].isin([220.0, 275.0, 400.0])
].index.tolist()

if len(transmission_buses) == 0:
    raise RuntimeError("No transmission buses found.")

print()
print(f"Transmission buses available : {len(transmission_buses)}")

# ------------------------------------------------------------
# SELECT LOAD / GENERATION BUSES
#
# These are the same buses used in the synthetic baseline.
# For this stage we preserve the network topology and only
# replace the synthetic operating quantities with real
# EirGrid system-level values.
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
        raise RuntimeError(f"Load bus not found: {bus}")

for bus in generation_buses:
    if bus not in n.buses.index:
        raise RuntimeError(f"Generation bus not found: {bus}")

print()
print("OK: All scenario buses exist in the network.")

# ------------------------------------------------------------
# REMOVE EXISTING OPERATING COMPONENTS
#
# We retain buses, lines and transformers.
# Synthetic generators/loads are removed because the new
# scenarios will use real EirGrid operating values.
# ------------------------------------------------------------

if len(n.loads) > 0:
    n.remove("Load", n.loads.index.tolist())

if len(n.generators) > 0:
    n.remove("Generator", n.generators.index.tolist())

print()
print("Synthetic generators and loads removed.")

# ------------------------------------------------------------
# CREATE LOAD DISTRIBUTION
#
# System demand is distributed across the selected buses.
# The distribution follows the same weighted structure used
# in the baseline model.
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

weight_total = sum(load_weights.values())

# ------------------------------------------------------------
# CREATE GENERATION DISTRIBUTION
#
# Wind generation is distributed across the five generation
# buses using the baseline generation weights.
# ------------------------------------------------------------

generation_weights = {
    generation_buses[0]: 5,
    generation_buses[1]: 4,
    generation_buses[2]: 3,
    generation_buses[3]: 2,
    generation_buses[4]: 1,
}

generation_weight_total = sum(generation_weights.values())

# ------------------------------------------------------------
# CREATE SCENARIO SNAPSHOTS
# ------------------------------------------------------------

snapshot_names = scenarios["Scenario"].tolist()

n.set_snapshots(snapshot_names)

# ------------------------------------------------------------
# CREATE LOADS AND GENERATORS
# ------------------------------------------------------------

for bus in load_buses:

    load_name = f"eirgrid_load_{bus}"

    n.add(
        "Load",
        load_name,
        bus=bus,
    )

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
# ADD A SYSTEM BALANCING GENERATOR
#
# EirGrid provides total system generation and demand.
# Wind is explicitly represented using measured wind
# generation. The remaining generation is represented by
# a balancing conventional generator at the first generation
# bus so that the total system generation matches the real
# EirGrid value.
# ------------------------------------------------------------

n.add(
    "Generator",
    "eirgrid_balancing_generation",
    bus=generation_buses[0],
    carrier="balancing",
    control="Slack",
)

# ------------------------------------------------------------
# TIME SERIES
# ------------------------------------------------------------

for _, row in scenarios.iterrows():

    scenario = row["Scenario"]

    total_demand = float(row["IE_Demand_MW"])
    wind_generation = float(row["IE_Wind_Generation_MW"])
    total_generation = float(row["IE_Generation_MW"])

    # --------------------------------------------------------
    # LOADS
    # --------------------------------------------------------

    for bus, weight in load_weights.items():

        load_name = f"eirgrid_load_{bus}"

        demand_share = total_demand * weight / weight_total

        n.loads_t.p_set.loc[scenario, load_name] = demand_share

    # --------------------------------------------------------
    # WIND GENERATION
    # --------------------------------------------------------

    for bus, weight in generation_weights.items():

        generator_name = f"eirgrid_wind_{bus}"

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
    # BALANCING GENERATION
    # --------------------------------------------------------

    balancing_generation = (
        total_generation - wind_generation
    )

    if balancing_generation < 0:
        balancing_generation = 0.0

    n.generators_t.p_set.loc[
        scenario,
        "eirgrid_balancing_generation"
    ] = balancing_generation

# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

n.export_to_netcdf(OUTPUT)

print()
print("=" * 70)
print("              EIRGRID SCENARIOS CREATED")
print("=" * 70)

print()
print(f"Scenarios       : {len(scenarios)}")
print(f"Loads           : {len(n.loads)}")
print(f"Generators      : {len(n.generators)}")
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
print("The EirGrid values are real system-level measurements.")
print("Their spatial distribution across buses is still a")
print("synthetic modelling assumption because the workbook")
print("does not provide bus-level demand/generation locations.")

print()
print("NEXT:")
print("Run the EirGrid scenario power-flow validation.")
print("=" * 70)