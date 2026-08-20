import pandas as pd
import pypsa
from pathlib import Path

print("=" * 70)
print("       IRELAND GRID - TRANSMISSION BOTTLENECK OPTIMIZATION")
print("=" * 70)

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

NETWORK_FILE = Path(
    "data/processed/eirgrid_interconnected_scenarios.nc"
)

TARGET_FILE = Path(
    "data/processed/optimization_targets.csv"
)

OUTPUT_NETWORK = Path(
    "data/processed/eirgrid_optimized_network.nc"
)

OUTPUT_RESULTS = Path(
    "data/processed/transmission_optimization_results.csv"
)

# ----------------------------------------------------------------------
# LOAD NETWORK
# ----------------------------------------------------------------------

print("\nLoading network:")
print(NETWORK_FILE)

if not NETWORK_FILE.exists():
    raise FileNotFoundError(
        f"Network file not found: {NETWORK_FILE}"
    )

network = pypsa.Network(NETWORK_FILE)

print("OK: Network loaded.")

# ----------------------------------------------------------------------
# LOAD OPTIMIZATION TARGETS
# ----------------------------------------------------------------------

print("\nLoading optimization targets:")
print(TARGET_FILE)

if not TARGET_FILE.exists():
    raise FileNotFoundError(
        f"Optimization target file not found: {TARGET_FILE}"
    )

targets = pd.read_csv(TARGET_FILE)

print("OK: Optimization targets loaded.")

# ----------------------------------------------------------------------
# NETWORK SUMMARY
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("NETWORK")
print("-" * 70)

print(f"Buses        : {len(network.buses)}")
print(f"Lines        : {len(network.lines)}")
print(f"Transformers : {len(network.transformers)}")
print(f"Generators   : {len(network.generators)}")
print(f"Loads        : {len(network.loads)}")
print(f"Links        : {len(network.links)}")

# ----------------------------------------------------------------------
# TARGET VALIDATION
# ----------------------------------------------------------------------

required_target_columns = [
    "Line",
    "Optimization_Rank",
    "Priority",
]

missing = [
    column
    for column in required_target_columns
    if column not in targets.columns
]

if missing:
    raise ValueError(
        "Missing required optimization columns: "
        + ", ".join(missing)
    )

print("\n" + "-" * 70)
print("OPTIMIZATION TARGETS")
print("-" * 70)

for _, row in targets.iterrows():

    print(
        f"Rank {int(row['Optimization_Rank'])}: "
        f"{row['Line']} -> {row['Priority']}"
    )

# ----------------------------------------------------------------------
# IMPORTANT MODELLING PRINCIPLE
# ----------------------------------------------------------------------
#
# We do NOT modify the original network topology directly.
#
# Instead, optimization creates candidate reinforcement capacity
# for recurring bottleneck lines.
#
# The reinforcement variable represents additional transferable
# capacity available on the targeted corridor.
#
# Existing network parameters remain unchanged.
#
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("CANDIDATE REINFORCEMENT MODEL")
print("-" * 70)

# Candidate reinforcement levels in MW.
#
# These are modelling candidates, NOT claims about actual EirGrid
# planned projects.

candidate_reinforcements = [
    0.0,
    250.0,
    500.0,
    750.0,
    1000.0,
]

print("\nCandidate reinforcement levels:")
for value in candidate_reinforcements:
    print(f"  {value:.0f} MW")

# ----------------------------------------------------------------------
# BUILD OPTIMIZATION TARGET TABLE
# ----------------------------------------------------------------------

optimization_rows = []

for _, target in targets.iterrows():

    line_name = target["Line"]

    if line_name not in network.lines.index:
        print(
            f"\nWARNING: Target line not found in network: "
            f"{line_name}"
        )
        continue

    original_limit = network.lines.loc[
        line_name,
        "s_nom"
    ]

    optimization_rows.append(
        {
            "Line": line_name,
            "Optimization_Rank": int(
                target["Optimization_Rank"]
            ),
            "Priority": target["Priority"],
            "Original_Capacity_MW": float(
                original_limit
            ),
        }
    )

optimization = pd.DataFrame(optimization_rows)

if optimization.empty:
    raise ValueError(
        "No valid optimization target lines "
        "were found in the network."
    )

# ----------------------------------------------------------------------
# SELECT REINFORCEMENT
# ----------------------------------------------------------------------
#
# We use a deterministic engineering screening approach:
#
# Critical lines receive 500 MW candidate reinforcement.
# Moderate lines receive 250 MW candidate reinforcement.
#
# This is deliberately a screening model.
# It does NOT claim that these are actual EirGrid projects.
#
# ----------------------------------------------------------------------

def choose_reinforcement(priority):

    if priority == "CRITICAL":
        return 500.0

    elif priority == "HIGH":
        return 500.0

    else:
        return 250.0


optimization["Added_Capacity_MW"] = (
    optimization["Priority"]
    .apply(choose_reinforcement)
)

optimization["Optimized_Capacity_MW"] = (
    optimization["Original_Capacity_MW"]
    + optimization["Added_Capacity_MW"]
)

# ----------------------------------------------------------------------
# DISPLAY PROPOSED REINFORCEMENTS
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("PROPOSED TRANSMISSION REINFORCEMENTS")
print("-" * 70)

for _, row in optimization.iterrows():

    print(f"\nLine : {row['Line']}")

    print(
        f"Original capacity : "
        f"{row['Original_Capacity_MW']:.2f} MW"
    )

    print(
        f"Added capacity    : "
        f"{row['Added_Capacity_MW']:.2f} MW"
    )

    print(
        f"New capacity      : "
        f"{row['Optimized_Capacity_MW']:.2f} MW"
    )

    print(
        f"Priority           : "
        f"{row['Priority']}"
    )

# ----------------------------------------------------------------------
# CREATE OPTIMIZED NETWORK COPY
# ----------------------------------------------------------------------

optimized_network = network.copy()

# Apply candidate reinforcement only to identified target lines.

for _, row in optimization.iterrows():

    line_name = row["Line"]

    optimized_network.lines.loc[
        line_name,
        "s_nom"
    ] = row["Optimized_Capacity_MW"]

# ----------------------------------------------------------------------
# SAVE OPTIMIZED NETWORK
# ----------------------------------------------------------------------

optimized_network.export_to_netcdf(
    OUTPUT_NETWORK
)

print("\n" + "-" * 70)
print("OPTIMIZED NETWORK SAVED")
print("-" * 70)

print(OUTPUT_NETWORK)

# ----------------------------------------------------------------------
# SAVE OPTIMIZATION RESULTS
# ----------------------------------------------------------------------

optimization.to_csv(
    OUTPUT_RESULTS,
    index=False
)

print("\nOptimization results saved:")
print(OUTPUT_RESULTS)

# ----------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("        TRANSMISSION OPTIMIZATION COMPLETE")
print("=" * 70)

print(
    f"\nTarget corridors reinforced : "
    f"{len(optimization)}"
)

print(
    f"Total added candidate capacity : "
    f"{optimization['Added_Capacity_MW'].sum():.2f} MW"
)

print("\nIMPORTANT:")
print(
    "The reinforcement values are modelling scenarios."
)

print(
    "They are NOT claims about actual EirGrid planned "
    "transmission projects."
)

print(
    "\nThe original interconnected network remains unchanged."
)

print("\nNEXT:")
print(
    "Run post-optimization AC power-flow validation "
    "and compare bottleneck loading before vs after."
)