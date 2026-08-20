"""
======================================================================
IRELAND GRID - SECOND REINFORCEMENT ITERATION
VALIDATED VERSION
======================================================================

Purpose:
    Apply a second reinforcement iteration using ONLY the validated
    line-level bottleneck results from:

        data/processed/line_level_bottleneck_comparison.csv

IMPORTANT:
    Do NOT recompute AC loading here.

Reason:
    A non-converged AC power flow can produce numerically absurd
    loading values. Those values must NEVER be used for reinforcement.

This script:
    1. Loads the optimized network.
    2. Loads the validated line-level comparison.
    3. Identifies residual overloaded target lines.
    4. Uses validated optimized maximum loading.
    5. Calculates a reasonable second reinforcement.
    6. Applies reinforcement.
    7. Saves iteration-2 network.
    8. Runs basic structural validation.
======================================================================
"""

import os
import sys
import math
import pandas as pd
import pypsa


# =====================================================================
# PATHS
# =====================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

OPTIMIZED_NETWORK_PATH = os.path.join(
    DATA_DIR,
    "eirgrid_optimized_network.nc"
)

LINE_ANALYSIS_PATH = os.path.join(
    DATA_DIR,
    "line_level_bottleneck_comparison.csv"
)

RESIDUAL_PATH = os.path.join(
    DATA_DIR,
    "residual_transmission_bottlenecks.csv"
)

OUTPUT_NETWORK_PATH = os.path.join(
    DATA_DIR,
    "eirgrid_second_reinforced_network.nc"
)

OUTPUT_RESULTS_PATH = os.path.join(
    DATA_DIR,
    "second_reinforcement_results.csv"
)


# =====================================================================
# TARGET LINES
# =====================================================================

TARGET_LINES = [
    "merged_way/257889771-220+1",
    "merged_way/1231251986-220+2",
    "way/343436171-220",
]


# =====================================================================
# SETTINGS
# =====================================================================

# Reinforcement target:
# We want the validated maximum loading to fall below this value.
TARGET_LOADING_PERCENT = 95.0

# Small engineering margin above the theoretical requirement.
SAFETY_MARGIN = 1.05

# Do not allow absurd reinforcement.
# If something outside this range appears, stop rather than creating
# a physically meaningless network.
MAX_CAPACITY_MULTIPLIER = 3.0


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def normalize_column_name(name):
    """
    Convert a column name into a normalized form for flexible matching.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, candidates):
    """
    Find a column using several possible names.
    """

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in normalized:
            return normalized[key]

    # Partial matching
    for col in df.columns:

        normalized_col = normalize_column_name(col)

        for candidate in candidates:

            normalized_candidate = normalize_column_name(candidate)

            if normalized_candidate in normalized_col:
                return col

    return None


def clean_numeric(value):
    """
    Convert a CSV value to a safe finite float.
    """

    try:

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):

        return None


def print_separator(title=None):

    print()
    print("=" * 70)

    if title:
        print(title)

    print("=" * 70)


# =====================================================================
# START
# =====================================================================

print_separator(
    "IRELAND GRID - SECOND REINFORCEMENT ITERATION"
)

print()
print("VALIDATED SECOND-ITERATION MODE")
print()
print("This version DOES NOT recompute AC loading.")
print("It uses the previously validated line-level analysis.")
print()


# =====================================================================
# LOAD NETWORK
# =====================================================================

print_separator("LOADING OPTIMIZED NETWORK")

print(OPTIMIZED_NETWORK_PATH)

if not os.path.exists(OPTIMIZED_NETWORK_PATH):

    print()
    print("ERROR: Optimized network not found.")
    print(OPTIMIZED_NETWORK_PATH)
    sys.exit(1)


try:

    network = pypsa.Network(
        OPTIMIZED_NETWORK_PATH
    )

except Exception as exc:

    print()
    print("ERROR: Could not load optimized network.")
    print(exc)
    sys.exit(1)


print("OK: Optimized network loaded.")


# =====================================================================
# NETWORK SUMMARY
# =====================================================================

print_separator("NETWORK")

print(f"Buses        : {len(network.buses)}")
print(f"Lines        : {len(network.lines)}")
print(f"Transformers : {len(network.transformers)}")
print(f"Generators   : {len(network.generators)}")
print(f"Loads        : {len(network.loads)}")


# =====================================================================
# LOAD VALIDATED LINE ANALYSIS
# =====================================================================

print_separator(
    "LOADING VALIDATED LINE-LEVEL ANALYSIS"
)

print(LINE_ANALYSIS_PATH)

if not os.path.exists(LINE_ANALYSIS_PATH):

    print()
    print("ERROR: Validated line-level analysis file not found.")
    print()
    print("Expected:")
    print(LINE_ANALYSIS_PATH)
    print()
    print(
        "Run analyse_line_level_bottlenecks.py first."
    )

    sys.exit(1)


try:

    df = pd.read_csv(
        LINE_ANALYSIS_PATH
    )

except Exception as exc:

    print()
    print("ERROR: Could not read line analysis CSV.")
    print(exc)
    sys.exit(1)


print("OK: Validated line-level analysis loaded.")

print()
print("Columns found:")
for column in df.columns:
    print(f"  {column}")


# =====================================================================
# IDENTIFY COLUMNS
# =====================================================================

LINE_COLUMN = find_column(
    df,
    [
        "line",
        "line_name",
        "name",
        "Line",
        "Line Name",
        "component",
        "branch",
    ]
)

OPTIMIZED_MAX_COLUMN = find_column(
    df,
    [
        "optimized_max_loading",
        "optimized_max_loading_percent",
        "optimized_max_loading_pct",
        "max_optimized_loading",
        "optimized_max",
        "optimized_loading_max",
    ]
)

ORIGINAL_MAX_COLUMN = find_column(
    df,
    [
        "original_max_loading",
        "original_max_loading_percent",
        "original_max_loading_pct",
        "max_original_loading",
        "original_max",
    ]
)

OPTIMIZED_AVG_COLUMN = find_column(
    df,
    [
        "optimized_average_loading",
        "optimized_avg_loading",
        "optimized_average",
        "optimized_avg",
    ]
)

ORIGINAL_AVG_COLUMN = find_column(
    df,
    [
        "original_average_loading",
        "original_avg_loading",
        "original_average",
        "original_avg",
    ]
)


print_separator("IDENTIFIED DATA COLUMNS")

print(f"Line column                 : {LINE_COLUMN}")
print(f"Original max loading        : {ORIGINAL_MAX_COLUMN}")
print(f"Optimized max loading       : {OPTIMIZED_MAX_COLUMN}")
print(f"Original average loading    : {ORIGINAL_AVG_COLUMN}")
print(f"Optimized average loading   : {OPTIMIZED_AVG_COLUMN}")


# =====================================================================
# VALIDATION
# =====================================================================

if LINE_COLUMN is None:

    print()
    print("ERROR: Could not identify line-name column.")
    print()
    print("Available columns:")

    for column in df.columns:
        print(f"  - {column}")

    sys.exit(1)


if OPTIMIZED_MAX_COLUMN is None:

    print()
    print(
        "ERROR: Could not identify validated optimized maximum "
        "loading column."
    )

    print()
    print(
        "The script will NOT recompute loading automatically."
    )

    print()
    print(
        "This is intentional: recomputing from a non-converged "
        "power flow can create meaningless values."
    )

    sys.exit(1)


# =====================================================================
# BUILD TARGET DATA
# =====================================================================

print_separator(
    "VALIDATED TARGET BOTTLENECKS"
)

target_rows = []


for target in TARGET_LINES:

    matches = df[
        df[LINE_COLUMN].astype(str).str.strip() == target
    ]

    if matches.empty:

        print()
        print(f"WARNING: Target line not found:")
        print(f"  {target}")

        continue

    row = matches.iloc[0]

    optimized_max = clean_numeric(
        row[OPTIMIZED_MAX_COLUMN]
    )

    original_max = None

    if ORIGINAL_MAX_COLUMN is not None:

        original_max = clean_numeric(
            row[ORIGINAL_MAX_COLUMN]
        )

    optimized_avg = None

    if OPTIMIZED_AVG_COLUMN is not None:

        optimized_avg = clean_numeric(
            row[OPTIMIZED_AVG_COLUMN]
        )

    original_avg = None

    if ORIGINAL_AVG_COLUMN is not None:

        original_avg = clean_numeric(
            row[ORIGINAL_AVG_COLUMN]
        )

    # ---------------------------------------------------------------
    # CRITICAL SANITY CHECK
    # ---------------------------------------------------------------

    if optimized_max is None:

        print()
        print(
            f"ERROR: Invalid optimized maximum loading for {target}"
        )

        continue

    if optimized_max < 0:

        print()
        print(
            f"ERROR: Negative loading detected for {target}"
        )

        continue

    if optimized_max > 10000:

        print()
        print(
            f"ERROR: Absurd loading detected for {target}"
        )

        print(
            f"  Loading = {optimized_max:.6f}%"
        )

        print(
            "This value will NOT be used."
        )

        continue

    if target not in network.lines.index:

        print()
        print(
            f"WARNING: Target line is not present in optimized network:"
        )

        print(f"  {target}")

        continue

    target_rows.append(
        {
            "line": target,
            "original_max_loading": original_max,
            "optimized_max_loading": optimized_max,
            "original_average_loading": original_avg,
            "optimized_average_loading": optimized_avg,
        }
    )


# =====================================================================
# CHECK TARGETS
# =====================================================================

if len(target_rows) == 0:

    print()
    print("ERROR: No valid target lines available.")
    print()
    print("NO REINFORCEMENT WAS APPLIED.")
    sys.exit(1)


# =====================================================================
# REINFORCEMENT CALCULATION
# =====================================================================

print_separator(
    "SECOND REINFORCEMENT CALCULATION"
)

reinforcement_results = []


for item in target_rows:

    line_name = item["line"]

    optimized_max = item["optimized_max_loading"]

    line = network.lines.loc[line_name]

    existing_capacity = clean_numeric(
        line["s_nom"]
    )

    if existing_capacity is None or existing_capacity <= 0:

        print()
        print(
            f"ERROR: Invalid existing capacity for {line_name}"
        )

        continue

    # ---------------------------------------------------------------
    # Required capacity:
    #
    # current loading is:
    #
    #       loading = flow / capacity
    #
    # Therefore:
    #
    #       required capacity =
    #       current capacity *
    #       current loading / target loading
    #
    # ---------------------------------------------------------------

    theoretical_capacity = (
        existing_capacity
        * optimized_max
        / TARGET_LOADING_PERCENT
    )

    required_capacity = (
        theoretical_capacity
        * SAFETY_MARGIN
    )

    multiplier = (
        required_capacity
        / existing_capacity
    )

    print()
    print(f"Line: {line_name}")

    print(
        f"  Validated optimized max loading : "
        f"{optimized_max:.2f}%"
    )

    print(
        f"  Existing capacity              : "
        f"{existing_capacity:.3f}"
    )

    print(
        f"  Target loading                 : "
        f"{TARGET_LOADING_PERCENT:.2f}%"
    )

    print(
        f"  Safety margin                  : "
        f"{SAFETY_MARGIN:.2f}x"
    )

    print(
        f"  Required capacity              : "
        f"{required_capacity:.3f}"
    )

    print(
        f"  Capacity multiplier             : "
        f"{multiplier:.3f}x"
    )

    # ---------------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------------

    if multiplier > MAX_CAPACITY_MULTIPLIER:

        print()
        print(
            "  WARNING: Required reinforcement exceeds "
            "maximum allowed multiplier."
        )

        print(
            "  This line will NOT be automatically reinforced."
        )

        continue

    if required_capacity <= existing_capacity:

        print(
            "  No additional reinforcement required."
        )

        continue

    reinforcement_results.append(
        {
            "line": line_name,
            "original_max_loading": item[
                "original_max_loading"
            ],
            "optimized_max_loading_before": optimized_max,
            "existing_capacity": existing_capacity,
            "theoretical_capacity": theoretical_capacity,
            "required_capacity": required_capacity,
            "new_capacity": required_capacity,
            "capacity_multiplier": multiplier,
            "target_loading_percent": TARGET_LOADING_PERCENT,
            "safety_margin": SAFETY_MARGIN,
        }
    )


# =====================================================================
# REINFORCEMENT SUMMARY
# =====================================================================

print_separator(
    "SELECTED SECOND-ITERATION REINFORCEMENTS"
)


if not reinforcement_results:

    print()
    print("No reinforcement passed safety validation.")
    print()
    print("The network will NOT be modified.")
    sys.exit(0)


for result in reinforcement_results:

    print()
    print(
        f"Line: {result['line']}"
    )

    print(
        f"  Previous validated loading : "
        f"{result['optimized_max_loading_before']:.2f}%"
    )

    print(
        f"  Existing capacity          : "
        f"{result['existing_capacity']:.3f}"
    )

    print(
        f"  New capacity               : "
        f"{result['new_capacity']:.3f}"
    )

    print(
        f"  Capacity increase          : "
        f"{result['new_capacity'] - result['existing_capacity']:.3f}"
    )

    print(
        f"  Capacity multiplier        : "
        f"{result['capacity_multiplier']:.3f}x"
    )


# =====================================================================
# APPLY REINFORCEMENTS
# =====================================================================

print_separator(
    "APPLYING SECOND REINFORCEMENTS"
)


for result in reinforcement_results:

    line_name = result["line"]

    old_capacity = network.lines.at[
        line_name,
        "s_nom"
    ]

    new_capacity = result["new_capacity"]

    network.lines.at[
        line_name,
        "s_nom"
    ] = new_capacity

    print(
        f"Reinforced: {line_name}"
    )

    print(
        f"  s_nom: {old_capacity:.3f} -> "
        f"{new_capacity:.3f}"
    )


# =====================================================================
# SAVE ITERATION-2 NETWORK
# =====================================================================

print_separator(
    "SAVING ITERATION-2 NETWORK"
)


try:

    # IMPORTANT:
    # Correct PyPSA method is export_to_netcdf(),
    # NOT export_netcdf().

    network.export_to_netcdf(
        OUTPUT_NETWORK_PATH
    )

except Exception as exc:

    print()
    print(
        "ERROR: Could not save iteration-2 network."
    )

    print(exc)

    sys.exit(1)


print(
    "OK: Iteration-2 network saved."
)

print(
    OUTPUT_NETWORK_PATH
)


# =====================================================================
# SAVE REINFORCEMENT RESULTS
# =====================================================================

print_separator(
    "SAVING REINFORCEMENT RESULTS"
)


results_df = pd.DataFrame(
    reinforcement_results
)


try:

    results_df.to_csv(
        OUTPUT_RESULTS_PATH,
        index=False
    )

except Exception as exc:

    print()
    print(
        "WARNING: Could not save results CSV."
    )

    print(exc)

else:

    print(
        "OK: Reinforcement results saved."
    )

    print(
        OUTPUT_RESULTS_PATH
    )


# =====================================================================
# BASIC STRUCTURAL VALIDATION
# =====================================================================

print_separator(
    "STRUCTURAL VALIDATION"
)

try:

    check_network = pypsa.Network(
        OUTPUT_NETWORK_PATH
    )

    print(
        "PASS: Iteration-2 network can be reloaded."
    )

    print(
        f"  Buses        : {len(check_network.buses)}"
    )

    print(
        f"  Lines        : {len(check_network.lines)}"
    )

    print(
        f"  Transformers : {len(check_network.transformers)}"
    )

    print(
        f"  Generators   : {len(check_network.generators)}"
    )

    print(
        f"  Loads        : {len(check_network.loads)}"
    )

except Exception as exc:

    print()
    print(
        "FAIL: Saved network could not be reloaded."
    )

    print(exc)

    sys.exit(1)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print_separator(
    "SECOND REINFORCEMENT ITERATION COMPLETE"
)

print()
print(
    f"Validated target lines found : {len(target_rows)}"
)

print(
    f"Lines reinforced             : "
    f"{len(reinforcement_results)}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "The script intentionally did NOT run a new AC power flow."
)

print(
    "The previous AC calculation contained a non-converged "
    "S2 scenario."
)

print(
    "Therefore the enormous numerical loading values from that "
    "run were rejected."
)

print()
print(
    "Saved:"
)

print(
    f"  {OUTPUT_NETWORK_PATH}"
)

print(
    f"  {OUTPUT_RESULTS_PATH}"
)

print()
print(
    "NEXT:"
)

print(
    "Run a SEPARATE validated AC power-flow analysis on the "
    "iteration-2 network."
)

print(
    "Do NOT assume the reinforcement solved the bottlenecks "
    "until every scenario converges."
)

print()
print("=" * 70)