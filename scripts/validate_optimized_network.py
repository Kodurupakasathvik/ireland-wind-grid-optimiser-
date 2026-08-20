import os
import numpy as np
import pandas as pd
import pypsa


# ================================================================
# CONFIGURATION
# ================================================================

ORIGINAL_NETWORK = "data/processed/eirgrid_interconnected_scenarios.nc"
OPTIMIZED_NETWORK = "data/processed/eirgrid_optimized_network.nc"
SCENARIOS_FILE = "data/processed/selected_operating_scenarios.csv"
TARGETS_FILE = "data/processed/optimization_targets.csv"
OPTIMIZATION_RESULTS_FILE = "data/processed/transmission_optimization_results.csv"

OUTPUT_FILE = "data/processed/optimized_network_validation.csv"


# ================================================================
# HELPERS
# ================================================================

def clean_columns(df):
    """
    Normalize column names so small differences in capitalization
    or spacing do not break the script.
    """
    df = df.copy()
    df.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def find_column(df, candidates, required=True):
    """
    Find a column using case-insensitive matching.
    """
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:
        return np.nan


def get_line_loading(network, snapshot):
    """
    Calculate AC power-flow line loading percentage.

    Loading is calculated using:
        abs(s0) / s_nom * 100

    where s0 is apparent power flow at the line's from-end.
    """

    try:
        network.pf(
            snapshots=[snapshot],
            x_tol=1e-6,
            use_seed=True
        )
    except Exception:
        return None, None, False

    # Check that AC power flow actually produced usable values.
    if not hasattr(network.lines_t, "p0"):
        return None, None, False

    if snapshot not in network.lines_t.p0.index:
        return None, None, False

    p0 = network.lines_t.p0.loc[snapshot]

    # q0 may not exist in some cases.
    if hasattr(network.lines_t, "q0"):
        q0 = network.lines_t.q0.loc[snapshot]
    else:
        q0 = pd.Series(0.0, index=p0.index)

    s_nom = network.lines.s_nom

    # Apparent power
    apparent_power = np.sqrt(
        np.square(p0.astype(float)) +
        np.square(q0.astype(float))
    )

    loading = (
        apparent_power /
        s_nom.replace(0, np.nan)
    ) * 100.0

    loading = loading.replace([np.inf, -np.inf], np.nan)

    # Reject clearly non-physical numerical explosions.
    if loading.dropna().empty:
        return None, None, False

    max_loading = loading.max()

    if not np.isfinite(max_loading):
        return None, None, False

    if max_loading > 1e6:
        return None, None, False

    return loading, apparent_power, True


def validate_network(network, snapshot):
    """
    Run power flow and return line-loading information.
    """

    try:
        loading, flows, valid = get_line_loading(
            network,
            snapshot
        )

        if not valid:
            return {
                "valid": False,
                "max_loading": np.nan,
                "overloaded_lines": np.nan,
                "line_loading": None,
            }

        overloaded = int(
            (loading > 100.0).sum()
        )

        return {
            "valid": True,
            "max_loading": float(loading.max()),
            "overloaded_lines": overloaded,
            "line_loading": loading,
        }

    except Exception:
        return {
            "valid": False,
            "max_loading": np.nan,
            "overloaded_lines": np.nan,
            "line_loading": None,
        }


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("       IRELAND GRID - OPTIMIZED NETWORK VALIDATION")
print("=" * 70)


# ================================================================
# LOAD NETWORKS
# ================================================================

print()
print("Loading original interconnected network:")
print(ORIGINAL_NETWORK)

if not os.path.exists(ORIGINAL_NETWORK):
    raise FileNotFoundError(
        f"Original network not found:\n{ORIGINAL_NETWORK}"
    )

original = pypsa.Network(ORIGINAL_NETWORK)

print("OK: Original network loaded.")


print()
print("Loading optimized network:")
print(OPTIMIZED_NETWORK)

if not os.path.exists(OPTIMIZED_NETWORK):
    raise FileNotFoundError(
        f"Optimized network not found:\n{OPTIMIZED_NETWORK}"
    )

optimized = pypsa.Network(OPTIMIZED_NETWORK)

print("OK: Optimized network loaded.")


# ================================================================
# LOAD SCENARIOS
# ================================================================

print()
print("Loading EirGrid scenarios:")
print(SCENARIOS_FILE)

if not os.path.exists(SCENARIOS_FILE):
    raise FileNotFoundError(
        f"Scenario file not found:\n{SCENARIOS_FILE}"
    )

scenarios_df = pd.read_csv(SCENARIOS_FILE)
scenarios_df = clean_columns(scenarios_df)

print(
    f"OK: {len(scenarios_df)} scenarios loaded."
)


# ================================================================
# LOAD OPTIMIZATION TARGETS
# ================================================================

print()
print("Loading optimization targets:")
print(TARGETS_FILE)

if not os.path.exists(TARGETS_FILE):
    raise FileNotFoundError(
        f"Optimization targets file not found:\n{TARGETS_FILE}"
    )

targets_df = pd.read_csv(TARGETS_FILE)
targets_df = clean_columns(targets_df)

print(
    f"OK: {len(targets_df)} optimization targets loaded."
)


# ================================================================
# LOAD ACTUAL REINFORCEMENT RESULTS
# ================================================================

print()
print("Loading transmission optimization results:")
print(OPTIMIZATION_RESULTS_FILE)

if not os.path.exists(OPTIMIZATION_RESULTS_FILE):
    raise FileNotFoundError(
        f"Transmission optimization results not found:\n"
        f"{OPTIMIZATION_RESULTS_FILE}"
    )

optimization_df = pd.read_csv(
    OPTIMIZATION_RESULTS_FILE
)

optimization_df = clean_columns(
    optimization_df
)

print(
    f"OK: {len(optimization_df)} reinforcement results loaded."
)


# ================================================================
# NETWORK COMPARISON
# ================================================================

print()
print("-" * 70)
print("NETWORK COMPARISON")
print("-" * 70)

print()
print("ORIGINAL NETWORK")
print(f"Buses        : {len(original.buses)}")
print(f"Lines        : {len(original.lines)}")
print(f"Transformers : {len(original.transformers)}")
print(f"Generators   : {len(original.generators)}")
print(f"Loads        : {len(original.loads)}")
print(f"Links        : {len(original.links)}")

print()
print("OPTIMIZED NETWORK")
print(f"Buses        : {len(optimized.buses)}")
print(f"Lines        : {len(optimized.lines)}")
print(f"Transformers : {len(optimized.transformers)}")
print(f"Generators   : {len(optimized.generators)}")
print(f"Loads        : {len(optimized.loads)}")
print(f"Links        : {len(optimized.links)}")


# ================================================================
# FIND REINFORCEMENT COLUMNS
# ================================================================

line_col = find_column(
    optimization_df,
    [
        "Line",
        "line",
        "Line_ID",
        "Line_Name"
    ]
)

original_capacity_col = find_column(
    optimization_df,
    [
        "Original_Capacity_MW",
        "Original_Capacity",
        "Original_s_nom",
        "Original_s_nom_MW"
    ],
    required=False
)

added_capacity_col = find_column(
    optimization_df,
    [
        "Added_Capacity_MW",
        "Added_Capacity",
        "Added_MW"
    ]
)

new_capacity_col = find_column(
    optimization_df,
    [
        "Optimized_Capacity_MW",
"New_Capacity_MW",
"New_Capacity",
"New_s_nom",
"New_s_nom_MW"
    ]
)


# ================================================================
# TARGET INFORMATION
# ================================================================

target_line_col = find_column(
    targets_df,
    [
        "Line",
        "line",
        "Line_ID",
        "Line_Name"
    ]
)

target_priority_col = find_column(
    targets_df,
    [
        "Priority",
        "priority"
    ],
    required=False
)


# ================================================================
# OPTIMIZATION TARGET SUMMARY
# ================================================================

print()
print("-" * 70)
print("OPTIMIZATION TARGETS")
print("-" * 70)

target_lines = []

for _, row in targets_df.iterrows():

    line = str(row[target_line_col])

    target_lines.append(line)

    if target_priority_col is not None:
        priority = str(row[target_priority_col])
    else:
        priority = "UNKNOWN"

    print(
        f"{line} -> {priority}"
    )


# ================================================================
# REINFORCEMENT SUMMARY
# ================================================================

print()
print("-" * 70)
print("TRANSMISSION REINFORCEMENTS")
print("-" * 70)

reinforcement_records = []

for _, row in optimization_df.iterrows():

    line = str(row[line_col])

    added = safe_float(
        row[added_capacity_col]
    )

    new_capacity = safe_float(
        row[new_capacity_col]
    )

    if original_capacity_col is not None:
        original_capacity = safe_float(
            row[original_capacity_col]
        )
    else:

        if line in original.lines.index:
            original_capacity = safe_float(
                original.lines.loc[line, "s_nom"]
            )
        else:
            original_capacity = np.nan

    reinforcement_records.append(
        {
            "Line": line,
            "Original_Capacity_MW": original_capacity,
            "Added_Capacity_MW": added,
            "New_Capacity_MW": new_capacity
        }
    )

    print()
    print(f"Line : {line}")
    print(
        f"Original capacity : "
        f"{original_capacity:.2f} MW"
    )
    print(
        f"Added capacity    : "
        f"{added:.2f} MW"
    )
    print(
        f"New capacity      : "
        f"{new_capacity:.2f} MW"
    )


reinforcement_df = pd.DataFrame(
    reinforcement_records
)


# ================================================================
# SNAPSHOT DETECTION
# ================================================================

print()
print("-" * 70)
print("SCENARIO VALIDATION")
print("-" * 70)

print()
print(
    f"Total scenarios : "
    f"{len(scenarios_df)}"
)

# Use scenario names from selected_operating_scenarios.csv.
scenario_column = find_column(
    scenarios_df,
    [
        "Scenario",
        "scenario",
        "Scenario_Name",
        "Scenario_ID"
    ],
    required=False
)

if scenario_column is not None:

    snapshots = [
        str(x)
        for x in scenarios_df[scenario_column]
        if str(x) in original.snapshots
    ]

else:

    snapshots = [
        str(x)
        for x in original.snapshots
    ]


# If the CSV names cannot be detected, use network snapshots.
if len(snapshots) == 0:
    snapshots = [
        str(x)
        for x in original.snapshots
    ]


print(
    f"Scenarios detected : "
    f"{len(snapshots)}"
)


# ================================================================
# POWER-FLOW VALIDATION
# ================================================================

results = []

valid_original = 0
valid_optimized = 0

invalid_original = []
invalid_optimized = []


for snapshot in snapshots:

    print()
    print("=" * 70)
    print(f"SCENARIO: {snapshot}")
    print("=" * 70)

    # ------------------------------------------------------------
    # ORIGINAL
    # ------------------------------------------------------------

    print()
    print("ORIGINAL NETWORK POWER FLOW")

    original_result = validate_network(
        original,
        snapshot
    )

    if original_result["valid"]:

        valid_original += 1

        print(
            f"Maximum line loading : "
            f"{original_result['max_loading']:.2f}%"
        )

        print(
            f"Overloaded lines     : "
            f"{original_result['overloaded_lines']}"
        )

    else:

        invalid_original.append(snapshot)

        print(
            "POWER FLOW INVALID / "
            "NON-CONVERGED"
        )

    # ------------------------------------------------------------
    # OPTIMIZED
    # ------------------------------------------------------------

    print()
    print("OPTIMIZED NETWORK POWER FLOW")

    optimized_result = validate_network(
        optimized,
        snapshot
    )

    if optimized_result["valid"]:

        valid_optimized += 1

        print(
            f"Maximum line loading : "
            f"{optimized_result['max_loading']:.2f}%"
        )

        print(
            f"Overloaded lines     : "
            f"{optimized_result['overloaded_lines']}"
        )

    else:

        invalid_optimized.append(snapshot)

        print(
            "POWER FLOW INVALID / "
            "NON-CONVERGED"
        )


    # ------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------

    if (
        original_result["valid"]
        and optimized_result["valid"]
    ):

        reduction = (
            original_result["max_loading"]
            -
            optimized_result["max_loading"]
        )

        if original_result["max_loading"] != 0:

            reduction_percent = (
                reduction /
                original_result["max_loading"]
            ) * 100.0

        else:

            reduction_percent = np.nan

        overload_reduction = (
            original_result["overloaded_lines"]
            -
            optimized_result["overloaded_lines"]
        )

        print()
        print("COMPARISON")

        print(
            f"Maximum loading reduction : "
            f"{reduction:.2f} percentage points"
        )

        print(
            f"Relative reduction        : "
            f"{reduction_percent:.2f}%"
        )

        print(
            f"Overloaded-line reduction : "
            f"{overload_reduction}"
        )

        if (
            optimized_result["max_loading"]
            <
            original_result["max_loading"]
        ):

            status = "IMPROVED"

        elif (
            optimized_result["max_loading"]
            >
            original_result["max_loading"]
        ):

            status = "WORSENED"

        else:

            status = "UNCHANGED"

    else:

        reduction = np.nan
        reduction_percent = np.nan
        overload_reduction = np.nan
        status = "INVALID / NON-CONVERGED"


    results.append(
        {
            "Scenario": snapshot,

            "Original_Max_Line_Loading_Percent":
                original_result["max_loading"],

            "Optimized_Max_Line_Loading_Percent":
                optimized_result["max_loading"],

            "Max_Loading_Reduction_Percentage_Points":
                reduction,

            "Relative_Max_Loading_Reduction_Percent":
                reduction_percent,

            "Original_Overloaded_Lines":
                original_result["overloaded_lines"],

            "Optimized_Overloaded_Lines":
                optimized_result["overloaded_lines"],

            "Overloaded_Line_Reduction":
                overload_reduction,

            "Original_Valid":
                original_result["valid"],

            "Optimized_Valid":
                optimized_result["valid"],

            "Status":
                status
        }
    )


# ================================================================
# SAVE RESULTS
# ================================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ================================================================
# SUMMARY
# ================================================================

print()
print("=" * 70)
print("        OPTIMIZED NETWORK VALIDATION SUMMARY")
print("=" * 70)

print()
print(
    f"Total scenarios          : "
    f"{len(snapshots)}"
)

print(
    f"Valid original scenarios : "
    f"{valid_original}"
)

print(
    f"Valid optimized scenarios: "
    f"{valid_optimized}"
)

print(
    f"Invalid original         : "
    f"{len(invalid_original)}"
)

print(
    f"Invalid optimized        : "
    f"{len(invalid_optimized)}"
)


if invalid_original:

    print()
    print("Original invalid scenarios:")

    for scenario in invalid_original:
        print(f"  {scenario}")


if invalid_optimized:

    print()
    print("Optimized invalid scenarios:")

    for scenario in invalid_optimized:
        print(f"  {scenario}")


# ================================================================
# VALID COMPARISON SUMMARY
# ================================================================

valid_comparisons = results_df[
    (results_df["Original_Valid"] == True)
    &
    (results_df["Optimized_Valid"] == True)
].copy()


print()

if len(valid_comparisons) > 0:

    average_original = (
        valid_comparisons[
            "Original_Max_Line_Loading_Percent"
        ].mean()
    )

    average_optimized = (
        valid_comparisons[
            "Optimized_Max_Line_Loading_Percent"
        ].mean()
    )

    average_reduction = (
        valid_comparisons[
            "Max_Loading_Reduction_Percentage_Points"
        ].mean()
    )

    print("-" * 70)
    print("VALID SCENARIO COMPARISON")
    print("-" * 70)

    print()
    print(
        f"Average original maximum loading : "
        f"{average_original:.2f}%"
    )

    print(
        f"Average optimized maximum loading : "
        f"{average_optimized:.2f}%"
    )

    print(
        f"Average reduction                 : "
        f"{average_reduction:.2f} percentage points"
    )

    print()
    print("Scenario results:")

    for _, row in valid_comparisons.iterrows():

        print(
            f"  {row['Scenario']:<35}"
            f"{row['Original_Max_Line_Loading_Percent']:>8.2f}% -> "
            f"{row['Optimized_Max_Line_Loading_Percent']:>8.2f}% "
            f"({row['Status']})"
        )


# ================================================================
# TARGET-SPECIFIC CAPACITY CHECK
# ================================================================

print()
print("-" * 70)
print("TARGET CAPACITY CHECK")
print("-" * 70)

for _, row in reinforcement_df.iterrows():

    line = row["Line"]

    if line in optimized.lines.index:

        actual_capacity = safe_float(
            optimized.lines.loc[line, "s_nom"]
        )

        expected_capacity = row[
            "New_Capacity_MW"
        ]

        difference = (
            actual_capacity -
            expected_capacity
        )

        print()
        print(f"Line : {line}")
        print(
            f"Expected optimized capacity : "
            f"{expected_capacity:.2f} MW"
        )
        print(
            f"Actual optimized capacity   : "
            f"{actual_capacity:.2f} MW"
        )
        print(
            f"Difference                  : "
            f"{difference:.2f} MW"
        )

        if abs(difference) < 1e-6:
            print("Capacity check              : PASS")
        else:
            print("Capacity check              : CHECK")

    else:

        print()
        print(
            f"Line : {line}"
        )
        print(
            "Capacity check : LINE NOT FOUND "
            "IN OPTIMIZED NETWORK"
        )


# ================================================================
# FINAL STATUS
# ================================================================

print()
print("=" * 70)

if len(valid_comparisons) > 0:

    all_improved_or_unchanged = all(
        valid_comparisons["Status"].isin(
            ["IMPROVED", "UNCHANGED"]
        )
    )

    if all_improved_or_unchanged:
        print(
            "STATUS: OPTIMIZATION VALIDATED "
            "FOR VALID SCENARIOS"
        )
    else:
        print(
            "STATUS: OPTIMIZATION REQUIRES REVIEW"
        )

else:

    print(
        "STATUS: NO VALID SCENARIO COMPARISONS"
    )

print("=" * 70)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("=" * 70)
print("        OPTIMIZED NETWORK VALIDATION COMPLETE")
print("=" * 70)

print()
print("NEXT:")
print(
    "Analyse whether the reinforcement targets "
    "actually reduce recurring bottleneck loading."
)
