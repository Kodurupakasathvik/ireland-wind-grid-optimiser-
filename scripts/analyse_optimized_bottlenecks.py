"""
IRELAND GRID - POST-OPTIMIZATION BOTTLENECK ANALYSIS

Purpose
-------
Compare original and optimized AC power-flow results and identify:

1. Remaining overloaded lines after optimization
2. Loading reduction on the original recurring bottlenecks
3. Residual bottleneck frequency across valid scenarios
4. Worst-case and average optimized loading
5. Whether the reinforcement targets are still bottlenecks

Inputs
------
data/processed/eirgrid_interconnected_powerflow.csv
data/processed/optimized_network_validation.csv
data/processed/recurring_transmission_bottlenecks.csv
data/processed/optimization_targets.csv

Output
------
data/processed/optimized_bottleneck_analysis.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

ORIGINAL_PF_FILE = DATA_DIR / "eirgrid_interconnected_powerflow.csv"
VALIDATION_FILE = DATA_DIR / "optimized_network_validation.csv"
RECURRING_FILE = DATA_DIR / "recurring_transmission_bottlenecks.csv"
TARGETS_FILE = DATA_DIR / "optimization_targets.csv"

OUTPUT_FILE = DATA_DIR / "optimized_bottleneck_analysis.csv"


# ======================================================================
# HELPERS
# ======================================================================

def find_column(df, candidates, description):
    """
    Find a column using case-insensitive matching and normalized names.
    """

    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def clean_line_name(value):
    return str(value).strip()


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except (TypeError, ValueError):
        return np.nan


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("     IRELAND GRID - POST-OPTIMIZATION BOTTLENECK ANALYSIS")
print("=" * 70)


# ======================================================================
# LOAD DATA
# ======================================================================

print()
print("Loading original power-flow results:")
print(ORIGINAL_PF_FILE)

if not ORIGINAL_PF_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {ORIGINAL_PF_FILE}"
    )

original_pf = pd.read_csv(ORIGINAL_PF_FILE)

print("OK: Original power-flow results loaded.")


print()
print("Loading optimized validation results:")
print(VALIDATION_FILE)

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {VALIDATION_FILE}"
    )

validation = pd.read_csv(VALIDATION_FILE)

print("OK: Optimized validation results loaded.")


print()
print("Loading recurring bottleneck analysis:")
print(RECURRING_FILE)

if not RECURRING_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {RECURRING_FILE}"
    )

recurring = pd.read_csv(RECURRING_FILE)

print("OK: Recurring bottleneck data loaded.")


print()
print("Loading optimization targets:")
print(TARGETS_FILE)

if not TARGETS_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {TARGETS_FILE}"
    )

targets = pd.read_csv(TARGETS_FILE)

print("OK: Optimization targets loaded.")


# ======================================================================
# IDENTIFY COLUMNS
# ======================================================================

# Original recurring bottleneck columns
recurring_line_col = find_column(
    recurring,
    ["Line", "line"],
    "recurring bottleneck line column"
)

recurring_avg_col = find_column(
    recurring,
    [
        "Average_Loading_Percent",
        "Average_Loading",
        "AverageLoadingPercent",
    ],
    "recurring average loading column"
)

recurring_max_col = find_column(
    recurring,
    [
        "Maximum_Loading_Percent",
        "Maximum_Loading",
        "MaximumLoadingPercent",
    ],
    "recurring maximum loading column"
)

recurring_min_col = find_column(
    recurring,
    [
        "Minimum_Loading_Percent",
        "Minimum_Loading",
        "MinimumLoadingPercent",
    ],
    "recurring minimum loading column"
)

recurring_frequency_col = find_column(
    recurring,
    [
        "Overload_Frequency_Percent",
        "Overload_Frequency",
        "OverloadFrequencyPercent",
    ],
    "recurring overload frequency column"
)


# Validation scenario columns
scenario_col = find_column(
    validation,
    ["Scenario", "scenario"],
    "validation scenario column"
)

original_max_col = find_column(
    validation,
    [
        "Original_Max_Line_Loading_Percent",
        "Original_Maximum_Loading_Percent",
        "Original_Max_Loading_Percent",
        "Original_Max_Loading",
    ],
    "original maximum loading column"
)

optimized_max_col = find_column(
    validation,
    [
        "Optimized_Max_Line_Loading_Percent",
        "Optimized_Maximum_Loading_Percent",
        "Optimized_Max_Loading_Percent",
        "Optimized_Max_Loading",
    ],
    "optimized maximum loading column"
)

original_overloaded_col = find_column(
    validation,
    [
        "Original_Overloaded_Lines",
        "Original_Overloaded_Line_Count",
    ],
    "original overloaded-line column"
)

optimized_overloaded_col = find_column(
    validation,
    [
        "Optimized_Overloaded_Lines",
        "Optimized_Overloaded_Line_Count",
    ],
    "optimized overloaded-line column"
)


# Target columns
target_line_col = find_column(
    targets,
    ["Line", "line"],
    "optimization target line column"
)

target_priority_col = find_column(
    targets,
    ["Priority", "priority"],
    "optimization target priority column"
)


# ======================================================================
# VALIDATION SCENARIOS
# ======================================================================

print()
print("-" * 70)
print("VALID SCENARIOS")
print("-" * 70)

validation[scenario_col] = validation[scenario_col].astype(str).str.strip()

valid_mask = (
    pd.to_numeric(
        validation[original_max_col],
        errors="coerce"
    ).notna()
    &
    pd.to_numeric(
        validation[optimized_max_col],
        errors="coerce"
    ).notna()
)

valid_validation = validation.loc[valid_mask].copy()

valid_validation[original_max_col] = pd.to_numeric(
    valid_validation[original_max_col],
    errors="coerce"
)

valid_validation[optimized_max_col] = pd.to_numeric(
    valid_validation[optimized_max_col],
    errors="coerce"
)

print(f"Total scenarios       : {len(validation)}")
print(f"Valid scenarios       : {len(valid_validation)}")
print(f"Invalid/non-converged : {len(validation) - len(valid_validation)}")

if len(validation) > len(valid_validation):

    invalid = validation.loc[~valid_mask, scenario_col].tolist()

    print()
    print("Excluded scenarios:")

    for scenario in invalid:
        print(f"  {scenario}")


# ======================================================================
# SCENARIO-LEVEL IMPROVEMENT
# ======================================================================

print()
print("-" * 70)
print("SCENARIO-LEVEL IMPROVEMENT")
print("-" * 70)

scenario_results = []

for _, row in valid_validation.iterrows():

    scenario = row[scenario_col]

    original_loading = safe_float(
        row[original_max_col]
    )

    optimized_loading = safe_float(
        row[optimized_max_col]
    )

    reduction = (
        original_loading - optimized_loading
    )

    if original_loading != 0:
        relative_reduction = (
            reduction / original_loading
        ) * 100.0
    else:
        relative_reduction = np.nan

    original_overloaded = safe_float(
        row[original_overloaded_col]
    )

    optimized_overloaded = safe_float(
        row[optimized_overloaded_col]
    )

    overloaded_reduction = (
        original_overloaded - optimized_overloaded
    )

    if optimized_loading <= 100.0:
        status = "NO THERMAL OVERLOAD"
    else:
        status = "RESIDUAL OVERLOAD"

    scenario_results.append(
        {
            "Scenario": scenario,
            "Original_Max_Loading_Percent": original_loading,
            "Optimized_Max_Loading_Percent": optimized_loading,
            "Loading_Reduction_Percentage_Points": reduction,
            "Relative_Reduction_Percent": relative_reduction,
            "Original_Overloaded_Lines": original_overloaded,
            "Optimized_Overloaded_Lines": optimized_overloaded,
            "Overloaded_Line_Reduction": overloaded_reduction,
            "Status": status,
        }
    )

    print()
    print(f"{scenario}")
    print(
        f"  Loading : "
        f"{original_loading:.2f}% -> "
        f"{optimized_loading:.2f}%"
    )
    print(
        f"  Reduction : "
        f"{reduction:.2f} percentage points"
    )
    print(
        f"  Relative reduction : "
        f"{relative_reduction:.2f}%"
    )
    print(
        f"  Overloaded lines : "
        f"{int(original_overloaded)} -> "
        f"{int(optimized_overloaded)}"
    )
    print(f"  Status : {status}")


scenario_results_df = pd.DataFrame(scenario_results)


# ======================================================================
# OVERALL SCENARIO SUMMARY
# ======================================================================

print()
print("=" * 70)
print("             OVERALL VALIDATION IMPROVEMENT")
print("=" * 70)

if len(scenario_results_df) > 0:

    avg_original = (
        scenario_results_df[
            "Original_Max_Loading_Percent"
        ].mean()
    )

    avg_optimized = (
        scenario_results_df[
            "Optimized_Max_Loading_Percent"
        ].mean()
    )

    avg_reduction = (
        scenario_results_df[
            "Loading_Reduction_Percentage_Points"
        ].mean()
    )

    no_overload_count = (
        scenario_results_df["Status"]
        .eq("NO THERMAL OVERLOAD")
        .sum()
    )

    residual_count = (
        scenario_results_df["Status"]
        .eq("RESIDUAL OVERLOAD")
        .sum()
    )

    print()
    print(
        f"Average original maximum loading : "
        f"{avg_original:.2f}%"
    )

    print(
        f"Average optimized maximum loading : "
        f"{avg_optimized:.2f}%"
    )

    print(
        f"Average loading reduction : "
        f"{avg_reduction:.2f} percentage points"
    )

    print(
        f"Scenarios with no thermal overload : "
        f"{no_overload_count}/{len(scenario_results_df)}"
    )

    print(
        f"Scenarios with residual overload   : "
        f"{residual_count}/{len(scenario_results_df)}"
    )


# ======================================================================
# TARGET SUMMARY
# ======================================================================

print()
print("=" * 70)
print("             REINFORCEMENT TARGET ANALYSIS")
print("=" * 70)

target_lines = set(
    targets[target_line_col]
    .astype(str)
    .str.strip()
)

recurring_lines = set(
    recurring[recurring_line_col]
    .astype(str)
    .str.strip()
)

print()
print(f"Optimization targets : {len(target_lines)}")
print(f"Recurring bottlenecks: {len(recurring_lines)}")


# ======================================================================
# BUILD TARGET ANALYSIS
# ======================================================================

target_results = []

for _, target_row in targets.iterrows():

    line = clean_line_name(
        target_row[target_line_col]
    )

    priority = str(
        target_row[target_priority_col]
    ).strip()

    recurring_match = recurring[
        recurring[recurring_line_col]
        .astype(str)
        .str.strip()
        == line
    ]

    if recurring_match.empty:

        print()
        print(f"WARNING: No recurring record found for {line}")

        continue

    recurring_row = recurring_match.iloc[0]

    original_average = safe_float(
        recurring_row[recurring_avg_col]
    )

    original_maximum = safe_float(
        recurring_row[recurring_max_col]
    )

    original_minimum = safe_float(
        recurring_row[recurring_min_col]
    )

    original_frequency = safe_float(
        recurring_row[recurring_frequency_col]
    )

    target_results.append(
        {
            "Line": line,
            "Priority": priority,
            "Original_Average_Loading_Percent":
                original_average,
            "Original_Maximum_Loading_Percent":
                original_maximum,
            "Original_Minimum_Loading_Percent":
                original_minimum,
            "Original_Overload_Frequency_Percent":
                original_frequency,
        }
    )


target_summary = pd.DataFrame(target_results)


# ======================================================================
# IMPORTANT: LINE-LEVEL POST-OPTIMIZATION ANALYSIS
# ======================================================================

"""
The validation CSV contains scenario-level maximum loading,
not individual line flows.

Therefore, we do NOT invent post-optimization loading for
individual lines.

Instead, this stage reports:

- original line-level bottleneck severity
- system-level post-optimization maximum loading
- residual system overload status

A future detailed line-flow extraction can provide exact
post-optimization loading for every individual corridor.
"""


# ======================================================================
# RESIDUAL SYSTEM BOTTLENECK ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("             RESIDUAL BOTTLENECK ANALYSIS")
print("=" * 70)

if len(scenario_results_df) > 0:

    residual_scenarios = scenario_results_df[
        scenario_results_df["Optimized_Max_Loading_Percent"] > 100.0
    ]

    print()

    if residual_scenarios.empty:

        print(
            "NO SYSTEM-LEVEL THERMAL OVERLOAD REMAINS "
            "IN VALID SCENARIOS."
        )

    else:

        print(
            f"Residual thermal overload exists in "
            f"{len(residual_scenarios)}/"
            f"{len(scenario_results_df)} valid scenarios."
        )

        print()

        for _, row in residual_scenarios.iterrows():

            print(
                f"  {row['Scenario']}: "
                f"{row['Optimized_Max_Loading_Percent']:.2f}%"
            )


# ======================================================================
# FINAL CLASSIFICATION
# ======================================================================

print()
print("=" * 70)
print("             OPTIMIZATION EFFECTIVENESS")
print("=" * 70)

if len(scenario_results_df) == 0:

    effectiveness = "UNDETERMINED"

else:

    all_improved = (
        scenario_results_df[
            "Loading_Reduction_Percentage_Points"
        ] > 0
    ).all()

    all_below_or_equal_100 = (
        scenario_results_df[
            "Optimized_Max_Loading_Percent"
        ] <= 100.0
    ).all()

    if all_below_or_equal_100:

        effectiveness = "FULL THERMAL RELIEF"

    elif all_improved:

        effectiveness = "IMPROVED WITH RESIDUAL OVERLOADS"

    else:

        effectiveness = "PARTIAL / INCONSISTENT IMPROVEMENT"


print()
print(f"STATUS: {effectiveness}")


# ======================================================================
# SAVE RESULTS
# ======================================================================

output_rows = []

for _, row in scenario_results_df.iterrows():

    output_rows.append(
        {
            "Record_Type": "Scenario",
            "Scenario": row["Scenario"],
            "Line": "",
            "Priority": "",
            "Original_Average_Loading_Percent": np.nan,
            "Original_Maximum_Loading_Percent":
                row["Original_Max_Loading_Percent"],
            "Optimized_Maximum_Loading_Percent":
                row["Optimized_Max_Loading_Percent"],
            "Loading_Reduction_Percentage_Points":
                row["Loading_Reduction_Percentage_Points"],
            "Relative_Reduction_Percent":
                row["Relative_Reduction_Percent"],
            "Original_Overload_Frequency_Percent": np.nan,
            "Original_Overloaded_Lines":
                row["Original_Overloaded_Lines"],
            "Optimized_Overloaded_Lines":
                row["Optimized_Overloaded_Lines"],
            "Status": row["Status"],
        }
    )


for _, row in target_summary.iterrows():

    output_rows.append(
        {
            "Record_Type": "Target",
            "Scenario": "",
            "Line": row["Line"],
            "Priority": row["Priority"],
            "Original_Average_Loading_Percent":
                row["Original_Average_Loading_Percent"],
            "Original_Maximum_Loading_Percent":
                row["Original_Maximum_Loading_Percent"],
            "Optimized_Maximum_Loading_Percent": np.nan,
            "Loading_Reduction_Percentage_Points": np.nan,
            "Relative_Reduction_Percent": np.nan,
            "Original_Overload_Frequency_Percent":
                row["Original_Overload_Frequency_Percent"],
            "Original_Overloaded_Lines": np.nan,
            "Optimized_Overloaded_Lines": np.nan,
            "Status": "TARGET_REINFORCED",
        }
    )


output_df = pd.DataFrame(output_rows)

output_df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.2f"
)

print()
print("=" * 70)
print("       POST-OPTIMIZATION BOTTLENECK ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("IMPORTANT:")
print(
    "S2_PEAK_DEMAND remains excluded because its AC power "
    "flow did not converge."
)

print()
print("NEXT:")
print(
    "Perform detailed post-optimization line-flow analysis "
    "to identify exactly which corridors remain overloaded."
)

print("=" * 70)