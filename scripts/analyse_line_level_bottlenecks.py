from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


# ================================================================
# IRELAND GRID - DETAILED LINE-LEVEL BOTTLENECK ANALYSIS
# CORRECTED VERSION
#
# IMPORTANT:
# - Non-converged scenarios are completely excluded.
# - S2_PEAK_DEMAND is expected to be excluded because its AC
#   power flow does not converge.
# - No line statistics are calculated from invalid scenarios.
# ================================================================


# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------

BASE_DIR = Path("data/processed")

ORIGINAL_NETWORK = BASE_DIR / "eirgrid_interconnected_scenarios.nc"
OPTIMIZED_NETWORK = BASE_DIR / "eirgrid_optimized_network.nc"

RECURRING_BOTTLENECKS = BASE_DIR / "recurring_transmission_bottlenecks.csv"
OPTIMIZATION_TARGETS = BASE_DIR / "optimization_targets.csv"

OUTPUT_COMPARISON = BASE_DIR / "line_level_bottleneck_comparison.csv"
OUTPUT_RESIDUAL = BASE_DIR / "residual_transmission_bottlenecks.csv"


# ----------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------

LOADING_LIMIT = 100.0

# Any result above this is treated as numerically invalid.
# This protects against non-converged Newton-Raphson explosions.
MAX_REASONABLE_LOADING = 10000.0

# Expected invalid scenario from previous validation.
# The convergence check remains the actual authority.
KNOWN_INVALID_SCENARIOS = {
    "S2_PEAK_DEMAND"
}


# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------

print("=" * 70)
print("       IRELAND GRID - DETAILED LINE-LEVEL BOTTLENECK ANALYSIS")
print("                 CORRECTED VALIDATION VERSION")
print("=" * 70)


# ----------------------------------------------------------------
# LOAD NETWORKS
# ----------------------------------------------------------------

print("\nLoading original interconnected network:")
print(ORIGINAL_NETWORK)

original = pypsa.Network(str(ORIGINAL_NETWORK))

print("OK: Original network loaded.")


print("\nLoading optimized network:")
print(OPTIMIZED_NETWORK)

optimized = pypsa.Network(str(OPTIMIZED_NETWORK))

print("OK: Optimized network loaded.")


# ----------------------------------------------------------------
# LOAD SUPPORTING DATA
# ----------------------------------------------------------------

print("\nLoading recurring bottlenecks:")
print(RECURRING_BOTTLENECKS)

recurring = pd.read_csv(RECURRING_BOTTLENECKS)

print("OK: Recurring bottlenecks loaded.")


print("\nLoading optimization targets:")
print(OPTIMIZATION_TARGETS)

targets = pd.read_csv(OPTIMIZATION_TARGETS)

print("OK: Optimization targets loaded.")


# ----------------------------------------------------------------
# NETWORK SUMMARY
# ----------------------------------------------------------------

print("\n" + "-" * 70)
print("NETWORK")
print("-" * 70)

print(f"Original buses        : {len(original.buses)}")
print(f"Original lines        : {len(original.lines)}")
print(f"Original transformers : {len(original.transformers)}")
print(f"Original generators   : {len(original.generators)}")
print(f"Original loads        : {len(original.loads)}")

print()

print(f"Optimized buses        : {len(optimized.buses)}")
print(f"Optimized lines        : {len(optimized.lines)}")
print(f"Optimized transformers : {len(optimized.transformers)}")
print(f"Optimized generators   : {len(optimized.generators)}")
print(f"Optimized loads        : {len(optimized.loads)}")


# ----------------------------------------------------------------
# SCENARIOS
# ----------------------------------------------------------------

original_scenarios = list(original.snapshots)
optimized_scenarios = list(optimized.snapshots)

common_scenarios = [
    s for s in original_scenarios
    if s in optimized_scenarios
]

print("\n" + "-" * 70)
print("SCENARIOS")
print("-" * 70)

print(f"Original snapshots : {len(original_scenarios)}")
print(f"Optimized snapshots: {len(optimized_scenarios)}")

print("\nCommon scenarios:")

for scenario in common_scenarios:
    print(f"  {scenario}")


# ----------------------------------------------------------------
# LINE COVERAGE
# ----------------------------------------------------------------

original_lines = set(original.lines.index)
optimized_lines = set(optimized.lines.index)

common_lines = sorted(original_lines.intersection(optimized_lines))

print("\n" + "-" * 70)
print("LINE COVERAGE")
print("-" * 70)

print(f"Original lines : {len(original_lines)}")
print(f"Optimized lines: {len(optimized_lines)}")
print(f"Common lines   : {len(common_lines)}")


# ----------------------------------------------------------------
# TARGET LINES
# ----------------------------------------------------------------

target_lines = []

for column in targets.columns:

    for value in targets[column].dropna():

        value = str(value)

        if value in common_lines and value not in target_lines:
            target_lines.append(value)


# Fallback: explicitly known targets from previous optimization
known_targets = [
    "merged_way/257889771-220+1",
    "merged_way/1231251986-220+2",
    "way/343436171-220",
]

for line in known_targets:

    if line in common_lines and line not in target_lines:
        target_lines.append(line)


print("\n" + "-" * 70)
print("TARGET LINES")
print("-" * 70)

for line in target_lines:
    print(f"  {line}")


# ================================================================
# POWER-FLOW FUNCTION
# ================================================================

def run_ac_power_flow(network, scenario):
    """
    Run AC power flow for exactly one scenario.

    Returns:
        {
            "valid": bool,
            "loading": pd.Series or None,
            "reason": str
        }
    """

    try:

        # Make sure only the requested snapshot is active.
        network.set_snapshots([scenario])

        with warnings.catch_warnings(record=True):

            warnings.simplefilter("always")

            result = network.pf(
                snapshots=[scenario]
            )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # PyPSA may return a result object even when the nonlinear
        # solver did not actually converge.
        #
        # Therefore we independently inspect the numerical output.
        # --------------------------------------------------------

        loading = pd.Series(
            dtype=float,
            index=network.lines.index
        )

        # --------------------------------------------------------
        # Determine loading from apparent power.
        #
        # For AC lines:
        #
        # s_nom is the thermal rating.
        #
        # p0/q0 are the power-flow values.
        #
        # Loading is based on apparent power at each end.
        # --------------------------------------------------------

        if (
            scenario not in network.lines_t.p0.index
            or scenario not in network.lines_t.q0.index
        ):
            return {
                "valid": False,
                "loading": None,
                "reason": "Missing AC line-flow results"
            }

        p0 = network.lines_t.p0.loc[scenario]
        q0 = network.lines_t.q0.loc[scenario]

        p1 = network.lines_t.p1.loc[scenario]
        q1 = network.lines_t.q1.loc[scenario]

        s0 = np.sqrt(
            np.square(p0) +
            np.square(q0)
        )

        s1 = np.sqrt(
            np.square(p1) +
            np.square(q1)
        )

        apparent_power = pd.concat(
            [s0, s1],
            axis=1
        ).max(axis=1)

        ratings = network.lines["s_nom"]

        loading = (
            apparent_power /
            ratings.replace(0, np.nan)
        ) * 100.0

        # --------------------------------------------------------
        # Numerical validation
        # --------------------------------------------------------

        if loading.empty:
            return {
                "valid": False,
                "loading": None,
                "reason": "No line loading results"
            }

        if not np.isfinite(loading.values).all():
            return {
                "valid": False,
                "loading": None,
                "reason": "Non-finite line loading detected"
            }

        max_loading = float(loading.max())

        # This catches the gigantic values produced by the
        # non-converged S2 scenario.
        if max_loading > MAX_REASONABLE_LOADING:

            return {
                "valid": False,
                "loading": None,
                "reason": (
                    f"Numerically invalid loading "
                    f"({max_loading:.2e}%)"
                )
            }

        return {
            "valid": True,
            "loading": loading,
            "reason": "AC power flow valid"
        }

    except Exception as exc:

        return {
            "valid": False,
            "loading": None,
            "reason": str(exc)
        }


# ================================================================
# SCENARIO VALIDATION
# ================================================================

print("\n")
print("=" * 70)
print("                 AC POWER-FLOW ANALYSIS")
print("=" * 70)


valid_scenarios = []

scenario_results = []


for scenario in common_scenarios:

    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario}")
    print("=" * 70)

    # ------------------------------------------------------------
    # ORIGINAL
    # ------------------------------------------------------------

    print("\nORIGINAL NETWORK")

    original_result = run_ac_power_flow(
        original,
        scenario
    )

    if original_result["valid"]:

        original_loading = original_result["loading"]

        print("  AC power flow : VALID")
        print(
            f"  Maximum loading : "
            f"{original_loading.max():.2f}%"
        )

        print(
            f"  Overloaded lines: "
            f"{(original_loading > LOADING_LIMIT).sum()}"
        )

    else:

        original_loading = None

        print("  AC power flow : INVALID")
        print(
            f"  Reason : "
            f"{original_result['reason']}"
        )


    # ------------------------------------------------------------
    # OPTIMIZED
    # ------------------------------------------------------------

    print("\nOPTIMIZED NETWORK")

    optimized_result = run_ac_power_flow(
        optimized,
        scenario
    )

    if optimized_result["valid"]:

        optimized_loading = optimized_result["loading"]

        print("  AC power flow : VALID")
        print(
            f"  Maximum loading : "
            f"{optimized_loading.max():.2f}%"
        )

        print(
            f"  Overloaded lines: "
            f"{(optimized_loading > LOADING_LIMIT).sum()}"
        )

    else:

        optimized_loading = None

        print("  AC power flow : INVALID")
        print(
            f"  Reason : "
            f"{optimized_result['reason']}"
        )


    # ------------------------------------------------------------
    # SCENARIO VALIDITY
    # ------------------------------------------------------------

    scenario_valid = (
        original_result["valid"]
        and optimized_result["valid"]
    )

    if scenario_valid:

        valid_scenarios.append(scenario)

        print("\n  >>> SCENARIO INCLUDED IN LINE ANALYSIS <<<")

    else:

        print("\n  >>> SCENARIO EXCLUDED FROM LINE ANALYSIS <<<")

        if scenario in KNOWN_INVALID_SCENARIOS:

            print(
                f"  Known non-converged scenario: {scenario}"
            )


    scenario_results.append({
        "scenario": scenario,
        "original_valid": original_result["valid"],
        "optimized_valid": optimized_result["valid"],
        "original_loading": original_loading,
        "optimized_loading": optimized_loading,
    })


# ================================================================
# VALID SCENARIO SUMMARY
# ================================================================

invalid_scenarios = [
    s for s in common_scenarios
    if s not in valid_scenarios
]

print("\n")
print("=" * 70)
print("                 VALID SCENARIO SUMMARY")
print("=" * 70)

print(f"\nTotal scenarios       : {len(common_scenarios)}")
print(f"Valid scenarios       : {len(valid_scenarios)}")
print(f"Invalid/non-converged : {len(invalid_scenarios)}")

print("\nValid scenarios:")

for scenario in valid_scenarios:
    print(f"  {scenario}")

print("\nExcluded scenarios:")

if invalid_scenarios:

    for scenario in invalid_scenarios:
        print(f"  {scenario}")

else:
    print("  None")


# ================================================================
# LINE-LEVEL DATA COLLECTION
# ================================================================

print("\n")
print("=" * 70)
print("             BUILDING LINE-LEVEL DATASET")
print("=" * 70)


line_records = []


for result in scenario_results:

    scenario = result["scenario"]

    # CRITICAL:
    # Never process invalid scenarios.
    if scenario not in valid_scenarios:
        continue

    original_loading = result["original_loading"]
    optimized_loading = result["optimized_loading"]

    for line in common_lines:

        original_value = float(
            original_loading.get(line, np.nan)
        )

        optimized_value = float(
            optimized_loading.get(line, np.nan)
        )

        if not np.isfinite(original_value):
            continue

        if not np.isfinite(optimized_value):
            continue

        line_records.append({
            "scenario": scenario,
            "line": line,
            "original_loading_pct": original_value,
            "optimized_loading_pct": optimized_value,
            "loading_reduction_pct_points":
                original_value - optimized_value,
            "original_overloaded":
                original_value > LOADING_LIMIT,
            "optimized_overloaded":
                optimized_value > LOADING_LIMIT,
        })


line_df = pd.DataFrame(line_records)


if line_df.empty:

    raise RuntimeError(
        "No valid line-level data was produced. "
        "Do not continue until the power-flow calculation is fixed."
    )


# ================================================================
# LINE-LEVEL AGGREGATION
# ================================================================

summary_records = []


for line in common_lines:

    data = line_df[
        line_df["line"] == line
    ]

    if data.empty:
        continue

    original_average = (
        data["original_loading_pct"].mean()
    )

    optimized_average = (
        data["optimized_loading_pct"].mean()
    )

    original_maximum = (
        data["original_loading_pct"].max()
    )

    optimized_maximum = (
        data["optimized_loading_pct"].max()
    )

    original_overload_frequency = (
        data["original_overloaded"].mean() * 100
    )

    optimized_overload_frequency = (
        data["optimized_overloaded"].mean() * 100
    )

    average_reduction = (
        original_average -
        optimized_average
    )

    maximum_reduction = (
        original_maximum -
        optimized_maximum
    )

    summary_records.append({

        "line": line,

        "scenarios_analysed": len(data),

        "original_average_loading_pct":
            original_average,

        "optimized_average_loading_pct":
            optimized_average,

        "original_maximum_loading_pct":
            original_maximum,

        "optimized_maximum_loading_pct":
            optimized_maximum,

        "average_loading_reduction_pct_points":
            average_reduction,

        "maximum_loading_reduction_pct_points":
            maximum_reduction,

        "original_overload_frequency_pct":
            original_overload_frequency,

        "optimized_overload_frequency_pct":
            optimized_overload_frequency,

        "original_overload_count":
            int(data["original_overloaded"].sum()),

        "optimized_overload_count":
            int(data["optimized_overloaded"].sum()),
    })


summary_df = pd.DataFrame(summary_records)


# ================================================================
# TARGET STATUS
# ================================================================

summary_df["is_optimization_target"] = (
    summary_df["line"].isin(target_lines)
)


def determine_status(row):

    optimized_overload_frequency = (
        row["optimized_overload_frequency_pct"]
    )

    original_overload_frequency = (
        row["original_overload_frequency_pct"]
    )

    if optimized_overload_frequency == 0:

        return "RESOLVED"

    if (
        optimized_overload_frequency <
        original_overload_frequency
    ):

        return "IMPROVED_RESIDUAL_OVERLOAD"

    if (
        optimized_overload_frequency >
        original_overload_frequency
    ):

        return "WORSENED"

    if optimized_overload_frequency > 0:

        return "RESIDUAL_OVERLOAD"

    return "NO_OVERLOAD"


summary_df["status"] = summary_df.apply(
    determine_status,
    axis=1
)


# ================================================================
# SAVE COMPLETE LINE-LEVEL DATA
# ================================================================

summary_df = summary_df.sort_values(
    by=[
        "optimized_overload_frequency_pct",
        "optimized_maximum_loading_pct"
    ],
    ascending=False
)

summary_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# ================================================================
# RESIDUAL BOTTLENECKS
# ================================================================

residual_df = summary_df[
    summary_df["optimized_overload_frequency_pct"] > 0
].copy()


residual_df = residual_df.sort_values(
    by=[
        "optimized_overload_frequency_pct",
        "optimized_average_loading_pct",
        "optimized_maximum_loading_pct"
    ],
    ascending=False
)


residual_df.to_csv(
    OUTPUT_RESIDUAL,
    index=False
)


# ================================================================
# TARGET RESULTS
# ================================================================

print("\n")
print("=" * 70)
print("                 REINFORCED TARGET RESULTS")
print("=" * 70)


for line in target_lines:

    row = summary_df[
        summary_df["line"] == line
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(f"\nLine: {line}")

    print(
        f"  Original average loading : "
        f"{row['original_average_loading_pct']:.2f}%"
    )

    print(
        f"  Optimized average loading: "
        f"{row['optimized_average_loading_pct']:.2f}%"
    )

    print(
        f"  Original maximum loading : "
        f"{row['original_maximum_loading_pct']:.2f}%"
    )

    print(
        f"  Optimized maximum loading: "
        f"{row['optimized_maximum_loading_pct']:.2f}%"
    )

    print(
        f"  Average reduction        : "
        f"{row['average_loading_reduction_pct_points']:.2f} "
        f"percentage points"
    )

    print(
        f"  Original overload freq.  : "
        f"{row['original_overload_frequency_pct']:.2f}%"
    )

    print(
        f"  Optimized overload freq. : "
        f"{row['optimized_overload_frequency_pct']:.2f}%"
    )

    print(
        f"  Status                   : "
        f"{row['status']}"
    )


# ================================================================
# RESIDUAL BOTTLENECKS
# ================================================================

print("\n")
print("=" * 70)
print("             RESIDUAL TRANSMISSION BOTTLENECKS")
print("=" * 70)


if residual_df.empty:

    print("\nNo residual thermal overloads detected.")

else:

    for i, (_, row) in enumerate(
        residual_df.iterrows(),
        start=1
    ):

        target_marker = ""

        if row["is_optimization_target"]:
            target_marker = " [OPTIMIZATION TARGET]"

        print(
            f"\n{i}. "
            f"{row['line']}"
            f"{target_marker}"
        )

        print(
            f"   Optimized average loading : "
            f"{row['optimized_average_loading_pct']:.2f}%"
        )

        print(
            f"   Optimized maximum loading : "
            f"{row['optimized_maximum_loading_pct']:.2f}%"
        )

        print(
            f"   Optimized overload freq.  : "
            f"{row['optimized_overload_frequency_pct']:.2f}%"
        )

        print(
            f"   Original average loading  : "
            f"{row['original_average_loading_pct']:.2f}%"
        )

        print(
            f"   Average reduction         : "
            f"{row['average_loading_reduction_pct_points']:.2f} "
            f"percentage points"
        )


# ================================================================
# NEW BOTTLENECKS
# ================================================================

new_bottlenecks = summary_df[
    (
        summary_df["original_overload_frequency_pct"] == 0
    )
    &
    (
        summary_df["optimized_overload_frequency_pct"] > 0
    )
].copy()


print("\n")
print("=" * 70)
print("                  NEW BOTTLENECKS")
print("=" * 70)


if new_bottlenecks.empty:

    print("\nNo new line became overloaded after optimization.")

else:

    for _, row in new_bottlenecks.iterrows():

        print(
            f"\n{row['line']}"
        )

        print(
            f"  Optimized overload frequency: "
            f"{row['optimized_overload_frequency_pct']:.2f}%"
        )


# ================================================================
# RESOLVED BOTTLENECKS
# ================================================================

resolved_bottlenecks = summary_df[
    (
        summary_df["original_overload_frequency_pct"] > 0
    )
    &
    (
        summary_df["optimized_overload_frequency_pct"] == 0
    )
].copy()


print("\n")
print("=" * 70)
print("                 RESOLVED BOTTLENECKS")
print("=" * 70)


if resolved_bottlenecks.empty:

    print(
        "\nNo previously overloaded line was completely resolved."
    )

else:

    for _, row in resolved_bottlenecks.iterrows():

        print(
            f"\n{row['line']}"
        )

        print(
            f"  Original overload frequency : "
            f"{row['original_overload_frequency_pct']:.2f}%"
        )

        print(
            f"  Optimized overload frequency: "
            f"{row['optimized_overload_frequency_pct']:.2f}%"
        )


# ================================================================
# OVERALL LINE ANALYSIS
# ================================================================

print("\n")
print("=" * 70)
print("                  OVERALL LINE ANALYSIS")
print("=" * 70)

print(
    f"\nValid scenarios analysed : "
    f"{len(valid_scenarios)}"
)

print(
    f"Lines analysed           : "
    f"{len(summary_df)}"
)

print(
    f"Residual overloaded lines: "
    f"{len(residual_df)}"
)

print(
    f"Resolved lines           : "
    f"{len(resolved_bottlenecks)}"
)

print(
    f"New bottlenecks          : "
    f"{len(new_bottlenecks)}"
)


# ================================================================
# TARGET PERFORMANCE SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("              TARGET PERFORMANCE SUMMARY")
print("=" * 70)


for line in target_lines:

    row = summary_df[
        summary_df["line"] == line
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(f"\n{line}")

    print(
        f"  Status : {row['status']}"
    )

    print(
        f"  Loading : "
        f"{row['original_maximum_loading_pct']:.2f}% "
        f"-> "
        f"{row['optimized_maximum_loading_pct']:.2f}%"
    )

    print(
        f"  Overload frequency : "
        f"{row['original_overload_frequency_pct']:.2f}% "
        f"-> "
        f"{row['optimized_overload_frequency_pct']:.2f}%"
    )


# ================================================================
# FINAL SANITY CHECK
# ================================================================

print("\n")
print("=" * 70)
print("                    SANITY CHECK")
print("=" * 70)


max_result = summary_df[
    [
        "original_average_loading_pct",
        "optimized_average_loading_pct",
        "original_maximum_loading_pct",
        "optimized_maximum_loading_pct",
    ]
].max().max()


if max_result > MAX_REASONABLE_LOADING:

    print(
        "\nWARNING: Unrealistically large loading detected."
    )

    print(
        "The analysis should NOT be used for reinforcement "
        "decisions."
    )

else:

    print(
        "\nPASS: No numerically absurd line-loading values detected."
    )


# ================================================================
# FINAL
# ================================================================

print("\n")
print("=" * 70)
print("       DETAILED LINE-LEVEL ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved:")

print(
    OUTPUT_COMPARISON
)

print(
    OUTPUT_RESIDUAL
)

print(
    f"\nValid scenarios analysed: "
    f"{len(valid_scenarios)}"
)

print(
    f"Residual overloaded lines: "
    f"{len(residual_df)}"
)

print(
    f"New bottlenecks: "
    f"{len(new_bottlenecks)}"
)

print(
    f"Resolved lines: "
    f"{len(resolved_bottlenecks)}"
)

print("\nExcluded/non-converged scenarios:")

if invalid_scenarios:

    for scenario in invalid_scenarios:
        print(f"  {scenario}")

else:

    print("  None")


print("\nNEXT:")
print(
    "Use the corrected residual bottleneck results to determine "
    "whether a second reinforcement iteration is required."
)

print("=" * 70)