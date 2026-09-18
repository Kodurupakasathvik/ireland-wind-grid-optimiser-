"""
======================================================================
IRELAND GRID - SECOND REINFORCEMENT AC VALIDATION
======================================================================

Purpose
-------
Validate the second-reinforced network using an independent AC
power-flow analysis.

Comparison:
    Optimized network
        ->
    Second-reinforced network

Rules
-----
1. Every scenario is tested independently.
2. A scenario is VALID only if AC power flow converges.
3. Non-converged scenarios are EXCLUDED from loading calculations.
4. Non-finite loading values are rejected.
5. Line loading uses apparent power at both ends:
       max(|S0|, |S1|) / s_nom * 100
6. No reinforcement is performed by this script.
7. No line is declared resolved unless validated by AC power flow.
8. Results are saved to data/processed/.

======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

OPTIMIZED_PATH = PROCESSED / "eirgrid_optimized_network.nc"
ITERATION2_PATH = PROCESSED / "eirgrid_second_reinforced_network.nc"

RESULT_LINE_LEVEL = PROCESSED / "iteration2_ac_line_level_validation.csv"
RESULT_SCENARIOS = PROCESSED / "iteration2_ac_scenario_validation.csv"
RESULT_RESIDUAL = PROCESSED / "iteration2_residual_bottlenecks.csv"
RESULT_NEW = PROCESSED / "iteration2_new_bottlenecks.csv"
RESULT_RESOLVED = PROCESSED / "iteration2_resolved_bottlenecks.csv"

OVERLOAD_LIMIT_PCT = 100.0

SCENARIOS = [
    "S1_NORMAL",
    "S2_PEAK_DEMAND",
    "S3_HIGH_WIND",
    "S4_HIGH_WIND_HIGH_DEMAND",
    "S5_HIGH_AVAILABILITY_LOW_GENERATION",
    "S6_MAXIMUM_STRESS",
]


# =====================================================================
# PRINTING HELPERS
# =====================================================================

def heading(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def subheading(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# =====================================================================
# NETWORK LOADING
# =====================================================================

def load_network(path, label):
    heading(f"LOADING {label.upper()} NETWORK")

    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nNetwork file does not exist:\n{path}\n"
        )

    network = pypsa.Network(path)

    print("OK: Network loaded.")

    print()
    print("NETWORK")
    print(f"  Buses        : {len(network.buses)}")
    print(f"  Lines        : {len(network.lines)}")
    print(f"  Transformers : {len(network.transformers)}")
    print(f"  Generators   : {len(network.generators)}")
    print(f"  Loads        : {len(network.loads)}")

    return network


# =====================================================================
# NETWORK STRUCTURAL CHECK
# =====================================================================

def validate_structure(optimized, iteration2):

    heading("STRUCTURAL VALIDATION")

    checks = {
        "buses": (
            len(optimized.buses),
            len(iteration2.buses),
        ),
        "lines": (
            len(optimized.lines),
            len(iteration2.lines),
        ),
        "transformers": (
            len(optimized.transformers),
            len(iteration2.transformers),
        ),
        "generators": (
            len(optimized.generators),
            len(iteration2.generators),
        ),
        "loads": (
            len(optimized.loads),
            len(iteration2.loads),
        ),
    }

    all_ok = True

    for component, (a, b) in checks.items():
        if a != b:
            print(
                f"FAIL: {component}: "
                f"optimized={a}, iteration2={b}"
            )
            all_ok = False
        else:
            print(
                f"PASS: {component}: {a}"
            )

    common_lines = set(optimized.lines.index) & set(iteration2.lines.index)

    print()
    print(f"Common lines: {len(common_lines)}")

    if len(common_lines) != len(optimized.lines):
        missing = set(optimized.lines.index) - set(iteration2.lines.index)

        print("WARNING: Some optimized lines are missing in iteration-2.")
        print("Missing lines:")
        for line in sorted(missing):
            print(f"  {line}")

    return all_ok


# =====================================================================
# SCENARIO PREPARATION
# =====================================================================

def get_common_scenarios(optimized, iteration2):

    opt_snapshots = set(str(x) for x in optimized.snapshots)
    it2_snapshots = set(str(x) for x in iteration2.snapshots)

    requested = set(SCENARIOS)

    common = [
        s for s in SCENARIOS
        if s in opt_snapshots
        and s in it2_snapshots
        and s in requested
    ]

    return common


# =====================================================================
# POWER-FLOW RESULT EXTRACTION
# =====================================================================

def get_convergence_status(pf_result, snapshot):
    """
    Extract convergence status from the dictionary returned by PyPSA pf().

    Different PyPSA versions can represent this slightly differently,
    so this function is intentionally defensive.
    """

    if pf_result is None:
        return False

    converged = pf_result.get("converged", None)

    if converged is None:
        return False

    try:
        if isinstance(converged, pd.DataFrame):
            if snapshot in converged.index:
                values = converged.loc[snapshot].values
            elif snapshot in converged.columns:
                values = converged[snapshot].values
            else:
                values = converged.values.flatten()

        elif isinstance(converged, pd.Series):
            if snapshot in converged.index:
                values = np.asarray([converged.loc[snapshot]])
            else:
                values = converged.values

        else:
            values = np.asarray(converged)

        values = values.astype(bool)

        if values.size == 0:
            return False

        return bool(np.all(values))

    except Exception:
        return False


def calculate_line_loading(network, snapshot):
    """
    Calculate AC apparent-power loading for every line.

    Loading is based on the worst of the two line ends:

        S0 = sqrt(p0^2 + q0^2)
        S1 = sqrt(p1^2 + q1^2)

        loading = max(S0, S1) / s_nom * 100

    Returns
    -------
    pandas.Series
        Line loading percentage.
    """

    lines = network.lines.index

    p0 = network.lines_t.p0.loc[snapshot].reindex(lines)
    q0 = network.lines_t.q0.loc[snapshot].reindex(lines)

    p1 = network.lines_t.p1.loc[snapshot].reindex(lines)
    q1 = network.lines_t.q1.loc[snapshot].reindex(lines)

    s0 = np.sqrt(
        np.square(p0.astype(float))
        + np.square(q0.astype(float))
    )

    s1 = np.sqrt(
        np.square(p1.astype(float))
        + np.square(q1.astype(float))
    )

    apparent_power = pd.concat(
        [s0.rename("s0"), s1.rename("s1")],
        axis=1
    ).max(axis=1)

    s_nom = network.lines["s_nom"].reindex(lines).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        loading = apparent_power / s_nom * 100.0

    loading = loading.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return loading


# =====================================================================
# SINGLE SCENARIO AC VALIDATION
# =====================================================================

def run_single_scenario(network, snapshot, label):

    print()
    print(f"{label.upper()} NETWORK")
    print(f"  Scenario: {snapshot}")

    # Work only on the requested snapshot.
    snapshots = [snapshot]

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            pf_result = network.pf(
                snapshots=snapshots
            )

        convergence_ok = get_convergence_status(
            pf_result,
            snapshot
        )

        if not convergence_ok:
            print("  AC power flow : NOT CONVERGED")
            print("  Loading data  : REJECTED")

            return {
                "valid": False,
                "loading": None,
                "warning_count": len(caught_warnings),
            }

        loading = calculate_line_loading(
            network,
            snapshot
        )

        # -------------------------------------------------------------
        # FINITE-NUMBER VALIDATION
        # -------------------------------------------------------------

        if loading.empty:
            print("  AC power flow : INVALID")
            print("  Reason        : No line loading data")
            return {
                "valid": False,
                "loading": None,
                "warning_count": len(caught_warnings),
            }

        finite = np.isfinite(
            loading.to_numpy(dtype=float)
        )

        if not np.all(finite):
            bad_lines = loading.index[~finite]

            print("  AC power flow : INVALID")
            print(
                "  Reason        : Non-finite line loading detected"
            )
            print("  Rejected lines:")

            for line in bad_lines:
                print(f"    {line}")

            return {
                "valid": False,
                "loading": None,
                "warning_count": len(caught_warnings),
            }

        max_loading = float(loading.max())

        overloaded = int(
            (loading > OVERLOAD_LIMIT_PCT).sum()
        )

        print("  AC power flow : VALID")
        print(f"  Maximum loading : {max_loading:.2f}%")
        print(f"  Overloaded lines: {overloaded}")

        if caught_warnings:
            print(
                f"  PyPSA warnings : {len(caught_warnings)}"
            )

        return {
            "valid": True,
            "loading": loading,
            "warning_count": len(caught_warnings),
        }

    except Exception as exc:

        print("  AC power flow : ERROR")
        print(f"  Error         : {exc}")
        print("  Loading data  : REJECTED")

        return {
            "valid": False,
            "loading": None,
            "warning_count": 0,
        }


# =====================================================================
# SCENARIO VALIDATION
# =====================================================================

def validate_scenarios(optimized, iteration2, scenarios):

    heading("AC POWER-FLOW VALIDATION")

    optimized_results = {}
    iteration2_results = {}

    scenario_rows = []

    for snapshot in scenarios:

        print()
        print("=" * 70)
        print(f"SCENARIO: {snapshot}")
        print("=" * 70)

        opt_result = run_single_scenario(
            optimized,
            snapshot,
            "Optimized"
        )

        it2_result = run_single_scenario(
            iteration2,
            snapshot,
            "Iteration-2"
        )

        optimized_results[snapshot] = opt_result
        iteration2_results[snapshot] = it2_result

        scenario_rows.append({
            "scenario": snapshot,
            "optimized_valid": opt_result["valid"],
            "iteration2_valid": it2_result["valid"],
            "optimized_warning_count": opt_result["warning_count"],
            "iteration2_warning_count": it2_result["warning_count"],
            "comparison_valid": (
                opt_result["valid"]
                and it2_result["valid"]
            ),
        })

    scenario_df = pd.DataFrame(scenario_rows)

    return (
        optimized_results,
        iteration2_results,
        scenario_df,
    )


# =====================================================================
# BUILD LINE-LEVEL DATASET
# =====================================================================

def build_line_dataset(
    optimized,
    iteration2,
    optimized_results,
    iteration2_results,
    scenarios,
):

    heading("BUILDING VALIDATED LINE-LEVEL DATASET")

    common_lines = sorted(
        set(optimized.lines.index)
        & set(iteration2.lines.index)
    )

    rows = []

    for line in common_lines:

        opt_values = []
        it2_values = []

        for scenario in scenarios:

            opt_result = optimized_results[scenario]
            it2_result = iteration2_results[scenario]

            # Only use a scenario when BOTH networks converged.
            if (
                opt_result["valid"]
                and it2_result["valid"]
            ):
                opt_values.append(
                    float(opt_result["loading"].loc[line])
                )

                it2_values.append(
                    float(it2_result["loading"].loc[line])
                )

        if not opt_values:
            continue

        opt_values = np.asarray(opt_values)
        it2_values = np.asarray(it2_values)

        original_max = float(np.max(opt_values))
        iteration2_max = float(np.max(it2_values))

        original_average = float(np.mean(opt_values))
        iteration2_average = float(np.mean(it2_values))

        original_overload_count = int(
            np.sum(opt_values > OVERLOAD_LIMIT_PCT)
        )

        iteration2_overload_count = int(
            np.sum(it2_values > OVERLOAD_LIMIT_PCT)
        )

        n_valid = len(opt_values)

        original_frequency = (
            original_overload_count
            / n_valid
            * 100.0
        )

        iteration2_frequency = (
            iteration2_overload_count
            / n_valid
            * 100.0
        )

        rows.append({
            "line": line,
            "scenarios_analysed": n_valid,

            "optimized_average_loading_pct":
                original_average,

            "iteration2_average_loading_pct":
                iteration2_average,

            "optimized_maximum_loading_pct":
                original_max,

            "iteration2_maximum_loading_pct":
                iteration2_max,

            "average_loading_change_pct_points":
                iteration2_average - original_average,

            "maximum_loading_change_pct_points":
                iteration2_max - original_max,

            "optimized_overload_frequency_pct":
                original_frequency,

            "iteration2_overload_frequency_pct":
                iteration2_frequency,

            "optimized_overload_count":
                original_overload_count,

            "iteration2_overload_count":
                iteration2_overload_count,

            "iteration2_status":
                (
                    "RESIDUAL_OVERLOAD"
                    if iteration2_max > OVERLOAD_LIMIT_PCT
                    else "WITHIN_LIMIT"
                ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# RESIDUAL BOTTLENECKS
# =====================================================================

def identify_residual_bottlenecks(line_df):

    if line_df.empty:
        return line_df.copy()

    residual = line_df[
        line_df["iteration2_maximum_loading_pct"]
        > OVERLOAD_LIMIT_PCT
    ].copy()

    residual = residual.sort_values(
        "iteration2_maximum_loading_pct",
        ascending=False
    )

    return residual


# =====================================================================
# NEW BOTTLENECKS
# =====================================================================

def identify_new_bottlenecks(line_df):

    if line_df.empty:
        return line_df.copy()

    new_bottlenecks = line_df[
        (
            line_df[
                "optimized_maximum_loading_pct"
            ] <= OVERLOAD_LIMIT_PCT
        )
        &
        (
            line_df[
                "iteration2_maximum_loading_pct"
            ] > OVERLOAD_LIMIT_PCT
        )
    ].copy()

    new_bottlenecks = new_bottlenecks.sort_values(
        "iteration2_maximum_loading_pct",
        ascending=False
    )

    return new_bottlenecks


# =====================================================================
# RESOLVED BOTTLENECKS
# =====================================================================

def identify_resolved_bottlenecks(line_df):

    if line_df.empty:
        return line_df.copy()

    resolved = line_df[
        (
            line_df[
                "optimized_maximum_loading_pct"
            ] > OVERLOAD_LIMIT_PCT
        )
        &
        (
            line_df[
                "iteration2_maximum_loading_pct"
            ] <= OVERLOAD_LIMIT_PCT
        )
    ].copy()

    resolved = resolved.sort_values(
        "optimized_maximum_loading_pct",
        ascending=False
    )

    return resolved


# =====================================================================
# TARGET RESULTS
# =====================================================================

def print_target_results(line_df):

    target_lines = [
        "merged_way/257889771-220+1",
        "merged_way/1231251986-220+2",
        "way/343436171-220",
    ]

    heading("TARGET LINE VALIDATION RESULTS")

    for line in target_lines:

        row = line_df[
            line_df["line"] == line
        ]

        if row.empty:
            print()
            print(f"{line}")
            print("  NOT AVAILABLE IN VALIDATED DATASET")
            continue

        row = row.iloc[0]

        opt_max = row[
            "optimized_maximum_loading_pct"
        ]

        it2_max = row[
            "iteration2_maximum_loading_pct"
        ]

        opt_avg = row[
            "optimized_average_loading_pct"
        ]

        it2_avg = row[
            "iteration2_average_loading_pct"
        ]

        print()
        print(f"{line}")

        print(
            f"  Optimized maximum loading : "
            f"{opt_max:.2f}%"
        )

        print(
            f"  Iteration-2 maximum       : "
            f"{it2_max:.2f}%"
        )

        print(
            f"  Optimized average loading : "
            f"{opt_avg:.2f}%"
        )

        print(
            f"  Iteration-2 average       : "
            f"{it2_avg:.2f}%"
        )

        print(
            f"  Maximum change             : "
            f"{it2_max - opt_max:+.2f} percentage points"
        )

        print(
            f"  Status                     : "
            f"{row['iteration2_status']}"
        )


# =====================================================================
# PRINT RESIDUAL BOTTLENECKS
# =====================================================================

def print_residual_bottlenecks(residual):

    heading("RESIDUAL TRANSMISSION BOTTLENECKS")

    if residual.empty:
        print("NONE")
        return

    for i, (_, row) in enumerate(
        residual.iterrows(),
        start=1
    ):

        print()
        print(
            f"{i}. {row['line']}"
        )

        print(
            f"   Iteration-2 maximum loading : "
            f"{row['iteration2_maximum_loading_pct']:.2f}%"
        )

        print(
            f"   Iteration-2 average loading : "
            f"{row['iteration2_average_loading_pct']:.2f}%"
        )

        print(
            f"   Overload frequency          : "
            f"{row['iteration2_overload_frequency_pct']:.2f}%"
        )

        print(
            f"   Previous maximum loading   : "
            f"{row['optimized_maximum_loading_pct']:.2f}%"
        )


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(
    line_df,
    scenario_df,
    residual,
    new_bottlenecks,
    resolved,
):

    heading("SAVING VALIDATION RESULTS")

    line_df.to_csv(
        RESULT_LINE_LEVEL,
        index=False
    )

    scenario_df.to_csv(
        RESULT_SCENARIOS,
        index=False
    )

    residual.to_csv(
        RESULT_RESIDUAL,
        index=False
    )

    new_bottlenecks.to_csv(
        RESULT_NEW,
        index=False
    )

    resolved.to_csv(
        RESULT_RESOLVED,
        index=False
    )

    print(
        "OK: Line-level validation saved."
    )
    print(RESULT_LINE_LEVEL)

    print()
    print(
        "OK: Scenario validation saved."
    )
    print(RESULT_SCENARIOS)

    print()
    print(
        "OK: Residual bottlenecks saved."
    )
    print(RESULT_RESIDUAL)

    print()
    print(
        "OK: New bottlenecks saved."
    )
    print(RESULT_NEW)

    print()
    print(
        "OK: Resolved bottlenecks saved."
    )
    print(RESULT_RESOLVED)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

def print_final_summary(
    scenario_df,
    line_df,
    residual,
    new_bottlenecks,
    resolved,
):

    heading("FINAL VALIDATION SUMMARY")

    total = len(scenario_df)

    optimized_valid = int(
        scenario_df["optimized_valid"].sum()
    )

    iteration2_valid = int(
        scenario_df["iteration2_valid"].sum()
    )

    comparison_valid = int(
        scenario_df["comparison_valid"].sum()
    )

    print(
        f"Total scenarios requested       : {total}"
    )

    print(
        f"Optimized scenarios converged   : "
        f"{optimized_valid}"
    )

    print(
        f"Iteration-2 scenarios converged : "
        f"{iteration2_valid}"
    )

    print(
        f"Valid comparison scenarios      : "
        f"{comparison_valid}"
    )

    print()
    print(
        f"Lines analysed                  : "
        f"{len(line_df)}"
    )

    print(
        f"Residual overloaded lines       : "
        f"{len(residual)}"
    )

    print(
        f"New bottlenecks                 : "
        f"{len(new_bottlenecks)}"
    )

    print(
        f"Resolved bottlenecks            : "
        f"{len(resolved)}"
    )

    print()

    if comparison_valid < total:

        print(
            "WARNING: Not every scenario has a valid "
            "optimized + iteration-2 comparison."
        )

        print(
            "Non-converged scenarios were excluded "
            "from line-loading calculations."
        )

    else:

        print(
            "PASS: All requested scenarios have "
            "valid AC comparison results."
        )

    if len(new_bottlenecks) == 0:

        print(
            "PASS: No new bottlenecks were created."
        )

    else:

        print(
            "WARNING: New bottlenecks were detected."
        )

    if len(resolved) > 0:

        print(
            "PASS: At least one previous bottleneck "
            "was fully resolved."
        )

    else:

        print(
            "INFO: No previously overloaded line "
            "was completely resolved."
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    heading(
        "IRELAND GRID - SECOND REINFORCEMENT "
        "AC VALIDATION"
    )

    print()
    print(
        "VALIDATION MODE"
    )

    print(
        "This script performs an independent AC "
        "power-flow validation."
    )

    print()
    print(
        "IMPORTANT RULES:"
    )

    print(
        "  1. Non-converged scenarios are rejected."
    )

    print(
        "  2. Non-finite loading values are rejected."
    )

    print(
        "  3. No reinforcement is performed."
    )

    print(
        "  4. No bottleneck is declared resolved "
        "without valid AC results."
    )

    # ---------------------------------------------------------------
    # LOAD NETWORKS
    # ---------------------------------------------------------------

    optimized = load_network(
        OPTIMIZED_PATH,
        "optimized"
    )

    iteration2 = load_network(
        ITERATION2_PATH,
        "iteration-2"
    )

    # ---------------------------------------------------------------
    # STRUCTURE
    # ---------------------------------------------------------------

    validate_structure(
        optimized,
        iteration2
    )

    # ---------------------------------------------------------------
    # SNAPSHOTS
    # ---------------------------------------------------------------

    heading("SCENARIO CHECK")

    print(
        f"Optimized snapshots : "
        f"{len(optimized.snapshots)}"
    )

    print(
        f"Iteration-2 snapshots : "
        f"{len(iteration2.snapshots)}"
    )

    scenarios = get_common_scenarios(
        optimized,
        iteration2
    )

    print()
    print("Common requested scenarios:")

    for scenario in scenarios:
        print(f"  {scenario}")

    missing = [
        s for s in SCENARIOS
        if s not in scenarios
    ]

    if missing:

        print()
        print("WARNING: Missing scenarios:")

        for scenario in missing:
            print(f"  {scenario}")

    if not scenarios:
        raise RuntimeError(
            "No common scenarios are available "
            "for validation."
        )

    # ---------------------------------------------------------------
    # AC POWER FLOW
    # ---------------------------------------------------------------

    (
        optimized_results,
        iteration2_results,
        scenario_df,
    ) = validate_scenarios(
        optimized,
        iteration2,
        scenarios
    )

    # ---------------------------------------------------------------
    # BUILD VALIDATED DATASET
    # ---------------------------------------------------------------

    line_df = build_line_dataset(
        optimized,
        iteration2,
        optimized_results,
        iteration2_results,
        scenarios,
    )

    # ---------------------------------------------------------------
    # IDENTIFY RESULTS
    # ---------------------------------------------------------------

    residual = identify_residual_bottlenecks(
        line_df
    )

    new_bottlenecks = identify_new_bottlenecks(
        line_df
    )

    resolved = identify_resolved_bottlenecks(
        line_df
    )

    # ---------------------------------------------------------------
    # PRINT TARGETS
    # ---------------------------------------------------------------

    print_target_results(
        line_df
    )

    # ---------------------------------------------------------------
    # PRINT RESIDUALS
    # ---------------------------------------------------------------

    print_residual_bottlenecks(
        residual
    )

    # ---------------------------------------------------------------
    # NEW BOTTLENECKS
    # ---------------------------------------------------------------

    heading("NEW BOTTLENECKS")

    if new_bottlenecks.empty:
        print(
            "No new line became overloaded "
            "after second reinforcement."
        )
    else:

        for i, (_, row) in enumerate(
            new_bottlenecks.iterrows(),
            start=1
        ):

            print(
                f"{i}. {row['line']}"
            )

            print(
                f"   Previous maximum : "
                f"{row['optimized_maximum_loading_pct']:.2f}%"
            )

            print(
                f"   Iteration-2 max  : "
                f"{row['iteration2_maximum_loading_pct']:.2f}%"
            )

    # ---------------------------------------------------------------
    # RESOLVED
    # ---------------------------------------------------------------

    heading("RESOLVED BOTTLENECKS")

    if resolved.empty:
        print(
            "No previously overloaded line was "
            "completely resolved."
        )
    else:

        for i, (_, row) in enumerate(
            resolved.iterrows(),
            start=1
        ):

            print(
                f"{i}. {row['line']}"
            )

            print(
                f"   Previous maximum : "
                f"{row['optimized_maximum_loading_pct']:.2f}%"
            )

            print(
                f"   Iteration-2 max  : "
                f"{row['iteration2_maximum_loading_pct']:.2f}%"
            )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    save_results(
        line_df,
        scenario_df,
        residual,
        new_bottlenecks,
        resolved,
    )

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------

    print_final_summary(
        scenario_df,
        line_df,
        residual,
        new_bottlenecks,
        resolved,
    )

    heading(
        "SECOND REINFORCEMENT AC VALIDATION COMPLETE"
    )

    print()
    print(
        "The iteration-2 network has now been "
        "independently tested using AC power flow."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Only converged scenarios were used "
        "for line-loading conclusions."
    )

    print()
    print(
        "Next decision depends on the validation "
        "results above:"
    )

    print(
        "  - All scenarios valid + no residuals"
        " -> iteration-2 can be accepted."
    )

    print(
        "  - Valid scenarios + residuals"
        " -> evaluate whether iteration-3 is justified."
    )

    print(
        "  - Non-converged scenarios"
        " -> fix/diagnose AC model before further reinforcement."
    )

    print()
    print(
        "======================================================================"
    )


if __name__ == "__main__":
    main()