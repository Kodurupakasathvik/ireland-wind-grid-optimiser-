import os
import copy
import numpy as np
import pandas as pd
import pypsa


# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_PATH = r"data\processed\eirgrid_second_reinforced_network.nc"

OUTPUT_PATH = (
    r"data\processed\s2_candidate_b_cross_scenario_validation.csv"
)

SNAPSHOTS = [
    "S1_NORMAL",
    "S2_PEAK_DEMAND",
    "S3_HIGH_WIND",
    "S4_HIGH_WIND_HIGH_DEMAND",
    "S5_HIGH_AVAILABILITY_LOW_GENERATION",
    "S6_MAXIMUM_STRESS",
]

REINFORCEMENT_MULTIPLIER = 1.50

# Candidate B from the previous audit
CANDIDATE_B_LINES = [
    "merged_way/61295764-220+1",
    "way/343436171-220",
    "merged_way/257889771-220+1",
]


# ============================================================
# HELPERS
# ============================================================

def run_ac_pf(n, snapshot):
    """
    Run controlled AC nonlinear power flow for one snapshot.
    """

    result = n.pf(
        snapshots=[snapshot],
        distribute_slack=True,
    )

    # PyPSA returns a Dict-like structure containing:
    # n_iter, error, converged

    converged_table = result["converged"]
    error_table = result["error"]
    iteration_table = result["n_iter"]

    converged_values = converged_table.loc[snapshot]
    error_values = error_table.loc[snapshot]
    iteration_values = iteration_table.loc[snapshot]

    all_converged = bool(np.all(np.asarray(converged_values, dtype=bool)))
    max_error = float(np.max(np.asarray(error_values, dtype=float)))
    max_iterations = int(np.max(np.asarray(iteration_values, dtype=int)))

    return {
        "converged": all_converged,
        "max_error": max_error,
        "max_iterations": max_iterations,
    }


def calculate_metrics(n, snapshot):
    """
    Extract voltage and AC line-loading metrics.
    """

    # --------------------------------------------------------
    # BUS VOLTAGES
    # --------------------------------------------------------

    v = n.buses_t.v_mag_pu.loc[snapshot]

    min_voltage = float(v.min())
    max_voltage = float(v.max())

    min_voltage_bus = str(v.idxmin())
    max_voltage_bus = str(v.idxmax())

    # --------------------------------------------------------
    # LINE FLOWS
    # --------------------------------------------------------

    s_nom = n.lines.s_nom

    p0 = n.lines_t.p0.loc[snapshot].abs()
    p1 = n.lines_t.p1.loc[snapshot].abs()

    loading_p0 = 100.0 * p0 / s_nom
    loading_p1 = 100.0 * p1 / s_nom

    max_loading = pd.concat(
        [loading_p0, loading_p1],
        axis=1
    ).max(axis=1)

    max_line_loading = float(max_loading.max())
    worst_line = str(max_loading.idxmax())

    lines_over_100 = int((max_loading > 100.0).sum())
    lines_over_110 = int((max_loading > 110.0).sum())
    lines_over_120 = int((max_loading > 120.0).sum())

    return {
        "min_voltage_pu": min_voltage,
        "max_voltage_pu": max_voltage,
        "min_voltage_bus": min_voltage_bus,
        "max_voltage_bus": max_voltage_bus,
        "max_line_loading_pct": max_line_loading,
        "worst_line": worst_line,
        "lines_over_100": lines_over_100,
        "lines_over_110": lines_over_110,
        "lines_over_120": lines_over_120,
    }


def reinforce_temporarily(n, line_names, multiplier):
    """
    Temporarily increase s_nom in memory only.
    """

    for line_name in line_names:

        if line_name not in n.lines.index:
            raise KeyError(
                f"Candidate B line not found in network: {line_name}"
            )

        original = float(n.lines.at[line_name, "s_nom"])

        n.lines.at[line_name, "s_nom"] = (
            original * multiplier
        )


def evaluate_network(network_path, snapshot, candidate_name):
    """
    Load a fresh network and evaluate one snapshot.
    """

    n = pypsa.Network(network_path)

    if snapshot not in n.snapshots:
        raise ValueError(
            f"Snapshot '{snapshot}' not found in network."
        )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    pf_base = run_ac_pf(n, snapshot)
    metrics_base = calculate_metrics(n, snapshot)

    base_row = {
        "candidate": candidate_name,
        "case": "BASELINE",
        "snapshot": snapshot,
        **pf_base,
        **metrics_base,
    }

    # --------------------------------------------------------
    # CANDIDATE B
    # --------------------------------------------------------

    n_candidate = pypsa.Network(network_path)

    reinforce_temporarily(
        n_candidate,
        CANDIDATE_B_LINES,
        REINFORCEMENT_MULTIPLIER,
    )

    pf_candidate = run_ac_pf(
        n_candidate,
        snapshot,
    )

    metrics_candidate = calculate_metrics(
        n_candidate,
        snapshot,
    )

    candidate_row = {
        "candidate": candidate_name,
        "case": "CANDIDATE_B",
        "snapshot": snapshot,
        **pf_candidate,
        **metrics_candidate,
    }

    return base_row, candidate_row


# ============================================================
# HEADER
# ============================================================

print("=" * 118)
print("IRELAND GRID - CANDIDATE B CROSS-SCENARIO VALIDATION")
print("=" * 118)

print()
print("Network       :", NETWORK_PATH)
print("PF            : AC nonlinear")
print("Slack         : distributed")
print("Dispatch      : unchanged")
print("Loads         : unchanged")
print("Source        : READ-ONLY")
print()
print("Candidate B reinforcement multiplier :",
      f"{REINFORCEMENT_MULTIPLIER:.2f}x")

print()
print("Candidate B lines:")

for line in CANDIDATE_B_LINES:
    print("  -", line)

print("=" * 118)


# ============================================================
# RUN ALL SCENARIOS
# ============================================================

rows = []

for snapshot in SNAPSHOTS:

    print()
    print("-" * 118)
    print("SNAPSHOT:", snapshot)
    print("-" * 118)

    base_row, candidate_row = evaluate_network(
        NETWORK_PATH,
        snapshot,
        "B_THERMAL_CORRIDOR",
    )

    rows.append(base_row)
    rows.append(candidate_row)

    print()
    print("BASELINE")
    print("  PF converged      :", base_row["converged"])
    print("  PF maximum error  :", base_row["max_error"])
    print("  Minimum voltage    :", base_row["min_voltage_pu"])
    print("  Maximum loading    :", base_row["max_line_loading_pct"])
    print("  Worst line         :", base_row["worst_line"])
    print("  Lines >100%        :", base_row["lines_over_100"])

    print()
    print("CANDIDATE B")
    print("  PF converged      :", candidate_row["converged"])
    print("  PF maximum error  :", candidate_row["max_error"])
    print("  Minimum voltage    :", candidate_row["min_voltage_pu"])
    print("  Maximum loading    :", candidate_row["max_line_loading_pct"])
    print("  Worst line         :", candidate_row["worst_line"])
    print("  Lines >100%        :", candidate_row["lines_over_100"])

    print()
    print("CHANGE")

    voltage_change = (
        candidate_row["min_voltage_pu"]
        - base_row["min_voltage_pu"]
    )

    loading_change = (
        candidate_row["max_line_loading_pct"]
        - base_row["max_line_loading_pct"]
    )

    overload_change = (
        candidate_row["lines_over_100"]
        - base_row["lines_over_100"]
    )

    print(
        "  Minimum voltage change : "
        f"{voltage_change:+.6f} pu"
    )

    print(
        "  Maximum loading change : "
        f"{loading_change:+.6f} percentage points"
    )

    print(
        "  Lines >100% change     : "
        f"{overload_change:+d}"
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# ADD CHANGE COLUMNS
# ============================================================

change_rows = []

for snapshot in SNAPSHOTS:

    base = df[
        (df["snapshot"] == snapshot)
        & (df["case"] == "BASELINE")
    ].iloc[0]

    candidate = df[
        (df["snapshot"] == snapshot)
        & (df["case"] == "CANDIDATE_B")
    ].iloc[0]

    change_rows.append({
        "snapshot": snapshot,

        "baseline_min_voltage_pu":
            base["min_voltage_pu"],

        "candidate_b_min_voltage_pu":
            candidate["min_voltage_pu"],

        "voltage_change_pu":
            candidate["min_voltage_pu"]
            - base["min_voltage_pu"],

        "baseline_max_loading_pct":
            base["max_line_loading_pct"],

        "candidate_b_max_loading_pct":
            candidate["max_line_loading_pct"],

        "loading_change_pct_points":
            candidate["max_line_loading_pct"]
            - base["max_line_loading_pct"],

        "baseline_lines_over_100":
            base["lines_over_100"],

        "candidate_b_lines_over_100":
            candidate["lines_over_100"],

        "over_100_change":
            candidate["lines_over_100"]
            - base["lines_over_100"],

        "baseline_lines_over_110":
            base["lines_over_110"],

        "candidate_b_lines_over_110":
            candidate["lines_over_110"],

        "over_110_change":
            candidate["lines_over_110"]
            - base["lines_over_110"],

        "baseline_lines_over_120":
            base["lines_over_120"],

        "candidate_b_lines_over_120":
            candidate["lines_over_120"],

        "over_120_change":
            candidate["lines_over_120"]
            - base["lines_over_120"],

        "baseline_worst_line":
            base["worst_line"],

        "candidate_b_worst_line":
            candidate["worst_line"],

        "baseline_min_voltage_bus":
            base["min_voltage_bus"],

        "candidate_b_min_voltage_bus":
            candidate["min_voltage_bus"],
    })


comparison = pd.DataFrame(change_rows)


# ============================================================
# PRINT COMPARISON
# ============================================================

print()
print("=" * 118)
print("CANDIDATE B CROSS-SCENARIO COMPARISON")
print("=" * 118)

print(
    comparison[
        [
            "snapshot",
            "baseline_min_voltage_pu",
            "candidate_b_min_voltage_pu",
            "voltage_change_pu",
            "baseline_max_loading_pct",
            "candidate_b_max_loading_pct",
            "loading_change_pct_points",
            "baseline_lines_over_100",
            "candidate_b_lines_over_100",
        ]
    ].to_string(index=False)
)


# ============================================================
# GLOBAL VALIDATION
# ============================================================

all_converged = bool(df["converged"].all())

worst_baseline_loading = float(
    comparison["baseline_max_loading_pct"].max()
)

worst_candidate_loading = float(
    comparison["candidate_b_max_loading_pct"].max()
)

worst_baseline_voltage = float(
    comparison["baseline_min_voltage_pu"].min()
)

worst_candidate_voltage = float(
    comparison["candidate_b_min_voltage_pu"].min()
)

total_loading_reduction = (
    worst_baseline_loading
    - worst_candidate_loading
)

total_voltage_change = (
    worst_candidate_voltage
    - worst_baseline_voltage
)


# ============================================================
# FIND WORST CASES
# ============================================================

worst_baseline_row = comparison.loc[
    comparison["baseline_max_loading_pct"].idxmax()
]

worst_candidate_row = comparison.loc[
    comparison["candidate_b_max_loading_pct"].idxmax()
]


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 118)
print("CANDIDATE B VALIDATION SUMMARY")
print("=" * 118)

print()
print("All PF cases converged       :", all_converged)

print()
print("Worst baseline loading      :",
      f"{worst_baseline_loading:.6f}%")

print("Worst Candidate B loading   :",
      f"{worst_candidate_loading:.6f}%")

print("Global loading improvement  :",
      f"{total_loading_reduction:+.6f} percentage points")

print()
print("Worst baseline voltage      :",
      f"{worst_baseline_voltage:.6f} pu")

print("Worst Candidate B voltage   :",
      f"{worst_candidate_voltage:.6f} pu")

print("Global voltage change       :",
      f"{total_voltage_change:+.6f} pu")

print()
print("Worst baseline-loading case :",
      worst_baseline_row["snapshot"])

print("Worst Candidate-B case      :",
      worst_candidate_row["snapshot"])


# ============================================================
# DECISION LOGIC
# ============================================================

print()
print("=" * 118)
print("ENGINEERING DECISION")
print("=" * 118)

if not all_converged:

    print()
    print("REJECT CANDIDATE B")
    print("Reason: At least one AC power-flow case failed to converge.")

elif worst_candidate_loading >= 100.0:

    print()
    print("CANDIDATE B DOES NOT FULLY RESOLVE THERMAL CONSTRAINTS")
    print(
        "Worst Candidate B loading remains "
        f"{worst_candidate_loading:.6f}%."
    )

    if worst_candidate_voltage < 0.95:

        print(
            "Voltage constraint also remains severe: "
            f"{worst_candidate_voltage:.6f} pu."
        )

    print()
    print(
        "Candidate B may still be useful as an incremental "
        "reinforcement, but it is NOT a final feasible solution."
    )

elif worst_candidate_voltage < 0.95:

    print()
    print("THERMAL CONSTRAINT RESOLVED, VOLTAGE CONSTRAINT REMAINS")

else:

    print()
    print("CANDIDATE B PASSES THE SCREENING THRESHOLDS")


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

comparison.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("=" * 118)
print("AUDIT FILE SAVED")
print("=" * 118)

print()
print(OUTPUT_PATH)

print()
print("NO SOURCE NETWORK MODIFICATION PERFORMED.")
print("NO DISPATCH MODIFICATION PERFORMED.")
print("NO LOAD MODIFICATION PERFORMED.")
print("NO OPTIMIZATION PERFORMED.")
print("ALL REINFORCEMENTS WERE TEMPORARY IN-MEMORY TESTS.")

print()
print("=" * 118)
print("CANDIDATE B CROSS-SCENARIO VALIDATION COMPLETE")
print("=" * 118)