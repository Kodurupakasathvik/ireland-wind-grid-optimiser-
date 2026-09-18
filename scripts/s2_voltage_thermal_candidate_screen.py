from pathlib import Path
import copy
import pandas as pd
import pypsa


# ============================================================
# IRELAND GRID
# S2 VOLTAGE + THERMAL CANDIDATE SCREEN
# ============================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

REINFORCEMENT_MULTIPLIER = 1.50

OUTPUT_PATH = Path(
    "data/processed/s2_voltage_thermal_candidate_screen.csv"
)


# ------------------------------------------------------------
# Candidate definitions
# ------------------------------------------------------------

CANDIDATES = {

    # Candidate B from previous validation
    "B_THERMAL_CORRIDOR": [
        "merged_way/61295764-220+1",
        "way/343436171-220",
        "merged_way/257889771-220+1",
    ],

    # Lines directly incident to minimum-voltage bus
    "D_MIN_VOLTAGE_SIDE": [
        "merged_way/148948901-220+1",
        "relation/5567982-220",
        "merged_way/104388602-220+1",
    ],

    # Combined B + minimum-voltage-side reinforcement
    "E_COMBINED_B_VOLTAGE": [
        "merged_way/61295764-220+1",
        "way/343436171-220",
        "merged_way/257889771-220+1",
        "merged_way/148948901-220+1",
        "relation/5567982-220",
        "merged_way/104388602-220+1",
    ],

    # Add the two strongest lines incident to the minimum-voltage bus
    # to the principal thermal corridor.
    "F_COMBINED_PRIORITY": [
        "merged_way/61295764-220+1",
        "way/343436171-220",
        "merged_way/257889771-220+1",
        "merged_way/148948901-220+1",
        "relation/5567982-220",
    ],
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def get_pf_result(network):
    """
    PyPSA versions return a dictionary-like object from pf().
    Do not use network.pf_converged because that attribute does
    not exist in the installed PyPSA version.
    """

    result = network.pf(
        snapshots=[SNAPSHOT]
    )

    return result


def extract_pf_metrics(network, pf_result):
    """
    Extract convergence/error information safely across PyPSA
    versions.
    """

    try:
        converged = pf_result["converged"]

        if isinstance(converged, pd.DataFrame):
            converged_values = converged.loc[SNAPSHOT].astype(bool)
            pf_converged = bool(converged_values.all())
        else:
            pf_converged = bool(converged)

    except Exception:
        pf_converged = False

    try:
        error = pf_result["error"]

        if isinstance(error, pd.DataFrame):
            pf_error = float(error.loc[SNAPSHOT].max())
        else:
            pf_error = float(error)

    except Exception:
        pf_error = float("nan")

    try:
        n_iter = pf_result["n_iter"]

        if isinstance(n_iter, pd.DataFrame):
            pf_iterations = int(n_iter.loc[SNAPSHOT].max())
        else:
            pf_iterations = int(n_iter)

    except Exception:
        pf_iterations = -1

    return pf_converged, pf_error, pf_iterations


def calculate_metrics(network, pf_result):
    """
    Calculate the main engineering metrics.
    """

    pf_converged, pf_error, pf_iterations = extract_pf_metrics(
        network,
        pf_result
    )

    # --------------------------------------------------------
    # Voltage
    # --------------------------------------------------------

    voltage = network.buses_t.v_mag_pu.loc[SNAPSHOT]

    min_voltage = float(voltage.min())
    min_voltage_bus = str(voltage.idxmin())

    # --------------------------------------------------------
    # Line loading
    # --------------------------------------------------------

    lines = network.lines

    s_nom = lines["s_nom"]

    p0 = network.lines_t.p0.loc[SNAPSHOT].abs()
    p1 = network.lines_t.p1.loc[SNAPSHOT].abs()

    loading_p0 = (p0 / s_nom) * 100.0
    loading_p1 = (p1 / s_nom) * 100.0

    max_loading_by_line = pd.concat(
        [loading_p0, loading_p1],
        axis=1
    ).max(axis=1)

    max_loading = float(max_loading_by_line.max())

    worst_line = str(
        max_loading_by_line.idxmax()
    )

    lines_over_100 = int(
        (max_loading_by_line > 100.0).sum()
    )

    lines_over_110 = int(
        (max_loading_by_line > 110.0).sum()
    )

    lines_over_120 = int(
        (max_loading_by_line > 120.0).sum()
    )

    return {
        "pf_converged": pf_converged,
        "pf_max_error": pf_error,
        "pf_max_iterations": pf_iterations,
        "min_voltage_pu": min_voltage,
        "min_voltage_bus": min_voltage_bus,
        "max_line_loading_pct": max_loading,
        "worst_line": worst_line,
        "lines_over_100": lines_over_100,
        "lines_over_110": lines_over_110,
        "lines_over_120": lines_over_120,
    }


def reinforce_lines(network, line_names):
    """
    Temporary in-memory reinforcement only.

    Source network is never modified or saved.
    """

    missing = []

    for line_name in line_names:

        if line_name not in network.lines.index:
            missing.append(line_name)
            continue

        network.lines.at[
            line_name,
            "s_nom"
        ] *= REINFORCEMENT_MULTIPLIER

    return missing


def run_case(label, line_names):

    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    print()
    print("Reinforced lines:")

    for line in line_names:
        print(f"  - {line}")

    network = pypsa.Network(
        NETWORK_PATH
    )

    missing = reinforce_lines(
        network,
        line_names
    )

    if missing:
        print()
        print("WARNING - missing lines:")

        for line in missing:
            print(f"  - {line}")

    pf_result = get_pf_result(
        network
    )

    metrics = calculate_metrics(
        network,
        pf_result
    )

    print()
    print("PF converged       :", metrics["pf_converged"])
    print("PF maximum error   :", metrics["pf_max_error"])
    print("PF maximum iters   :", metrics["pf_max_iterations"])

    print(
        "Minimum voltage    :",
        metrics["min_voltage_pu"],
        "pu"
    )

    print(
        "Minimum-V bus      :",
        metrics["min_voltage_bus"]
    )

    print(
        "Maximum loading    :",
        metrics["max_line_loading_pct"],
        "%"
    )

    print(
        "Worst line         :",
        metrics["worst_line"]
    )

    print(
        "Lines >100%        :",
        metrics["lines_over_100"]
    )

    print(
        "Lines >110%        :",
        metrics["lines_over_110"]
    )

    print(
        "Lines >120%        :",
        metrics["lines_over_120"]
    )

    return metrics


# ============================================================
# START
# ============================================================

print("=" * 110)
print("IRELAND GRID - S2 VOLTAGE + THERMAL CANDIDATE SCREEN")
print("=" * 110)

print()
print("Network  :", NETWORK_PATH)
print("Snapshot :", SNAPSHOT)
print("PF       : AC nonlinear")
print("Slack    : distributed")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Source   : READ-ONLY")
print()
print(
    "Reinforcement multiplier :",
    f"{REINFORCEMENT_MULTIPLIER:.2f}x"
)

print()
print("=" * 110)
print("BASELINE")
print("=" * 110)

baseline_network = pypsa.Network(
    NETWORK_PATH
)

baseline_pf = get_pf_result(
    baseline_network
)

baseline = calculate_metrics(
    baseline_network,
    baseline_pf
)

print()
print("PF converged       :", baseline["pf_converged"])
print("PF maximum error   :", baseline["pf_max_error"])
print("PF maximum iters   :", baseline["pf_max_iterations"])
print("Minimum voltage    :", baseline["min_voltage_pu"])
print("Minimum-V bus      :", baseline["min_voltage_bus"])
print("Maximum loading    :", baseline["max_line_loading_pct"])
print("Worst line         :", baseline["worst_line"])
print("Lines >100%        :", baseline["lines_over_100"])
print("Lines >110%        :", baseline["lines_over_110"])
print("Lines >120%        :", baseline["lines_over_120"])


# ============================================================
# RUN CANDIDATES
# ============================================================

results = []

for candidate_name, lines in CANDIDATES.items():

    metrics = run_case(
        candidate_name,
        lines
    )

    row = {
        "candidate": candidate_name,
        "pf_converged": metrics["pf_converged"],
        "pf_max_error": metrics["pf_max_error"],
        "pf_max_iterations": metrics["pf_max_iterations"],
        "min_voltage_pu": metrics["min_voltage_pu"],
        "min_voltage_bus": metrics["min_voltage_bus"],
        "max_line_loading_pct": metrics["max_line_loading_pct"],
        "worst_line": metrics["worst_line"],
        "lines_over_100": metrics["lines_over_100"],
        "lines_over_110": metrics["lines_over_110"],
        "lines_over_120": metrics["lines_over_120"],
        "voltage_change_pu": (
            metrics["min_voltage_pu"]
            - baseline["min_voltage_pu"]
        ),
        "loading_change_pct_points": (
            baseline["max_line_loading_pct"]
            - metrics["max_line_loading_pct"]
        ),
        "over_100_change": (
            baseline["lines_over_100"]
            - metrics["lines_over_100"]
        ),
        "over_110_change": (
            baseline["lines_over_110"]
            - metrics["lines_over_110"]
        ),
        "over_120_change": (
            baseline["lines_over_120"]
            - metrics["lines_over_120"]
        ),
    }

    results.append(row)


# ============================================================
# COMPARISON
# ============================================================

df = pd.DataFrame(
    results
)

print()
print("=" * 110)
print("S2 VOLTAGE + THERMAL CANDIDATE COMPARISON")
print("=" * 110)

print(
    df[
        [
            "candidate",
            "pf_converged",
            "min_voltage_pu",
            "max_line_loading_pct",
            "lines_over_100",
            "lines_over_110",
            "lines_over_120",
            "voltage_change_pu",
            "loading_change_pct_points",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# ENGINEERING SCREEN
# ============================================================

print()
print("=" * 110)
print("ENGINEERING SCREEN")
print("=" * 110)

# Voltage target is deliberately conservative.
#
# 0.95 pu is used as a screening threshold.
# This is NOT being claimed as the final grid-code limit.
#
# Thermal feasibility requires every monitored line <= 100%.

VOLTAGE_TARGET = 0.95
THERMAL_TARGET = 100.0

df["thermal_feasible"] = (
    df["max_line_loading_pct"]
    <= THERMAL_TARGET
)

df["voltage_feasible"] = (
    df["min_voltage_pu"]
    >= VOLTAGE_TARGET
)

df["fully_feasible"] = (
    df["thermal_feasible"]
    & df["voltage_feasible"]
    & df["pf_converged"]
)


# ------------------------------------------------------------
# Rank candidates by:
#
# 1. PF convergence
# 2. Full feasibility
# 3. Higher minimum voltage
# 4. Lower maximum loading
# 5. Fewer overloaded lines
# ------------------------------------------------------------

df_ranked = df.sort_values(
    by=[
        "fully_feasible",
        "pf_converged",
        "min_voltage_pu",
        "max_line_loading_pct",
        "lines_over_100",
    ],
    ascending=[
        False,
        False,
        False,
        True,
        True,
    ],
)


print()
print(
    df_ranked[
        [
            "candidate",
            "pf_converged",
            "min_voltage_pu",
            "max_line_loading_pct",
            "lines_over_100",
            "thermal_feasible",
            "voltage_feasible",
            "fully_feasible",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST CANDIDATE
# ============================================================

best = df_ranked.iloc[0]

print()
print("=" * 110)
print("PRELIMINARY BEST CANDIDATE")
print("=" * 110)

print(
    "Candidate          :",
    best["candidate"]
)

print(
    "Minimum voltage    :",
    best["min_voltage_pu"],
    "pu"
)

print(
    "Maximum loading    :",
    best["max_line_loading_pct"],
    "%"
)

print(
    "Lines >100%        :",
    int(best["lines_over_100"])
)

print(
    "Lines >110%        :",
    int(best["lines_over_110"])
)

print(
    "Lines >120%        :",
    int(best["lines_over_120"])
)

print(
    "Voltage feasible   :",
    bool(best["voltage_feasible"])
)

print(
    "Thermal feasible   :",
    bool(best["thermal_feasible"])
)

print(
    "Fully feasible     :",
    bool(best["fully_feasible"])
)


# ============================================================
# DECISION
# ============================================================

print()
print("=" * 110)
print("ENGINEERING DECISION")
print("=" * 110)

if bool(best["fully_feasible"]):

    print(
        "A fully feasible candidate was found under the "
        "screening thresholds."
    )

else:

    print(
        "NO FULLY FEASIBLE CANDIDATE FOUND."
    )

    print()
    print(
        "The reinforcement candidates tested here do not "
        "simultaneously resolve the S2 voltage and thermal "
        "constraints."
    )

    print()
    print(
        "Next stage should therefore investigate the underlying "
        "voltage-support mechanism rather than simply adding "
        "more thermal line capacity."
    )


# ============================================================
# SAVE AUDIT
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 110)
print("AUDIT FILE SAVED")
print("=" * 110)

print(
    OUTPUT_PATH
)

print()
print("NO SOURCE NETWORK MODIFICATION PERFORMED.")
print("NO DISPATCH MODIFICATION PERFORMED.")
print("NO LOAD MODIFICATION PERFORMED.")
print("NO OPTIMIZATION PERFORMED.")
print("ALL REINFORCEMENTS WERE TEMPORARY IN-MEMORY TESTS.")

print()
print("=" * 110)
print("S2 VOLTAGE + THERMAL CANDIDATE SCREEN COMPLETE")
print("=" * 110)