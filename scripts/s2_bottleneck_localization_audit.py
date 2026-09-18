from pathlib import Path
import copy
import pandas as pd
import pypsa
# ============================================================
# IRELAND GRID - S2 TARGETED REINFORCEMENT CANDIDATE SCREEN
# ============================================================
NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)
SNAPSHOT = "S2_PEAK_DEMAND"
OUTPUT_PATH = Path(
    "data/processed/s2_targeted_reinforcement_screen.csv"
)
# Reinforcement multiplier.
# 1.50 means the candidate line/corridor receives 50% more capacity.
REINFORCEMENT_MULTIPLIER = 1.50
# ============================================================
# CANDIDATES
# ============================================================
CANDIDATES = {
    "A_WORST_LINE": [
        "merged_way/61295764-220+1",
    ],
    "B_THERMAL_CORRIDOR": [
        "merged_way/61295764-220+1",
        "way/343436171-220",
        "merged_way/257889771-220+1",
    ],
    "C_88462768_PATH": [
        "way/343436171-220",
    ],
    "D_MIN_VOLTAGE_SIDE": [
        "merged_way/148948901-220+1",
        "relation/5567982-220",
        "merged_way/104388602-220+1",
    ],
    "E_COMBINED_TARGETED": [
        "merged_way/61295764-220+1",
        "way/343436171-220",
        "merged_way/257889771-220+1",
        "merged_way/148948901-220+1",
        "relation/5567982-220",
        "merged_way/104388602-220+1",
    ],
}
# ============================================================
# LOAD NETWORK
# ============================================================
if not NETWORK_PATH.exists():
    raise FileNotFoundError(
        f"Network not found:\n{NETWORK_PATH}"
    )
base_network = pypsa.Network(NETWORK_PATH)
# ============================================================
# BASIC CHECK
# ============================================================
if SNAPSHOT not in base_network.snapshots:
    raise ValueError(
        f"Snapshot '{SNAPSHOT}' not found.\n"
        f"Available snapshots:\n{list(base_network.snapshots)}"
    )
# ============================================================
# HELPER: APPLY REINFORCEMENT
# ============================================================
def reinforce_lines(n, line_names, multiplier):
    missing = [
        name for name in line_names
        if name not in n.lines.index
    ]
    if missing:
        raise ValueError(
            "Candidate contains line(s) not found in network:\n"
            + "\n".join(missing)
        )
    for name in line_names:
        original = float(n.lines.at[name, "s_nom"])
        n.lines.at[name, "s_nom"] = (
            original * multiplier
        )
# ============================================================
# HELPER: RUN CONTROLLED AC PF
# ============================================================
def run_pf(n):
    # Explicitly restrict to S2.
    n.snapshots = pd.Index(
        [SNAPSHOT],
        name="snapshot"
    )
    pf = n.pf(
        snapshots=[SNAPSHOT],
        distribute_slack=True
    )
    converged_series = pf["converged"].loc[SNAPSHOT]
    error_series = pf["error"].loc[SNAPSHOT]
    iteration_series = pf["n_iter"].loc[SNAPSHOT]
    all_converged = bool(
        converged_series.all()
    )
    max_error = float(
        converged_series.index.to_series()
        .map(lambda _: 0.0).max()
    )
    # Correct extraction of PF error.
    max_error = float(
        error_series.max()
    )
    max_iterations = int(
        iteration_series.max()
    )
    return (
        pf,
        all_converged,
        max_error,
        max_iterations
    )
# ============================================================
# HELPER: CALCULATE NETWORK METRICS
# ============================================================
def calculate_metrics(n):
    # --------------------------------------------------------
    # BUS VOLTAGES
    # --------------------------------------------------------
    v = n.buses_t.v_mag_pu.loc[SNAPSHOT]
    min_voltage_bus = v.idxmin()
    max_voltage_bus = v.idxmax()
    min_voltage = float(v.min())
    max_voltage = float(v.max())
    # --------------------------------------------------------
    # LINE LOADINGS
    # --------------------------------------------------------
    s_nom = n.lines["s_nom"]
    p0 = n.lines_t.p0.loc[SNAPSHOT].abs()
    p1 = n.lines_t.p1.loc[SNAPSHOT].abs()
    loading_p0 = (
        p0 / s_nom
    ) * 100.0
    loading_p1 = (
        p1 / s_nom
    ) * 100.0
    max_loading = pd.concat(
        [
            loading_p0.rename("p0"),
            loading_p1.rename("p1"),
        ],
        axis=1
    ).max(axis=1)
    worst_line = max_loading.idxmax()
    max_line_loading = float(
        max_loading.max()
    )
    lines_over_100 = int(
        (max_loading > 100.0).sum()
    )
    lines_over_110 = int(
        (max_loading > 110.0).sum()
    )
    lines_over_120 = int(
        (max_loading > 120.0).sum()
    )
    return {
        "min_voltage_pu": min_voltage,
        "min_voltage_bus": min_voltage_bus,
        "max_voltage_pu": max_voltage,
        "max_voltage_bus": max_voltage_bus,
        "max_line_loading_pct": max_line_loading,
        "worst_line": worst_line,
        "lines_over_100": lines_over_100,
        "lines_over_110": lines_over_110,
        "lines_over_120": lines_over_120,
    }
# ============================================================
# BASELINE
# ============================================================
print("=" * 110)
print("IRELAND GRID - S2 TARGETED REINFORCEMENT CANDIDATE SCREEN")
print("=" * 110)
print()
print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print("PF       : AC nonlinear")
print("Slack    : distributed")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Source   : READ-ONLY")
print()
print(
    f"Reinforcement multiplier : "
    f"{REINFORCEMENT_MULTIPLIER:.2f}x"
)
print()
print("=" * 110)
print("BASELINE")
print("=" * 110)
baseline = pypsa.Network(NETWORK_PATH)
(
    baseline_pf,
    baseline_converged,
    baseline_error,
    baseline_iterations,
) = run_pf(baseline)
baseline_metrics = calculate_metrics(
    baseline
)
print()
print(
    f"PF converged       : {baseline_converged}"
)
print(
    f"PF maximum error   : {baseline_error:.12e}"
)
print(
    f"PF maximum iters   : {baseline_iterations}"
)
print(
    f"Minimum voltage    : "
    f"{baseline_metrics['min_voltage_pu']:.6f} pu"
)
print(
    f"Minimum-V bus      : "
    f"{baseline_metrics['min_voltage_bus']}"
)
print(
    f"Maximum loading    : "
    f"{baseline_metrics['max_line_loading_pct']:.6f}%"
)
print(
    f"Worst line         : "
    f"{baseline_metrics['worst_line']}"
)
print(
    f"Lines >100%        : "
    f"{baseline_metrics['lines_over_100']}"
)
print(
    f"Lines >110%        : "
    f"{baseline_metrics['lines_over_110']}"
)
print(
    f"Lines >120%        : "
    f"{baseline_metrics['lines_over_120']}"
)
# ============================================================
# CANDIDATE SCREENING
# ============================================================
results = []
for candidate_name, line_names in CANDIDATES.items():
    print()
    print("=" * 110)
    print(candidate_name)
    print("=" * 110)
    print()
    print("Reinforced lines:")
    for line in line_names:
        print(f"  - {line}")
    # --------------------------------------------------------
    # LOAD FRESH NETWORK
    # --------------------------------------------------------
    n = pypsa.Network(
        NETWORK_PATH
    )
    # --------------------------------------------------------
    # REINFORCE ONLY TEMPORARY COPY
    # --------------------------------------------------------
    reinforce_lines(
        n,
        line_names,
        REINFORCEMENT_MULTIPLIER
    )
    # --------------------------------------------------------
    # RUN PF
    # --------------------------------------------------------
    (
        pf,
        converged,
        max_error,
        max_iterations,
    ) = run_pf(n)
    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    metrics = calculate_metrics(n)
    # --------------------------------------------------------
    # IMPROVEMENT RELATIVE TO BASELINE
    # --------------------------------------------------------
    voltage_change = (
        metrics["min_voltage_pu"]
        - baseline_metrics["min_voltage_pu"]
    )
    loading_change = (
        metrics["max_line_loading_pct"]
        - baseline_metrics["max_line_loading_pct"]
    )
    overload_reduction = (
        baseline_metrics["lines_over_100"]
        - metrics["lines_over_100"]
    )
    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------
    print()
    print(
        f"PF converged       : {converged}"
    )
    print(
        f"PF maximum error   : "
        f"{max_error:.12e}"
    )
    print(
        f"PF maximum iters   : "
        f"{max_iterations}"
    )
    print(
        f"Minimum voltage    : "
        f"{metrics['min_voltage_pu']:.6f} pu"
    )
    print(
        f"Voltage change     : "
        f"{voltage_change:+.6f} pu"
    )
    print(
        f"Minimum-V bus      : "
        f"{metrics['min_voltage_bus']}"
    )
    print(
        f"Maximum loading    : "
        f"{metrics['max_line_loading_pct']:.6f}%"
    )
    print(
        f"Loading change     : "
        f"{loading_change:+.6f} percentage points"
    )
    print(
        f"Worst line         : "
        f"{metrics['worst_line']}"
    )
    print(
        f"Lines >100%        : "
        f"{metrics['lines_over_100']}"
    )
    print(
        f"Lines >110%        : "
        f"{metrics['lines_over_110']}"
    )
    print(
        f"Lines >120%        : "
        f"{metrics['lines_over_120']}"
    )
    print(
        f"Overload reduction : "
        f"{overload_reduction}"
    )
    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------
    results.append(
        {
            "candidate": candidate_name,
            "reinforcement_multiplier":
                REINFORCEMENT_MULTIPLIER,
            "reinforced_line_count":
                len(line_names),
            "reinforced_lines":
                " | ".join(line_names),
            "pf_converged":
                converged,
            "pf_max_error":
                max_error,
            "pf_max_iterations":
                max_iterations,
            "min_voltage_pu":
                metrics["min_voltage_pu"],
            "min_voltage_bus":
                metrics["min_voltage_bus"],
            "max_voltage_pu":
                metrics["max_voltage_pu"],
            "max_voltage_bus":
                metrics["max_voltage_bus"],
            "max_line_loading_pct":
                metrics["max_line_loading_pct"],
            "worst_line":
                metrics["worst_line"],
            "lines_over_100":
                metrics["lines_over_100"],
            "lines_over_110":
                metrics["lines_over_110"],
            "lines_over_120":
                metrics["lines_over_120"],
            "voltage_change_pu":
                voltage_change,
            "loading_change_pct_points":
                loading_change,
            "overload_reduction":
                overload_reduction,
        }
    )
# ============================================================
# COMPARISON TABLE
# ============================================================
results_df = pd.DataFrame(
    results
)
results_df = results_df.sort_values(
    by=[
        "lines_over_100",
        "max_line_loading_pct",
        "min_voltage_pu",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)
print()
print("=" * 110)
print("S2 TARGETED REINFORCEMENT COMPARISON")
print("=" * 110)
display_columns = [
    "candidate",
    "pf_converged",
    "min_voltage_pu",
    "min_voltage_bus",
    "max_line_loading_pct",
    "worst_line",
    "lines_over_100",
    "lines_over_110",
    "lines_over_120",
    "voltage_change_pu",
    "loading_change_pct_points",
]
print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)
# ============================================================
# SELECT BEST CANDIDATE
# ============================================================
successful = results_df[
    results_df["pf_converged"] == True
].copy()
if len(successful) == 0:
    print()
    print("=" * 110)
    print("NO SUCCESSFUL CANDIDATE")
    print("=" * 110)
    best_candidate = None
else:
    successful = successful.sort_values(
        by=[
            "lines_over_120",
            "lines_over_110",
            "lines_over_100",
            "max_line_loading_pct",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )
    best_candidate = (
        successful.iloc[0]["candidate"]
    )
    print()
    print("=" * 110)
    print("PRELIMINARY BEST CANDIDATE")
    print("=" * 110)
    print(
        f"Candidate : {best_candidate}"
    )
    best_row = successful.iloc[0]
    print(
        f"Minimum voltage : "
        f"{best_row['min_voltage_pu']:.6f} pu"
    )
    print(
        f"Maximum loading : "
        f"{best_row['max_line_loading_pct']:.6f}%"
    )
    print(
        f"Lines >100%     : "
        f"{int(best_row['lines_over_100'])}"
    )
    print(
        f"Lines >110%     : "
        f"{int(best_row['lines_over_110'])}"
    )
    print(
        f"Lines >120%     : "
        f"{int(best_row['lines_over_120'])}"
    )
# ============================================================
# SAVE RESULTS
# ============================================================
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)
results_df.to_csv(
    OUTPUT_PATH,
    index=False
)
print()
print("=" * 110)
print("AUDIT FILE SAVED")
print("=" * 110)
print(
    f"{OUTPUT_PATH}"
)
print()
print("NO SOURCE NETWORK MODIFICATION PERFORMED.")
print("NO DISPATCH MODIFICATION PERFORMED.")
print("NO LOAD MODIFICATION PERFORMED.")
print("NO OPTIMIZATION PERFORMED.")
print("ALL REINFORCEMENTS WERE TEMPORARY IN-MEMORY TESTS.")
print()
print("=" * 110)
print("S2 TARGETED REINFORCEMENT SCREEN COMPLETE")
print("=" * 110)