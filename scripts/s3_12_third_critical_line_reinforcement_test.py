from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# S3.12 — THIRD CRITICAL LINE REINFORCEMENT TEST
# =============================================================================
#
# Purpose:
#   Test reinforcement of the NEW thermal bottleneck identified in S3.11.
#
# Fixed interventions:
#   1. merged_way/1231251986-220+2  -> 1.50x
#   2. merged_way/61295764-220+1    -> 1.50x
#   3. way/343436171-220            -> 1.50x
#   4. Local reactive support       -> +500 MVAr
#
# New target:
#   merged_way/257889771-220+1
#
# Test:
#   1.25x, 1.50x, 1.75x, 2.00x
#
# Source network remains READ-ONLY.
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eirgrid_second_reinforced_network.nc"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "s3_12_third_critical_line_reinforcement_results.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

LAMBDA = 0.953125

WEAK_BUS = "way/104388595-220"

# -------------------------------------------------------------------------
# Fixed reinforcement configuration
# -------------------------------------------------------------------------

EXISTING_REINFORCEMENT = "merged_way/1231251986-220+2"
EXISTING_MULTIPLIER = 1.50

CRITICAL_CORRIDOR = "merged_way/61295764-220+1"
CRITICAL_CORRIDOR_MULTIPLIER = 1.50

SECONDARY_LINE = "way/343436171-220"
SECONDARY_MULTIPLIER = 1.50

# -------------------------------------------------------------------------
# New target identified by S3.11
# -------------------------------------------------------------------------

THIRD_CRITICAL_LINE = "merged_way/257889771-220+1"

THIRD_MULTIPLIERS = [
    1.25,
    1.50,
    1.75,
    2.00,
]

Q_SUPPORT = 500.0

SUPPORT_GENERATOR = "S3_12_LOCAL_REACTIVE_SUPPORT"


# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

VOLTAGE_TARGET = 0.95
THERMAL_TARGET = 100.0


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("S3.12 — THIRD CRITICAL LINE REINFORCEMENT TEST")
print("=" * 110)

print(f"Network                 : {NETWORK_PATH}")
print(f"Snapshot                : {SNAPSHOT}")
print(f"Lambda                  : {LAMBDA}")
print(f"Weak bus                : {WEAK_BUS}")

print(
    f"\nFixed reinforcement 1   : "
    f"{EXISTING_REINFORCEMENT}"
)
print(
    f"Fixed multiplier        : "
    f"{EXISTING_MULTIPLIER:.2f}x"
)

print(
    f"\nFixed reinforcement 2   : "
    f"{CRITICAL_CORRIDOR}"
)
print(
    f"Fixed multiplier        : "
    f"{CRITICAL_CORRIDOR_MULTIPLIER:.2f}x"
)

print(
    f"\nFixed reinforcement 3   : "
    f"{SECONDARY_LINE}"
)
print(
    f"Fixed multiplier        : "
    f"{SECONDARY_MULTIPLIER:.2f}x"
)

print(
    f"\nThird critical line     : "
    f"{THIRD_CRITICAL_LINE}"
)

print(
    f"Test multipliers        : "
    f"{THIRD_MULTIPLIERS}"
)

print(
    f"Q support               : "
    f"+{Q_SUPPORT:.0f} MVAr"
)


# =============================================================================
# VALIDATION
# =============================================================================

if not NETWORK_PATH.exists():

    raise FileNotFoundError(
        f"Network file not found:\n{NETWORK_PATH}"
    )


# =============================================================================
# LOAD NETWORK
# =============================================================================

def load_network():

    n = pypsa.Network(
        str(NETWORK_PATH)
    )

    if SNAPSHOT not in n.snapshots:

        raise ValueError(
            f"Snapshot {SNAPSHOT} not found.\n"
            f"Available snapshots: {list(n.snapshots)}"
        )

    required_lines = [
        EXISTING_REINFORCEMENT,
        CRITICAL_CORRIDOR,
        SECONDARY_LINE,
        THIRD_CRITICAL_LINE,
    ]

    for line in required_lines:

        if line not in n.lines.index:

            raise ValueError(
                f"Required line not found:\n{line}"
            )

    if WEAK_BUS not in n.buses.index:

        raise ValueError(
            f"Weak bus not found:\n{WEAK_BUS}"
        )

    return n


# =============================================================================
# APPLY CRITICAL OPERATING POINT
# =============================================================================

def apply_lambda(n):

    # -------------------------------------------------------------------------
    # Scale loads
    # -------------------------------------------------------------------------

    for load in n.loads.index:

        if load in n.loads_t.p_set.columns:

            value = n.loads_t.p_set.loc[
                SNAPSHOT,
                load
            ]

            if pd.notna(value):

                n.loads_t.p_set.loc[
                    SNAPSHOT,
                    load
                ] = value * LAMBDA

    # -------------------------------------------------------------------------
    # Scale generator dispatch
    # -------------------------------------------------------------------------

    for generator in n.generators.index:

        if generator in n.generators_t.p_set.columns:

            value = n.generators_t.p_set.loc[
                SNAPSHOT,
                generator
            ]

            if pd.notna(value):

                n.generators_t.p_set.loc[
                    SNAPSHOT,
                    generator
                ] = value * LAMBDA


# =============================================================================
# APPLY LINE REINFORCEMENT
# =============================================================================

def reinforce_line(
    n,
    line_name,
    multiplier
):

    original = float(
        n.lines.at[
            line_name,
            "s_nom"
        ]
    )

    new_capacity = (
        original * multiplier
    )

    n.lines.at[
        line_name,
        "s_nom"
    ] = new_capacity

    return original, new_capacity


# =============================================================================
# ADD LOCAL REACTIVE SUPPORT
# =============================================================================

def add_reactive_support(n):

    p_set_series = pd.Series(
        0.0,
        index=n.snapshots,
        dtype=float,
    )

    q_set_series = pd.Series(
        Q_SUPPORT,
        index=n.snapshots,
        dtype=float,
    )

    n.add(
        "Generator",
        SUPPORT_GENERATOR,
        bus=WEAK_BUS,
        carrier="AC",
        p_nom=0.0,
        p_nom_extendable=False,
        control="PQ",
        p_set=p_set_series,
        q_set=q_set_series,
    )

    assert n.generators_t.p_set.index.equals(
        n.snapshots
    )

    assert n.generators_t.q_set.index.equals(
        n.snapshots
    )


# =============================================================================
# CALCULATE LINE LOADING
# =============================================================================

def calculate_line_loading(n):

    p0 = pd.to_numeric(
        n.lines_t.p0.loc[SNAPSHOT],
        errors="coerce"
    )

    q0 = pd.to_numeric(
        n.lines_t.q0.loc[SNAPSHOT],
        errors="coerce"
    )

    apparent_power = np.sqrt(
        p0 ** 2 + q0 ** 2
    )

    s_nom = pd.to_numeric(
        n.lines.s_nom,
        errors="coerce"
    )

    loading = (
        apparent_power
        / s_nom
        * 100.0
    )

    return loading.replace(
        [np.inf, -np.inf],
        np.nan
    )


# =============================================================================
# CALCULATE TRANSFORMER LOADING
# =============================================================================

def calculate_transformer_loading(n):

    if len(n.transformers) == 0:

        return pd.Series(
            dtype=float
        )

    p0 = pd.to_numeric(
        n.transformers_t.p0.loc[SNAPSHOT],
        errors="coerce"
    )

    q0 = pd.to_numeric(
        n.transformers_t.q0.loc[SNAPSHOT],
        errors="coerce"
    )

    apparent_power = np.sqrt(
        p0 ** 2 + q0 ** 2
    )

    s_nom = pd.to_numeric(
        n.transformers.s_nom,
        errors="coerce"
    )

    loading = (
        apparent_power
        / s_nom
        * 100.0
    )

    return loading.replace(
        [np.inf, -np.inf],
        np.nan
    )


# =============================================================================
# RUN POWER FLOW
# =============================================================================

def run_power_flow(n):

    try:

        n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True,
        )

        return True

    except Exception as exc:

        print("\nPOWER FLOW ERROR")
        print("-" * 80)
        print(exc)

        return False


# =============================================================================
# EXTRACT RESULTS
# =============================================================================

def extract_results(
    n,
    third_multiplier,
    converged,
    original_third,
    new_third,
):

    if not converged:

        return {
            "third_multiplier":
                third_multiplier,

            "q_support_mvar":
                Q_SUPPORT,

            "converged":
                False,

            "min_voltage_pu":
                np.nan,

            "weak_bus_voltage_pu":
                np.nan,

            "max_line_loading_pct":
                np.nan,

            "overloaded_lines":
                np.nan,

            "max_loaded_line":
                "",

            "existing_reinforced_line_loading_pct":
                np.nan,

            "critical_corridor_loading_pct":
                np.nan,

            "secondary_critical_line_loading_pct":
                np.nan,

            "third_critical_line_loading_pct":
                np.nan,

            "max_transformer_loading_pct":
                np.nan,

            "weak_voltage_ok":
                False,

            "minimum_voltage_ok":
                False,

            "thermal_loading_ok":
                False,

            "overload_count_ok":
                False,

            "fully_acceptable":
                False,

            "third_original_s_nom_mw":
                original_third,

            "third_new_s_nom_mw":
                new_third,
        }

    # -------------------------------------------------------------------------
    # Voltage
    # -------------------------------------------------------------------------

    voltage = pd.to_numeric(
        n.buses_t.v_mag_pu.loc[SNAPSHOT],
        errors="coerce"
    )

    finite_voltage = voltage.dropna()

    valid_voltage = (
        len(finite_voltage) > 0
        and np.isfinite(
            finite_voltage.values
        ).all()
    )

    if not valid_voltage:

        return {
            "third_multiplier":
                third_multiplier,

            "q_support_mvar":
                Q_SUPPORT,

            "converged":
                False,

            "min_voltage_pu":
                np.nan,

            "weak_bus_voltage_pu":
                np.nan,

            "max_line_loading_pct":
                np.nan,

            "overloaded_lines":
                np.nan,

            "max_loaded_line":
                "",

            "existing_reinforced_line_loading_pct":
                np.nan,

            "critical_corridor_loading_pct":
                np.nan,

            "secondary_critical_line_loading_pct":
                np.nan,

            "third_critical_line_loading_pct":
                np.nan,

            "max_transformer_loading_pct":
                np.nan,

            "weak_voltage_ok":
                False,

            "minimum_voltage_ok":
                False,

            "thermal_loading_ok":
                False,

            "overload_count_ok":
                False,

            "fully_acceptable":
                False,

            "third_original_s_nom_mw":
                original_third,

            "third_new_s_nom_mw":
                new_third,
        }

    min_voltage = float(
        voltage.min()
    )

    weak_bus_voltage = float(
        voltage.loc[WEAK_BUS]
    )

    # -------------------------------------------------------------------------
    # Line loading
    # -------------------------------------------------------------------------

    line_loading = calculate_line_loading(n)

    valid_line_loading = (
        line_loading.dropna()
    )

    max_line_loading = float(
        valid_line_loading.max()
    )

    overloaded_lines = int(
        (
            valid_line_loading
            > THERMAL_TARGET
        ).sum()
    )

    max_loaded_line = str(
        valid_line_loading.idxmax()
    )

    # -------------------------------------------------------------------------
    # Specific line loadings
    # -------------------------------------------------------------------------

    existing_loading = float(
        line_loading.loc[
            EXISTING_REINFORCEMENT
        ]
    )

    critical_loading = float(
        line_loading.loc[
            CRITICAL_CORRIDOR
        ]
    )

    secondary_loading = float(
        line_loading.loc[
            SECONDARY_LINE
        ]
    )

    third_loading = float(
        line_loading.loc[
            THIRD_CRITICAL_LINE
        ]
    )

    # -------------------------------------------------------------------------
    # Transformer loading
    # -------------------------------------------------------------------------

    transformer_loading = (
        calculate_transformer_loading(n)
    )

    valid_transformer_loading = (
        transformer_loading.dropna()
    )

    if len(valid_transformer_loading) > 0:

        max_transformer_loading = float(
            valid_transformer_loading.max()
        )

    else:

        max_transformer_loading = np.nan

    # -------------------------------------------------------------------------
    # Acceptance flags
    # -------------------------------------------------------------------------

    weak_voltage_ok = (
        weak_bus_voltage
        >= VOLTAGE_TARGET
    )

    minimum_voltage_ok = (
        min_voltage
        >= VOLTAGE_TARGET
    )

    thermal_loading_ok = (
        max_line_loading
        <= THERMAL_TARGET
    )

    overload_count_ok = (
        overloaded_lines
        == 0
    )

    fully_acceptable = (
        weak_voltage_ok
        and minimum_voltage_ok
        and thermal_loading_ok
        and overload_count_ok
    )

    return {

        "third_multiplier":
            third_multiplier,

        "q_support_mvar":
            Q_SUPPORT,

        "converged":
            True,

        "min_voltage_pu":
            min_voltage,

        "weak_bus_voltage_pu":
            weak_bus_voltage,

        "max_line_loading_pct":
            max_line_loading,

        "overloaded_lines":
            overloaded_lines,

        "max_loaded_line":
            max_loaded_line,

        "existing_reinforced_line_loading_pct":
            existing_loading,

        "critical_corridor_loading_pct":
            critical_loading,

        "secondary_critical_line_loading_pct":
            secondary_loading,

        "third_critical_line_loading_pct":
            third_loading,

        "max_transformer_loading_pct":
            max_transformer_loading,

        "weak_voltage_ok":
            weak_voltage_ok,

        "minimum_voltage_ok":
            minimum_voltage_ok,

        "thermal_loading_ok":
            thermal_loading_ok,

        "overload_count_ok":
            overload_count_ok,

        "fully_acceptable":
            fully_acceptable,

        "third_original_s_nom_mw":
            original_third,

        "third_new_s_nom_mw":
            new_third,
    }


# =============================================================================
# NETWORK INFORMATION
# =============================================================================

n0 = load_network()

print("\nNETWORK")
print("-" * 110)

print(
    f"Buses        : {len(n0.buses)}"
)

print(
    f"Lines        : {len(n0.lines)}"
)

print(
    f"Transformers : {len(n0.transformers)}"
)

print(
    f"Generators   : {len(n0.generators)}"
)

print(
    f"Loads        : {len(n0.loads)}"
)

print(
    f"\nSnapshots    : {list(n0.snapshots)}"
)


# =============================================================================
# TEST LOOP
# =============================================================================

results = []

print("\n" + "=" * 110)
print(
    "TESTING THIRD CRITICAL LINE REINFORCEMENT "
    "+ LOCAL REACTIVE SUPPORT"
)
print("=" * 110)


for third_multiplier in THIRD_MULTIPLIERS:

    print("\n" + "-" * 110)

    print(
        f"TEST — existing {EXISTING_MULTIPLIER:.2f}x"
        f" — critical corridor "
        f"{CRITICAL_CORRIDOR_MULTIPLIER:.2f}x"
        f" — {SECONDARY_LINE} "
        f"{SECONDARY_MULTIPLIER:.2f}x"
        f" — {THIRD_CRITICAL_LINE} "
        f"{third_multiplier:.2f}x"
        f" — +{Q_SUPPORT:.0f} MVAr"
    )

    print("-" * 110)

    # -------------------------------------------------------------------------
    # Fresh network
    # -------------------------------------------------------------------------

    n = load_network()

    # -------------------------------------------------------------------------
    # Critical operating point
    # -------------------------------------------------------------------------

    apply_lambda(n)

    # -------------------------------------------------------------------------
    # Fixed reinforcements
    # -------------------------------------------------------------------------

    original_existing, new_existing = reinforce_line(
        n,
        EXISTING_REINFORCEMENT,
        EXISTING_MULTIPLIER
    )

    original_critical, new_critical = reinforce_line(
        n,
        CRITICAL_CORRIDOR,
        CRITICAL_CORRIDOR_MULTIPLIER
    )

    original_secondary, new_secondary = reinforce_line(
        n,
        SECONDARY_LINE,
        SECONDARY_MULTIPLIER
    )

    # -------------------------------------------------------------------------
    # Third reinforcement under test
    # -------------------------------------------------------------------------

    original_third, new_third = reinforce_line(
        n,
        THIRD_CRITICAL_LINE,
        third_multiplier
    )

    print(
        f"Existing reinforcement s_nom : "
        f"{original_existing:.6f} -> "
        f"{new_existing:.6f} MW"
    )

    print(
        f"Critical corridor s_nom       : "
        f"{original_critical:.6f} -> "
        f"{new_critical:.6f} MW"
    )

    print(
        f"Secondary line s_nom           : "
        f"{original_secondary:.6f} -> "
        f"{new_secondary:.6f} MW"
    )

    print(
        f"Third critical line s_nom      : "
        f"{original_third:.6f} -> "
        f"{new_third:.6f} MW"
    )

    # -------------------------------------------------------------------------
    # Local reactive support
    # -------------------------------------------------------------------------

    add_reactive_support(n)

    # -------------------------------------------------------------------------
    # AC power flow
    # -------------------------------------------------------------------------

    converged = run_power_flow(n)

    # -------------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------------

    result = extract_results(
        n,
        third_multiplier,
        converged,
        original_third,
        new_third,
    )

    results.append(result)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    print("\nRESULT")
    print("-" * 100)

    print(
        f"Converged                 : "
        f"{result['converged']}"
    )

    if result["converged"]:

        print(
            f"Min V magnitude           : "
            f"{result['min_voltage_pu']:.6f} pu"
        )

        print(
            f"Weak bus voltage          : "
            f"{result['weak_bus_voltage_pu']:.6f} pu"
        )

        print(
            f"Max line loading          : "
            f"{result['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded lines          : "
            f"{result['overloaded_lines']}"
        )

        print(
            f"Critical/max loaded line  : "
            f"{result['max_loaded_line']}"
        )

        print(
            f"Existing reinforced load  : "
            f"{result['existing_reinforced_line_loading_pct']:.6f} %"
        )

        print(
            f"Critical corridor loading: "
            f"{result['critical_corridor_loading_pct']:.6f} %"
        )

        print(
            f"Secondary line loading    : "
            f"{result['secondary_critical_line_loading_pct']:.6f} %"
        )

        print(
            f"Third critical loading   : "
            f"{result['third_critical_line_loading_pct']:.6f} %"
        )

        print(
            f"Max transformer loading  : "
            f"{result['max_transformer_loading_pct']:.6f} %"
        )

        print(
            f"Fully acceptable         : "
            f"{result['fully_acceptable']}"
        )


# =============================================================================
# DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    results
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 110)
print("S3.12 SUMMARY")
print("=" * 110)

summary_columns = [

    "third_multiplier",

    "q_support_mvar",

    "converged",

    "min_voltage_pu",

    "weak_bus_voltage_pu",

    "max_line_loading_pct",

    "overloaded_lines",

    "max_loaded_line",

    "existing_reinforced_line_loading_pct",

    "critical_corridor_loading_pct",

    "secondary_critical_line_loading_pct",

    "third_critical_line_loading_pct",

    "max_transformer_loading_pct",

    "weak_voltage_ok",

    "minimum_voltage_ok",

    "thermal_loading_ok",

    "overload_count_ok",

    "fully_acceptable",
]

print(
    results_df[
        summary_columns
    ].to_string(
        index=False
    )
)


# =============================================================================
# FULLY ACCEPTABLE CASES
# =============================================================================

print("\n" + "-" * 110)
print("FULLY ACCEPTABLE CASES")
print("-" * 110)

acceptable = results_df[
    results_df["fully_acceptable"] == True
]

if len(acceptable) == 0:

    print(
        "NO CASE SATISFIES ALL ACCEPTANCE CRITERIA."
    )

else:

    print(
        acceptable[
            summary_columns
        ].to_string(
            index=False
        )
    )


# =============================================================================
# BEST THERMAL CASE
# =============================================================================

valid = results_df[
    results_df["converged"] == True
].copy()

if len(valid) > 0:

    best = valid.loc[
        valid[
            "max_line_loading_pct"
        ].idxmin()
    ]

    print("\n" + "-" * 110)
    print("BEST THERMAL CASE")
    print("-" * 110)

    print(
        f"Third reinforcement : "
        f"{best['third_multiplier']:.2f}x"
    )

    print(
        f"Max loading         : "
        f"{best['max_line_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded lines    : "
        f"{best['overloaded_lines']}"
    )

    print(
        f"Critical line       : "
        f"{best['max_loaded_line']}"
    )


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =============================================================================
# COMPLETE
# =============================================================================

print("\n" + "=" * 110)
print("S3.12 COMPLETE")
print("=" * 110)

print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 110)