from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# S3.11 — SECONDARY CRITICAL LINE REINFORCEMENT TEST
# =============================================================================
#
# Purpose:
#   Continue the controlled congestion-reinforcement sequence identified
#   through S3.7 -> S3.10.
#
# Existing successful interventions:
#
#   1. merged_way/1231251986-220+2  -> 1.50x
#   2. merged_way/61295764-220+1    -> 1.50x
#   3. Local reactive support       -> +500 MVAr
#
# Newly emerged critical line:
#
#   way/343436171-220
#
# Test:
#
#   way/343436171-220 -> 1.50x, 1.75x, 2.00x
#
# Source network is READ-ONLY.
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
    / "s3_11_secondary_critical_line_results.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

LAMBDA = 0.953125

WEAK_BUS = "way/104388595-220"

# Existing reinforcement from S3.7/S3.9/S3.10
EXISTING_REINFORCEMENT = "merged_way/1231251986-220+2"
EXISTING_MULTIPLIER = 1.50

# Critical corridor from S3.10
CRITICAL_CORRIDOR = "merged_way/61295764-220+1"
CRITICAL_CORRIDOR_MULTIPLIER = 1.50

# Newly emerged critical line from S3.10
SECONDARY_CRITICAL_LINE = "way/343436171-220"

# Local voltage support already established as sufficient
Q_SUPPORT = 500.0

# Test progressively stronger reinforcement
SECONDARY_MULTIPLIERS = [1.50, 1.75, 2.00]

SUPPORT_GENERATOR = "S3_11_LOCAL_REACTIVE_SUPPORT"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("S3.11 — SECONDARY CRITICAL LINE REINFORCEMENT TEST")
print("=" * 110)

print(f"Network                 : {NETWORK_PATH}")
print(f"Snapshot                : {SNAPSHOT}")
print(f"Lambda                  : {LAMBDA}")
print(f"Weak bus                : {WEAK_BUS}")
print()
print(
    f"Existing reinforcement  : "
    f"{EXISTING_REINFORCEMENT}"
)
print(
    f"Existing multiplier     : "
    f"{EXISTING_MULTIPLIER:.2f}x"
)
print()
print(
    f"Critical corridor       : "
    f"{CRITICAL_CORRIDOR}"
)
print(
    f"Critical corridor mult. : "
    f"{CRITICAL_CORRIDOR_MULTIPLIER:.2f}x"
)
print()
print(
    f"Secondary critical line : "
    f"{SECONDARY_CRITICAL_LINE}"
)
print(
    f"Test multipliers        : "
    f"{SECONDARY_MULTIPLIERS}"
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

    if WEAK_BUS not in n.buses.index:
        raise ValueError(
            f"Weak bus {WEAK_BUS} not found."
        )

    required_lines = [
        EXISTING_REINFORCEMENT,
        CRITICAL_CORRIDOR,
        SECONDARY_CRITICAL_LINE,
    ]

    for line in required_lines:

        if line not in n.lines.index:

            raise ValueError(
                f"Required line not found: {line}"
            )

    return n


# =============================================================================
# APPLY CRITICAL OPERATING POINT
# =============================================================================

def apply_lambda(n):

    # Loads
    if "p_set" in n.loads_t:

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

    # Generator dispatch
    if "p_set" in n.generators_t:

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
# LINE LOADING
# =============================================================================

def calculate_line_loading(n):

    if len(n.lines) == 0:

        return pd.Series(dtype=float)

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
# TRANSFORMER LOADING
# =============================================================================

def calculate_transformer_loading(n):

    if len(n.transformers) == 0:

        return pd.Series(dtype=float)

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
# RESULT EXTRACTION
# =============================================================================

def extract_results(
    n,
    secondary_multiplier,
):

    # -------------------------------------------------------------------------
    # Voltage
    # -------------------------------------------------------------------------

    voltage = pd.to_numeric(
        n.buses_t.v_mag_pu.loc[SNAPSHOT],
        errors="coerce"
    )

    finite_voltage = voltage.dropna()

    converged = (
        len(finite_voltage) > 0
        and np.isfinite(
            finite_voltage.values
        ).all()
    )

    if converged:

        min_voltage = float(
            voltage.min()
        )

        weak_voltage = float(
            voltage.loc[WEAK_BUS]
        )

    else:

        min_voltage = np.nan
        weak_voltage = np.nan


    # -------------------------------------------------------------------------
    # LINE LOADING
    # -------------------------------------------------------------------------

    line_loading = (
        calculate_line_loading(n)
        if converged
        else pd.Series(dtype=float)
    )

    valid_loading = line_loading.dropna()

    if len(valid_loading):

        max_loading = float(
            valid_loading.max()
        )

        overloaded = int(
            (
                valid_loading > 100.0
            ).sum()
        )

        max_loaded_line = str(
            valid_loading.idxmax()
        )

    else:

        max_loading = np.nan
        overloaded = np.nan
        max_loaded_line = ""


    # -------------------------------------------------------------------------
    # SPECIFIC LINE LOADINGS
    # -------------------------------------------------------------------------

    existing_loading = float(
        line_loading.loc[EXISTING_REINFORCEMENT]
    )

    corridor_loading = float(
        line_loading.loc[CRITICAL_CORRIDOR]
    )

    secondary_loading = float(
        line_loading.loc[SECONDARY_CRITICAL_LINE]
    )


    # -------------------------------------------------------------------------
    # TRANSFORMER LOADING
    # -------------------------------------------------------------------------

    transformer_loading = (
        calculate_transformer_loading(n)
        if converged
        else pd.Series(dtype=float)
    )

    valid_transformers = (
        transformer_loading.dropna()
    )

    if len(valid_transformers):

        max_transformer_loading = float(
            valid_transformers.max()
        )

        max_loaded_transformer = str(
            valid_transformers.idxmax()
        )

    else:

        max_transformer_loading = np.nan
        max_loaded_transformer = ""


    # -------------------------------------------------------------------------
    # ACCEPTANCE FLAGS
    # -------------------------------------------------------------------------

    weak_voltage_ok = (
        weak_voltage >= 0.95
        if np.isfinite(weak_voltage)
        else False
    )

    minimum_voltage_ok = (
        min_voltage >= 0.95
        if np.isfinite(min_voltage)
        else False
    )

    thermal_loading_ok = (
        max_loading <= 100.0
        if np.isfinite(max_loading)
        else False
    )

    overload_count_ok = (
        overloaded == 0
        if np.isfinite(overloaded)
        else False
    )

    fully_acceptable = (
        converged
        and weak_voltage_ok
        and minimum_voltage_ok
        and thermal_loading_ok
        and overload_count_ok
    )


    return {

        "secondary_multiplier":
            secondary_multiplier,

        "q_support_mvar":
            Q_SUPPORT,

        "converged":
            converged,

        "min_voltage_pu":
            min_voltage,

        "weak_bus_voltage_pu":
            weak_voltage,

        "max_line_loading_pct":
            max_loading,

        "overloaded_lines":
            overloaded,

        "max_loaded_line":
            max_loaded_line,

        "existing_reinforced_line_loading_pct":
            existing_loading,

        "critical_corridor_loading_pct":
            corridor_loading,

        "secondary_critical_line_loading_pct":
            secondary_loading,

        "max_transformer_loading_pct":
            max_transformer_loading,

        "max_loaded_transformer":
            max_loaded_transformer,

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
    f"Snapshots    : {list(n0.snapshots)}"
)


# =============================================================================
# TEST LOOP
# =============================================================================

results = []

print("\n" + "=" * 110)
print(
    "TESTING SECONDARY CRITICAL LINE REINFORCEMENT"
)
print("=" * 110)


for multiplier in SECONDARY_MULTIPLIERS:

    print("\n" + "-" * 110)

    print(
        f"TEST — existing {EXISTING_MULTIPLIER:.2f}x — "
        f"critical corridor {CRITICAL_CORRIDOR_MULTIPLIER:.2f}x — "
        f"{SECONDARY_CRITICAL_LINE} — "
        f"{multiplier:.2f}x — "
        f"+{Q_SUPPORT:.0f} MVAr"
    )

    print("-" * 110)


    # -------------------------------------------------------------------------
    # Fresh network
    # -------------------------------------------------------------------------

    n = load_network()

    apply_lambda(n)


    # -------------------------------------------------------------------------
    # Existing reinforcement
    # -------------------------------------------------------------------------

    original_existing = float(
        n.lines.at[
            EXISTING_REINFORCEMENT,
            "s_nom"
        ]
    )

    n.lines.at[
        EXISTING_REINFORCEMENT,
        "s_nom"
    ] = (
        original_existing
        * EXISTING_MULTIPLIER
    )


    # -------------------------------------------------------------------------
    # Critical corridor reinforcement
    # -------------------------------------------------------------------------

    original_corridor = float(
        n.lines.at[
            CRITICAL_CORRIDOR,
            "s_nom"
        ]
    )

    n.lines.at[
        CRITICAL_CORRIDOR,
        "s_nom"
    ] = (
        original_corridor
        * CRITICAL_CORRIDOR_MULTIPLIER
    )


    # -------------------------------------------------------------------------
    # Secondary critical line reinforcement
    # -------------------------------------------------------------------------

    original_secondary = float(
        n.lines.at[
            SECONDARY_CRITICAL_LINE,
            "s_nom"
        ]
    )

    new_secondary = (
        original_secondary
        * multiplier
    )

    n.lines.at[
        SECONDARY_CRITICAL_LINE,
        "s_nom"
    ] = new_secondary


    print(
        f"Existing reinforcement s_nom : "
        f"{original_existing:.6f} -> "
        f"{n.lines.at[EXISTING_REINFORCEMENT, 's_nom']:.6f} MW"
    )

    print(
        f"Critical corridor s_nom       : "
        f"{original_corridor:.6f} -> "
        f"{n.lines.at[CRITICAL_CORRIDOR, 's_nom']:.6f} MW"
    )

    print(
        f"Secondary line s_nom           : "
        f"{original_secondary:.6f} -> "
        f"{new_secondary:.6f} MW"
    )


    # -------------------------------------------------------------------------
    # Reactive support
    # -------------------------------------------------------------------------

    add_reactive_support(n)


    # -------------------------------------------------------------------------
    # AC POWER FLOW
    # -------------------------------------------------------------------------

    try:

        n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True,
        )

        result = extract_results(
            n,
            multiplier
        )

        results.append(result)


        print("\nRESULT")
        print("-" * 100)

        print(
            f"Converged                 : "
            f"{result['converged']}"
        )

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
            f"Max transformer loading   : "
            f"{result['max_transformer_loading_pct']:.6f} %"
        )

        print(
            f"Fully acceptable          : "
            f"{result['fully_acceptable']}"
        )


    except Exception as exc:

        print("\nPOWER FLOW FAILED")
        print("-" * 100)
        print(exc)

        results.append({

            "secondary_multiplier":
                multiplier,

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

            "max_transformer_loading_pct":
                np.nan,

            "max_loaded_transformer":
                "",

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
        })


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
print("S3.11 SUMMARY")
print("=" * 110)

if len(results_df):

    print(
        results_df.to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # FULLY ACCEPTABLE
    # -------------------------------------------------------------------------

    acceptable = results_df[
        results_df["fully_acceptable"] == True
    ]

    print("\n" + "-" * 110)
    print("FULLY ACCEPTABLE CASES")
    print("-" * 110)

    if len(acceptable):

        print(
            acceptable.to_string(
                index=False
            )
        )

    else:

        print(
            "NO CASE SATISFIES ALL FIVE "
            "ACCEPTANCE CRITERIA."
        )


    # -------------------------------------------------------------------------
    # BEST THERMAL CASE
    # -------------------------------------------------------------------------

    valid = results_df[
        results_df["converged"] == True
    ]

    if len(valid):

        best = valid.loc[
            valid[
                "max_line_loading_pct"
            ].idxmin()
        ]

        print("\n" + "-" * 110)
        print("BEST THERMAL CASE")
        print("-" * 110)

        print(
            f"Secondary reinforcement : "
            f"{best['secondary_multiplier']:.2f}x"
        )

        print(
            f"Max loading             : "
            f"{best['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded lines        : "
            f"{best['overloaded_lines']}"
        )

        print(
            f"Critical line           : "
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
print("S3.11 COMPLETE")
print("=" * 110)

print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 110)