# =============================================================================
# S3.8 — TARGETED VOLTAGE SUPPORT TEST — CORRECTED
# =============================================================================
#
# Purpose:
#   Determine how much local reactive support is required at the critical
#   weak bus to improve voltage at the critical operating point.
#
# Network:
#   data/processed/eirgrid_second_reinforced_network.nc
#
# Snapshot:
#   S2_PEAK_DEMAND
#
# Critical operating point:
#   Lambda = 0.953125
#
# Weak bus:
#   way/104388595-220
#
# Source network:
#   READ-ONLY
#
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


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
    / "s3_8_voltage_support_results.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

LAMBDA = 0.953125

WEAK_BUS = "way/104388595-220"

SUPPORT_GENERATOR = "S3_8_LOCAL_REACTIVE_SUPPORT"

# -------------------------------------------------------------------------
# First broad sweep
# -------------------------------------------------------------------------

Q_SUPPORT_LEVELS = [
    0,
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
]


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("S3.8 — TARGETED VOLTAGE SUPPORT TEST — CORRECTED")
print("=" * 100)

print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print(f"Lambda   : {LAMBDA}")
print(f"Weak bus : {WEAK_BUS}")


# =============================================================================
# VALIDATION
# =============================================================================

if not NETWORK_PATH.exists():
    raise FileNotFoundError(
        f"Network file not found:\n{NETWORK_PATH}"
    )


# =============================================================================
# LOAD FRESH NETWORK
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

    return n


# =============================================================================
# APPLY CRITICAL OPERATING POINT
# =============================================================================

def apply_lambda(n):

    # -------------------------------------------------------------------------
    # Loads
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Generator dispatch
    # -------------------------------------------------------------------------

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
# RUN AC POWER FLOW
# =============================================================================

def run_pf(n):

    try:

        result = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True,
        )

        return result

    except Exception as exc:

        print("\nPOWER FLOW ERROR")
        print("-" * 80)
        print(exc)

        return None


# =============================================================================
# CALCULATE LINE LOADING
# =============================================================================

def calculate_line_loading(n):

    """
    Calculate apparent power from PyPSA line p0/q0.

        S = sqrt(P^2 + Q^2)

    Loading:

        loading[%] = S / s_nom * 100
    """

    if len(n.lines) == 0:

        return pd.Series(
            dtype=float
        )

    try:

        p0 = n.lines_t.p0.loc[
            SNAPSHOT
        ]

        q0 = n.lines_t.q0.loc[
            SNAPSHOT
        ]

        p0 = pd.to_numeric(
            p0,
            errors="coerce"
        )

        q0 = pd.to_numeric(
            q0,
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

        loading = loading.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return loading

    except Exception as exc:

        print(
            f"\nLine loading calculation warning: {exc}"
        )

        return pd.Series(
            dtype=float
        )


# =============================================================================
# CALCULATE TRANSFORMER LOADING
# =============================================================================

def calculate_transformer_loading(n):

    if len(n.transformers) == 0:

        return pd.Series(
            dtype=float
        )

    try:

        p0 = n.transformers_t.p0.loc[
            SNAPSHOT
        ]

        q0 = n.transformers_t.q0.loc[
            SNAPSHOT
        ]

        p0 = pd.to_numeric(
            p0,
            errors="coerce"
        )

        q0 = pd.to_numeric(
            q0,
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

        loading = loading.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return loading

    except Exception as exc:

        print(
            f"\nTransformer loading calculation warning: {exc}"
        )

        return pd.Series(
            dtype=float
        )


# =============================================================================
# EXTRACT RESULTS
# =============================================================================

def extract_results(
    n,
    q_support,
    pf_result
):

    # -------------------------------------------------------------------------
    # Voltage
    # -------------------------------------------------------------------------

    try:

        voltage = pd.to_numeric(
            n.buses_t.v_mag_pu.loc[
                SNAPSHOT
            ],
            errors="coerce"
        )

        finite_voltage = voltage.dropna()

        converged = (
            len(finite_voltage) > 0
            and np.isfinite(
                finite_voltage.values
            ).all()
        )

    except Exception:

        voltage = pd.Series(
            dtype=float
        )

        converged = False

    if converged:

        min_voltage = float(
            voltage.min()
        )

        weak_bus_voltage = float(
            voltage.loc[WEAK_BUS]
        )

    else:

        min_voltage = np.nan
        weak_bus_voltage = np.nan

    # -------------------------------------------------------------------------
    # LINE LOADING
    # -------------------------------------------------------------------------

    if converged:

        line_loading = calculate_line_loading(
            n
        )

    else:

        line_loading = pd.Series(
            dtype=float
        )

    if len(line_loading.dropna()) > 0:

        valid_line_loading = line_loading.dropna()

        max_line_loading = float(
            valid_line_loading.max()
        )

        overloaded_lines = int(
            (
                valid_line_loading > 100.0
            ).sum()
        )

        max_loaded_line = str(
            valid_line_loading.idxmax()
        )

    else:

        max_line_loading = np.nan
        overloaded_lines = np.nan
        max_loaded_line = ""

    # -------------------------------------------------------------------------
    # TRANSFORMER LOADING
    # -------------------------------------------------------------------------

    if converged:

        transformer_loading = (
            calculate_transformer_loading(n)
        )

    else:

        transformer_loading = pd.Series(
            dtype=float
        )

    if len(
        transformer_loading.dropna()
    ) > 0:

        valid_transformer_loading = (
            transformer_loading.dropna()
        )

        max_transformer_loading = float(
            valid_transformer_loading.max()
        )

        max_loaded_transformer = str(
            valid_transformer_loading.idxmax()
        )

    else:

        max_transformer_loading = np.nan
        max_loaded_transformer = ""

    # -------------------------------------------------------------------------
    # WEAK BUS POWER
    # -------------------------------------------------------------------------

    weak_bus_p = np.nan
    weak_bus_q = np.nan

    try:

        if (
            "p" in n.buses_t
            and WEAK_BUS in n.buses_t.p.columns
        ):

            weak_bus_p = float(
                n.buses_t.p.loc[
                    SNAPSHOT,
                    WEAK_BUS
                ]
            )

        if (
            "q" in n.buses_t
            and WEAK_BUS in n.buses_t.q.columns
        ):

            weak_bus_q = float(
                n.buses_t.q.loc[
                    SNAPSHOT,
                    WEAK_BUS
                ]
            )

    except Exception:

        pass

    # -------------------------------------------------------------------------
    # Voltage adequacy flags
    # -------------------------------------------------------------------------

    voltage_ok_090 = (
        weak_bus_voltage >= 0.90
        if np.isfinite(weak_bus_voltage)
        else False
    )

    voltage_ok_095 = (
        weak_bus_voltage >= 0.95
        if np.isfinite(weak_bus_voltage)
        else False
    )

    voltage_ok_0975 = (
        weak_bus_voltage >= 0.975
        if np.isfinite(weak_bus_voltage)
        else False
    )

    voltage_ok_100 = (
        weak_bus_voltage >= 1.00
        if np.isfinite(weak_bus_voltage)
        else False
    )

    return {

        "q_support_mvar": q_support,

        "converged": converged,

        "min_voltage_pu": min_voltage,

        "weak_bus_voltage_pu":
            weak_bus_voltage,

        "voltage_ge_0.90":
            voltage_ok_090,

        "voltage_ge_0.95":
            voltage_ok_095,

        "voltage_ge_0.975":
            voltage_ok_0975,

        "voltage_ge_1.00":
            voltage_ok_100,

        "max_line_loading_pct":
            max_line_loading,

        "overloaded_lines":
            overloaded_lines,

        "max_loaded_line":
            max_loaded_line,

        "max_transformer_loading_pct":
            max_transformer_loading,

        "max_loaded_transformer":
            max_loaded_transformer,

        "weak_bus_p_mw":
            weak_bus_p,

        "weak_bus_q_mvar":
            weak_bus_q,
    }


# =============================================================================
# NETWORK INFORMATION
# =============================================================================

n0 = load_network()

print("\nNETWORK")
print("-" * 100)

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
    f"\nSnapshots   : {list(n0.snapshots)}"
)

print(
    f"\nQ levels    : {Q_SUPPORT_LEVELS}"
)


# =============================================================================
# TEST LOOP
# =============================================================================

results = []

print("\n" + "=" * 100)
print("TESTING LOCAL REACTIVE SUPPORT")
print("=" * 100)


for q_support in Q_SUPPORT_LEVELS:

    print("\n" + "-" * 100)

    print(
        f"TEST — LOCAL Q SUPPORT = "
        f"+{q_support} MVAr"
    )

    print("-" * 100)

    # -------------------------------------------------------------------------
    # Fresh network
    # -------------------------------------------------------------------------

    n = load_network()

    # -------------------------------------------------------------------------
    # Critical operating point
    # -------------------------------------------------------------------------

    apply_lambda(n)

    # -------------------------------------------------------------------------
    # Time-series values aligned EXACTLY to network snapshots
    # -------------------------------------------------------------------------

    p_set_series = pd.Series(
        0.0,
        index=n.snapshots,
        dtype=float,
    )

    q_set_series = pd.Series(
        float(q_support),
        index=n.snapshots,
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Add local reactive support
    # -------------------------------------------------------------------------

    n.add(
        "Generator",
        SUPPORT_GENERATOR,
        bus=WEAK_BUS,
        carrier="AC",

        # No active-power capacity required.
        p_nom=0.0,

        p_nom_extendable=False,

        control="PQ",

        p_set=p_set_series,

        q_set=q_set_series,
    )

    # -------------------------------------------------------------------------
    # Verify alignment
    # -------------------------------------------------------------------------

    assert n.generators_t.p_set.index.equals(
        n.snapshots
    )

    assert n.generators_t.q_set.index.equals(
        n.snapshots
    )

    # -------------------------------------------------------------------------
    # AC POWER FLOW
    # -------------------------------------------------------------------------

    pf_result = run_pf(n)

    # -------------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------------

    result = extract_results(
        n,
        q_support,
        pf_result
    )

    results.append(result)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    print("\nRESULT")
    print("-" * 80)

    print(
        f"Converged               : "
        f"{result['converged']}"
    )

    print(
        f"Min V magnitude         : "
        f"{result['min_voltage_pu']:.6f} pu"
    )

    print(
        f"Weak bus voltage        : "
        f"{result['weak_bus_voltage_pu']:.6f} pu"
    )

    print(
        f"Max line loading        : "
        f"{result['max_line_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded lines        : "
        f"{result['overloaded_lines']}"
    )

    print(
        f"Max transformer loading : "
        f"{result['max_transformer_loading_pct']:.6f} %"
    )


# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    results
)


# =============================================================================
# CLEAN NUMERIC VALID RESULTS
# =============================================================================

valid_results = results_df[
    results_df["converged"] == True
].copy()


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("S3.8 SUMMARY")
print("=" * 100)


if len(valid_results) == 0:

    print(
        "\nNO VALID AC SOLUTIONS."
    )

else:

    # -------------------------------------------------------------------------
    # BEST VOLTAGE
    # -------------------------------------------------------------------------

    voltage_valid = valid_results[
        valid_results[
            "weak_bus_voltage_pu"
        ].notna()
    ]

    best_voltage = voltage_valid.loc[
        voltage_valid[
            "weak_bus_voltage_pu"
        ].idxmax()
    ]

    print(
        "\nBEST WEAK-BUS VOLTAGE"
    )

    print("-" * 80)

    print(
        f"Q support        : "
        f"+{best_voltage['q_support_mvar']:.0f} MVAr"
    )

    print(
        f"Weak bus voltage : "
        f"{best_voltage['weak_bus_voltage_pu']:.6f} pu"
    )

    print(
        f"Minimum voltage  : "
        f"{best_voltage['min_voltage_pu']:.6f} pu"
    )

    # -------------------------------------------------------------------------
    # MINIMUM Q FOR 0.95 PU
    # -------------------------------------------------------------------------

    acceptable_095 = voltage_valid[
        voltage_valid[
            "weak_bus_voltage_pu"
        ] >= 0.95
    ]

    print(
        "\nMINIMUM TESTED Q FOR WEAK-BUS "
        "VOLTAGE >= 0.95 PU"
    )

    print("-" * 80)

    if len(acceptable_095) > 0:

        minimum_q = acceptable_095.loc[
            acceptable_095[
                "q_support_mvar"
            ].idxmin()
        ]

        print(
            f"Q support        : "
            f"+{minimum_q['q_support_mvar']:.0f} MVAr"
        )

        print(
            f"Weak bus voltage : "
            f"{minimum_q['weak_bus_voltage_pu']:.6f} pu"
        )

        print(
            f"Minimum voltage  : "
            f"{minimum_q['min_voltage_pu']:.6f} pu"
        )

    else:

        print(
            "No tested Q level reached 0.95 pu."
        )

    # -------------------------------------------------------------------------
    # BEST LINE CONGESTION
    # -------------------------------------------------------------------------

    congestion_valid = valid_results[
        valid_results[
            "max_line_loading_pct"
        ].notna()
    ]

    if len(congestion_valid) > 0:

        best_congestion = congestion_valid.loc[
            congestion_valid[
                "max_line_loading_pct"
            ].idxmin()
        ]

        print(
            "\nBEST FOR LINE CONGESTION"
        )

        print("-" * 80)

        print(
            f"Q support        : "
            f"+{best_congestion['q_support_mvar']:.0f} MVAr"
        )

        print(
            f"Max line loading : "
            f"{best_congestion['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded lines : "
            f"{best_congestion['overloaded_lines']}"
        )

        print(
            f"Critical line    : "
            f"{best_congestion['max_loaded_line']}"
        )


# =============================================================================
# FULL TABLE
# =============================================================================

print("\n" + "=" * 100)
print("FULL S3.8 RESULTS")
print("=" * 100)

display_columns = [

    "q_support_mvar",

    "converged",

    "min_voltage_pu",

    "weak_bus_voltage_pu",

    "voltage_ge_0.90",

    "voltage_ge_0.95",

    "voltage_ge_0.975",

    "voltage_ge_1.00",

    "max_line_loading_pct",

    "overloaded_lines",

    "max_loaded_line",

    "max_transformer_loading_pct",

    "max_loaded_transformer",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
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

print("\n" + "=" * 100)
print("S3.8 COMPLETE")
print("=" * 100)

print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 100)