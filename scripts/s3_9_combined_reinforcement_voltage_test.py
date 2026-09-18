# =============================================================================
# S3.9 — COMBINED REINFORCEMENT + VOLTAGE SUPPORT TEST
# =============================================================================
#
# Purpose:
#   Test whether the strongest thermal reinforcement candidate identified in
#   S3.7, combined with local reactive support identified in S3.8, can
#   simultaneously satisfy voltage and thermal operating constraints.
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
# Thermal candidate:
#   merged_way/1231251986-220+2
#
# Reinforcement:
#   1.25x
#   1.50x
#
# Local reactive support:
#   0
#   200
#   300
#   400
#   500 MVAr
#
# Source network:
#   READ-ONLY
#
# Acceptance criteria:
#   AC convergence          = TRUE
#   Weak bus voltage        >= 0.95 pu
#   Minimum system voltage  >= 0.95 pu
#   Maximum line loading    <= 100 %
#   Overloaded lines        = 0
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
    / "s3_9_combined_reinforcement_voltage_results.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

LAMBDA = 0.953125

WEAK_BUS = "way/104388595-220"

THERMAL_CANDIDATE = "merged_way/1231251986-220+2"

REINFORCEMENT_MULTIPLIERS = [
    1.25,
    1.50,
]

Q_SUPPORT_LEVELS = [
    0,
    200,
    300,
    400,
    500,
]

SUPPORT_GENERATOR = "S3_9_LOCAL_REACTIVE_SUPPORT"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 110)
print("S3.9 — COMBINED REINFORCEMENT + VOLTAGE SUPPORT TEST")
print("=" * 110)

print(f"Network          : {NETWORK_PATH}")
print(f"Snapshot         : {SNAPSHOT}")
print(f"Lambda           : {LAMBDA}")
print(f"Weak bus         : {WEAK_BUS}")
print(f"Thermal candidate: {THERMAL_CANDIDATE}")

print(
    f"Reinforcement    : "
    f"{REINFORCEMENT_MULTIPLIERS}"
)

print(
    f"Q support        : "
    f"{Q_SUPPORT_LEVELS}"
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

    if THERMAL_CANDIDATE not in n.lines.index:
        raise ValueError(
            f"Thermal candidate {THERMAL_CANDIDATE} "
            f"not found in network lines."
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
# CALCULATE LINE LOADING
# =============================================================================

def calculate_line_loading(n):

    if len(n.lines) == 0:

        return pd.Series(
            dtype=float
        )

    try:

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
        print("-" * 90)
        print(exc)

        return None


# =============================================================================
# EXTRACT RESULTS
# =============================================================================

def extract_results(
    n,
    multiplier,
    q_support,
    original_s_nom,
    pf_result,
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
            and (finite_voltage.abs() < 2.0).all()
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

        line_loading = calculate_line_loading(n)

    else:

        line_loading = pd.Series(
            dtype=float
        )

    if len(line_loading.dropna()) > 0:

        valid_line_loading = (
            line_loading.dropna()
        )

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

        candidate_loading = float(
            valid_line_loading.loc[
                THERMAL_CANDIDATE
            ]
        )

    else:

        max_line_loading = np.nan
        overloaded_lines = np.nan
        max_loaded_line = ""
        candidate_loading = np.nan


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
    # ACCEPTANCE CRITERIA
    # -------------------------------------------------------------------------

    ac_ok = bool(converged)

    weak_voltage_ok = bool(
        np.isfinite(weak_bus_voltage)
        and weak_bus_voltage >= 0.95
    )

    minimum_voltage_ok = bool(
        np.isfinite(min_voltage)
        and min_voltage >= 0.95
    )

    thermal_loading_ok = bool(
        np.isfinite(max_line_loading)
        and max_line_loading <= 100.0
    )

    overload_count_ok = bool(
        np.isfinite(overloaded_lines)
        and overloaded_lines == 0
    )

    fully_acceptable = bool(
        ac_ok
        and weak_voltage_ok
        and minimum_voltage_ok
        and thermal_loading_ok
        and overload_count_ok
    )


    # -------------------------------------------------------------------------
    # RESULT
    # -------------------------------------------------------------------------

    return {

        "candidate":
            THERMAL_CANDIDATE,

        "reinforcement_multiplier":
            multiplier,

        "q_support_mvar":
            q_support,

        "original_s_nom_mw":
            original_s_nom,

        "reinforced_s_nom_mw":
            original_s_nom * multiplier,

        "converged":
            converged,

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

        "reinforced_line_loading_pct":
            candidate_loading,

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

print(
    f"\nCandidate    : {THERMAL_CANDIDATE}"
)

print(
    f"Multipliers  : {REINFORCEMENT_MULTIPLIERS}"
)

print(
    f"Q levels     : {Q_SUPPORT_LEVELS}"
)


# =============================================================================
# TEST LOOP
# =============================================================================

results = []

print("\n" + "=" * 110)
print("TESTING COMBINED REINFORCEMENT + LOCAL REACTIVE SUPPORT")
print("=" * 110)


for multiplier in REINFORCEMENT_MULTIPLIERS:

    for q_support in Q_SUPPORT_LEVELS:

        print("\n" + "-" * 110)

        print(
            f"TEST — {THERMAL_CANDIDATE} — "
            f"{multiplier:.2f}x — "
            f"+{q_support} MVAr"
        )

        print("-" * 110)

        # ---------------------------------------------------------------------
        # Fresh network for every case
        # ---------------------------------------------------------------------

        n = load_network()

        # ---------------------------------------------------------------------
        # Critical operating point
        # ---------------------------------------------------------------------

        apply_lambda(n)

        # ---------------------------------------------------------------------
        # Reinforce thermal candidate
        # ---------------------------------------------------------------------

        original_s_nom = float(
            n.lines.at[
                THERMAL_CANDIDATE,
                "s_nom"
            ]
        )

        new_s_nom = (
            original_s_nom
            * multiplier
        )

        n.lines.at[
            THERMAL_CANDIDATE,
            "s_nom"
        ] = new_s_nom

        print(
            f"Original s_nom : "
            f"{original_s_nom:.6f} MW"
        )

        print(
            f"New s_nom      : "
            f"{new_s_nom:.6f} MW"
        )

        # ---------------------------------------------------------------------
        # Time-series values aligned exactly to network snapshots
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Add local reactive support
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Verify time-series alignment
        # ---------------------------------------------------------------------

        assert n.generators_t.p_set.index.equals(
            n.snapshots
        )

        assert n.generators_t.q_set.index.equals(
            n.snapshots
        )

        # ---------------------------------------------------------------------
        # AC POWER FLOW
        # ---------------------------------------------------------------------

        pf_result = run_pf(n)

        # ---------------------------------------------------------------------
        # Extract results
        # ---------------------------------------------------------------------

        result = extract_results(
            n=n,
            multiplier=multiplier,
            q_support=q_support,
            original_s_nom=original_s_nom,
            pf_result=pf_result,
        )

        results.append(result)

        # ---------------------------------------------------------------------
        # Display
        # ---------------------------------------------------------------------

        print("\nRESULT")
        print("-" * 100)

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
            f"Critical/max loaded line: "
            f"{result['max_loaded_line']}"
        )

        print(
            f"Reinforced line loading : "
            f"{result['reinforced_line_loading_pct']:.6f} %"
        )

        print(
            f"Max transformer loading : "
            f"{result['max_transformer_loading_pct']:.6f} %"
        )

        print(
            f"Fully acceptable        : "
            f"{result['fully_acceptable']}"
        )


# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    results
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 110)
print("S3.9 SUMMARY")
print("=" * 110)


if len(results_df) == 0:

    print(
        "\nNO RESULTS GENERATED."
    )

else:

    display_columns = [

        "reinforcement_multiplier",

        "q_support_mvar",

        "converged",

        "min_voltage_pu",

        "weak_bus_voltage_pu",

        "max_line_loading_pct",

        "overloaded_lines",

        "max_loaded_line",

        "reinforced_line_loading_pct",

        "max_transformer_loading_pct",

        "weak_voltage_ok",

        "minimum_voltage_ok",

        "thermal_loading_ok",

        "overload_count_ok",

        "fully_acceptable",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # FULLY ACCEPTABLE CASES
    # -------------------------------------------------------------------------

    acceptable = results_df[
        results_df[
            "fully_acceptable"
        ] == True
    ].copy()

    print("\n" + "-" * 110)
    print("FULLY ACCEPTABLE CASES")
    print("-" * 110)

    if len(acceptable) == 0:

        print(
            "NO CASE SATISFIES ALL FIVE ACCEPTANCE CRITERIA."
        )

    else:

        print(
            acceptable[
                [
                    "reinforcement_multiplier",
                    "q_support_mvar",
                    "min_voltage_pu",
                    "weak_bus_voltage_pu",
                    "max_line_loading_pct",
                    "overloaded_lines",
                    "max_loaded_line",
                    "reinforced_line_loading_pct",
                ]
            ].to_string(
                index=False
            )
        )

        # -------------------------------------------------------------
        # Minimum practical combination:
        # lowest reinforcement first, then lowest Q
        # -------------------------------------------------------------

        acceptable_sorted = acceptable.sort_values(
            [
                "reinforcement_multiplier",
                "q_support_mvar",
            ]
        )

        best = acceptable_sorted.iloc[0]

        print("\nMINIMUM TESTED FULLY ACCEPTABLE COMBINATION")
        print("-" * 110)

        print(
            f"Reinforcement : "
            f"{best['reinforcement_multiplier']:.2f}x"
        )

        print(
            f"Q support     : "
            f"+{best['q_support_mvar']:.0f} MVAr"
        )

        print(
            f"Minimum V     : "
            f"{best['min_voltage_pu']:.6f} pu"
        )

        print(
            f"Weak bus V    : "
            f"{best['weak_bus_voltage_pu']:.6f} pu"
        )

        print(
            f"Max loading   : "
            f"{best['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloads     : "
            f"{best['overloaded_lines']}"
        )


    # -------------------------------------------------------------------------
    # BEST VOLTAGE CASE
    # -------------------------------------------------------------------------

    valid_voltage = results_df[
        results_df["converged"] == True
    ].copy()

    if len(valid_voltage) > 0:

        best_voltage = valid_voltage.loc[
            valid_voltage[
                "weak_bus_voltage_pu"
            ].idxmax()
        ]

        print("\nBEST WEAK-BUS VOLTAGE")
        print("-" * 110)

        print(
            f"Reinforcement : "
            f"{best_voltage['reinforcement_multiplier']:.2f}x"
        )

        print(
            f"Q support     : "
            f"+{best_voltage['q_support_mvar']:.0f} MVAr"
        )

        print(
            f"Weak bus V    : "
            f"{best_voltage['weak_bus_voltage_pu']:.6f} pu"
        )

        print(
            f"Minimum V     : "
            f"{best_voltage['min_voltage_pu']:.6f} pu"
        )


        # ---------------------------------------------------------------------
        # BEST THERMAL CASE
        # ---------------------------------------------------------------------

        best_thermal = valid_voltage.loc[
            valid_voltage[
                "max_line_loading_pct"
            ].idxmin()
        ]

        print("\nBEST THERMAL CASE")
        print("-" * 110)

        print(
            f"Reinforcement : "
            f"{best_thermal['reinforcement_multiplier']:.2f}x"
        )

        print(
            f"Q support     : "
            f"+{best_thermal['q_support_mvar']:.0f} MVAr"
        )

        print(
            f"Max loading   : "
            f"{best_thermal['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloads     : "
            f"{best_thermal['overloaded_lines']}"
        )

        print(
            f"Critical line : "
            f"{best_thermal['max_loaded_line']}"
        )


# =============================================================================
# SAVE RESULTS
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
print("S3.9 COMPLETE")
print("=" * 110)

print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 110)