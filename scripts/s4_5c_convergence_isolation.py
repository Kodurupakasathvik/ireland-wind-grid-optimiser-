# ==================================================================================================
#
# S4.5C — AC POWER-FLOW CONVERGENCE ISOLATION
#
# Purpose
# -------
# Determine exactly which network configuration causes AC nonlinear
# power-flow divergence at S2_PEAK_DEMAND.
#
# Cases:
#
#   A — ORIGINAL BASELINE
#   B — P3 ONLY
#   C — ALL FOUR RESIDUAL LINES 1.25x ONLY
#   D — P3 + ALL FOUR RESIDUAL LINES 1.25x
#
# No reactive support is added.
# Dispatch is unchanged.
# Loads are unchanged.
# Source network is READ-ONLY.
#
# ==================================================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


warnings.filterwarnings("ignore")


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT = Path(
    "data/processed/s4_5c_convergence_isolation.csv"
)

P3_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.75,
    "merged_way/61295764-220+1": 2.00,
    "way/343436171-220": 2.00,
    "merged_way/257889771-220+1": 1.75,
    "merged_relation/4872159-220+1": 1.75,
}

RESIDUAL_LINES = [
    "way/235559472-220",
    "way/713396116-220",
    "way/42838773-220",
    "merged_way/516651706-220+2",
]

RESIDUAL_MULTIPLIER = 1.25


# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def apply_reinforcements(
    network,
    apply_p3=False,
    apply_residual=False,
):

    if apply_p3:

        for line, multiplier in P3_REINFORCEMENTS.items():

            if line not in network.lines.index:
                raise KeyError(
                    f"P3 line not found: {line}"
                )

            network.lines.at[
                line,
                "s_nom",
            ] *= multiplier

    if apply_residual:

        for line in RESIDUAL_LINES:

            if line not in network.lines.index:
                raise KeyError(
                    f"Residual line not found: {line}"
                )

            network.lines.at[
                line,
                "s_nom",
            ] *= RESIDUAL_MULTIPLIER


def get_generator_data(network):

    rows = []

    for generator in network.generators.index:

        try:
            p_set = float(
                network.generators_t.p_set.at[
                    SNAPSHOT,
                    generator,
                ]
            )
        except Exception:
            p_set = np.nan

        try:
            q_set = float(
                network.generators_t.q_set.at[
                    SNAPSHOT,
                    generator,
                ]
            )
        except Exception:
            q_set = np.nan

        rows.append(
            {
                "generator": generator,
                "bus": network.generators.at[
                    generator,
                    "bus",
                ],
                "control": network.generators.at[
                    generator,
                    "control",
                ],
                "p_set_mw": p_set,
                "q_set_mvar": q_set,
            }
        )

    return pd.DataFrame(rows)


def calculate_voltage_result(network):

    try:

        v = (
            network.buses_t.v_mag_pu.loc[
                SNAPSHOT
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(v) == 0:
            return {
                "voltage_available": False,
                "min_voltage_pu": np.nan,
                "max_voltage_pu": np.nan,
                "min_voltage_bus": "",
                "max_voltage_bus": "",
            }

        values = v.to_numpy(
            dtype=float
        )

        finite = np.all(
            np.isfinite(values)
        )

        if not finite:
            return {
                "voltage_available": False,
                "min_voltage_pu": np.nan,
                "max_voltage_pu": np.nan,
                "min_voltage_bus": "",
                "max_voltage_bus": "",
            }

        return {
            "voltage_available": True,
            "min_voltage_pu": float(v.min()),
            "max_voltage_pu": float(v.max()),
            "min_voltage_bus": str(v.idxmin()),
            "max_voltage_bus": str(v.idxmax()),
        }

    except Exception:

        return {
            "voltage_available": False,
            "min_voltage_pu": np.nan,
            "max_voltage_pu": np.nan,
            "min_voltage_bus": "",
            "max_voltage_bus": "",
        }


def calculate_line_result(network):

    try:

        loading = (
            network.lines_t.loading_percent.loc[
                SNAPSHOT
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(loading) == 0:

            return {
                "line_loading_available": False,
                "max_line_loading_pct": np.nan,
                "overloaded_lines": np.nan,
            }

        values = loading.to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(values)
        ):

            return {
                "line_loading_available": False,
                "max_line_loading_pct": np.nan,
                "overloaded_lines": np.nan,
            }

        return {
            "line_loading_available": True,
            "max_line_loading_pct": float(
                loading.max()
            ),
            "overloaded_lines": int(
                (loading > 100.0).sum()
            ),
        }

    except Exception:

        return {
            "line_loading_available": False,
            "max_line_loading_pct": np.nan,
            "overloaded_lines": np.nan,
        }


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.5C — AC POWER-FLOW CONVERGENCE ISOLATION")
print("=" * 100)

print()
print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print("PF       : AC nonlinear")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Reactive : unchanged")
print("Source   : READ-ONLY")

print()
print("Cases:")
print("  A — ORIGINAL BASELINE")
print("  B — P3 ONLY")
print("  C — ALL FOUR RESIDUAL LINES 1.25x ONLY")
print("  D — P3 + ALL FOUR RESIDUAL LINES 1.25x")


# ==================================================================================================
# CASE DEFINITIONS
# ==================================================================================================

CASES = [

    {
        "case": "A",
        "name": "ORIGINAL_BASELINE",
        "p3": False,
        "residual": False,
    },

    {
        "case": "B",
        "name": "P3_ONLY",
        "p3": True,
        "residual": False,
    },

    {
        "case": "C",
        "name": "ALL_FOUR_1.25X_ONLY",
        "p3": False,
        "residual": True,
    },

    {
        "case": "D",
        "name": "P3_PLUS_ALL_FOUR_1.25X",
        "p3": True,
        "residual": True,
    },

]


# ==================================================================================================
# RESULTS
# ==================================================================================================

results = []


# ==================================================================================================
# RUN CASES
# ==================================================================================================

for case in CASES:

    print()
    print("=" * 100)
    print(
        f"CASE {case['case']} — {case['name']}"
    )
    print("=" * 100)

    print()
    print(
        f"P3 reinforcements      : "
        f"{case['p3']}"
    )

    print(
        f"Residual reinforcements: "
        f"{case['residual']}"
    )

    # ----------------------------------------------------------------------------------------------
    # LOAD FRESH NETWORK
    # ----------------------------------------------------------------------------------------------

    print()
    print("Loading fresh source network...")

    network = pypsa.Network(
        str(NETWORK_PATH)
    )

    if SNAPSHOT not in network.snapshots:
        raise ValueError(
            f"Snapshot '{SNAPSHOT}' not found."
        )

    # ----------------------------------------------------------------------------------------------
    # ISOLATE SNAPSHOT
    # ----------------------------------------------------------------------------------------------

    network.set_snapshots(
        [SNAPSHOT]
    )

    if (
        len(network.snapshots) != 1
        or network.snapshots[0] != SNAPSHOT
    ):
        raise RuntimeError(
            "Snapshot isolation failed."
        )

    # ----------------------------------------------------------------------------------------------
    # GENERATOR DATA BEFORE PF
    # ----------------------------------------------------------------------------------------------

    generator_df = get_generator_data(
        network
    )

    total_generation = float(
        generator_df[
            "p_set_mw"
        ]
        .sum()
    )

    try:

        total_load = float(
            network.loads_t.p_set.loc[
                SNAPSHOT
            ]
            .sum()
        )

    except Exception:

        total_load = np.nan

    # ----------------------------------------------------------------------------------------------
    # APPLY REINFORCEMENTS
    # ----------------------------------------------------------------------------------------------

    apply_reinforcements(
        network,
        apply_p3=case["p3"],
        apply_residual=case["residual"],
    )

    # ----------------------------------------------------------------------------------------------
    # RECORD TARGET LINE RATINGS
    # ----------------------------------------------------------------------------------------------

    target_ratings = {}

    for line in (
        list(P3_REINFORCEMENTS.keys())
        + RESIDUAL_LINES
    ):

        if line in network.lines.index:

            target_ratings[line] = float(
                network.lines.at[
                    line,
                    "s_nom",
                ]
            )

    # ----------------------------------------------------------------------------------------------
    # Q SET DIAGNOSTIC
    # ----------------------------------------------------------------------------------------------

    q_values = generator_df[
        "q_set_mvar"
    ].to_numpy(
        dtype=float
    )

    q_nan_count = int(
        np.isnan(q_values).sum()
    )

    # ----------------------------------------------------------------------------------------------
    # RUN AC PF
    # ----------------------------------------------------------------------------------------------

    print()
    print(
        "Running AC nonlinear power flow..."
    )

    pf_exception = None
    pf_result = None

    try:

        pf_result = network.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True,
        )

        print(
            "PF returned."
        )

    except Exception as exc:

        pf_exception = repr(exc)

        print(
            "PF raised exception:"
        )

        print(
            pf_exception
        )

    # ----------------------------------------------------------------------------------------------
    # DETERMINE CONVERGENCE
    # ----------------------------------------------------------------------------------------------

    converged = False

    if pf_result is not None:

        try:

            if isinstance(
                pf_result,
                tuple
            ):

                convergence_data = pf_result[0]

            else:

                convergence_data = pf_result

            if isinstance(
                convergence_data,
                pd.DataFrame
            ):

                if SNAPSHOT in convergence_data.index:

                    converged = bool(
                        convergence_data.loc[
                            SNAPSHOT
                        ].all()
                    )

                elif SNAPSHOT in convergence_data.columns:

                    converged = bool(
                        convergence_data[
                            SNAPSHOT
                        ].all()
                    )

            elif np.isscalar(
                convergence_data
            ):

                converged = bool(
                    convergence_data
                )

        except Exception:

            converged = False

    # ----------------------------------------------------------------------------------------------
    # VOLTAGE VALIDITY
    # ----------------------------------------------------------------------------------------------

    voltage_result = (
        calculate_voltage_result(
            network
        )
    )

    # ----------------------------------------------------------------------------------------------
    # LINE VALIDITY
    # ----------------------------------------------------------------------------------------------

    line_result = (
        calculate_line_result(
            network
        )
    )

    # ----------------------------------------------------------------------------------------------
    # PHYSICAL SOLUTION CHECK
    # ----------------------------------------------------------------------------------------------

    valid_physical_solution = False

    if (
        converged
        and voltage_result[
            "voltage_available"
        ]
    ):

        min_v = voltage_result[
            "min_voltage_pu"
        ]

        max_v = voltage_result[
            "max_voltage_pu"
        ]

        if (
            np.isfinite(min_v)
            and np.isfinite(max_v)
            and min_v > 0.0
            and max_v < 2.0
        ):

            valid_physical_solution = True

    # ----------------------------------------------------------------------------------------------
    # PRINT RESULT
    # ----------------------------------------------------------------------------------------------

    print()
    print(
        f"Converged                    : "
        f"{converged}"
    )

    print(
        f"Valid physical solution      : "
        f"{valid_physical_solution}"
    )

    if voltage_result[
        "voltage_available"
    ]:

        print(
            f"Minimum voltage              : "
            f"{voltage_result['min_voltage_pu']:.6f} pu"
        )

        print(
            f"Maximum voltage              : "
            f"{voltage_result['max_voltage_pu']:.6f} pu"
        )

    else:

        print(
            "Voltage result               : INVALID / UNAVAILABLE"
        )

    if line_result[
        "line_loading_available"
    ]:

        print(
            f"Maximum line loading         : "
            f"{line_result['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded lines             : "
            f"{line_result['overloaded_lines']}"
        )

    else:

        print(
            "Line loading result          : INVALID / UNAVAILABLE"
        )

    print(
        f"Generator Q-set NaNs         : "
        f"{q_nan_count}"
    )

    print(
        f"Generation                   : "
        f"{total_generation:.6f} MW"
    )

    print(
        f"Load                         : "
        f"{total_load:.6f} MW"
    )

    print(
        f"Generation - Load            : "
        f"{total_generation - total_load:.6f} MW"
    )

    # ----------------------------------------------------------------------------------------------
    # SAVE CASE
    # ----------------------------------------------------------------------------------------------

    results.append(
        {
            "case": case["case"],
            "configuration": case["name"],
            "p3_applied": case["p3"],
            "residual_1_25x_applied": case["residual"],
            "snapshot": SNAPSHOT,
            "converged": converged,
            "valid_physical_solution": valid_physical_solution,
            "min_voltage_pu": (
                voltage_result[
                    "min_voltage_pu"
                ]
            ),
            "max_voltage_pu": (
                voltage_result[
                    "max_voltage_pu"
                ]
            ),
            "min_voltage_bus": (
                voltage_result[
                    "min_voltage_bus"
                ]
            ),
            "max_voltage_bus": (
                voltage_result[
                    "max_voltage_bus"
                ]
            ),
            "max_line_loading_pct": (
                line_result[
                    "max_line_loading_pct"
                ]
            ),
            "overloaded_lines": (
                line_result[
                    "overloaded_lines"
                ]
            ),
            "generator_q_nan_count": q_nan_count,
            "total_generation_mw": total_generation,
            "total_load_mw": total_load,
            "generation_minus_load_mw": (
                total_generation
                - total_load
            ),
            "pf_exception": pf_exception,
        }
    )


# ==================================================================================================
# SUMMARY
# ==================================================================================================

summary_df = pd.DataFrame(
    results
)

print()
print("=" * 100)
print("S4.5C — CONVERGENCE ISOLATION SUMMARY")
print("=" * 100)

print()

print(
    summary_df[
        [
            "case",
            "configuration",
            "converged",
            "valid_physical_solution",
            "min_voltage_pu",
            "max_voltage_pu",
            "max_line_loading_pct",
            "overloaded_lines",
            "generator_q_nan_count",
        ]
    ].to_string(
        index=False
    )
)


# ==================================================================================================
# INTERPRETATION
# ==================================================================================================

print()
print("=" * 100)
print("S4.5C INTERPRETATION")
print("=" * 100)

case_a = summary_df[
    summary_df["case"] == "A"
]

case_b = summary_df[
    summary_df["case"] == "B"
]

case_c = summary_df[
    summary_df["case"] == "C"
]

case_d = summary_df[
    summary_df["case"] == "D"
]


def get_bool(df, column):

    if len(df) == 0:
        return False

    value = df.iloc[0][column]

    if pd.isna(value):
        return False

    return bool(value)


a_ok = get_bool(
    case_a,
    "converged"
)

b_ok = get_bool(
    case_b,
    "converged"
)

c_ok = get_bool(
    case_c,
    "converged"
)

d_ok = get_bool(
    case_d,
    "converged"
)


print()

print(
    f"Case A — Original baseline              : "
    f"{a_ok}"
)

print(
    f"Case B — P3 only                         : "
    f"{b_ok}"
)

print(
    f"Case C — Residual 1.25x only             : "
    f"{c_ok}"
)

print(
    f"Case D — P3 + residual 1.25x             : "
    f"{d_ok}"
)

print()

if not a_ok:

    print(
        "FINDING:"
    )

    print(
        "The ORIGINAL BASELINE itself does not converge."
    )

    print(
        "Therefore the problem is upstream of S4.5."
    )

elif a_ok and not b_ok:

    print(
        "FINDING:"
    )

    print(
        "P3 reinforcement is associated with loss of convergence."
    )

elif a_ok and not c_ok:

    print(
        "FINDING:"
    )

    print(
        "The ALL-FOUR 1.25x reinforcement configuration "
        "is associated with loss of convergence."
    )

elif a_ok and b_ok and c_ok and not d_ok:

    print(
        "FINDING:"
    )

    print(
        "The combined P3 + ALL-FOUR configuration causes "
        "the convergence failure."
    )

elif a_ok and b_ok and c_ok and d_ok:

    print(
        "FINDING:"
    )

    print(
        "All four configurations converge."
    )

    print(
        "The previous S4.5 divergence is therefore "
        "not reproduced by this controlled isolation test."
    )

else:

    print(
        "FINDING:"
    )

    print(
        "Convergence pattern requires further inspection."
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

print()
print("=" * 100)
print("SAVING RESULTS")
print("=" * 100)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary_df.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    f"Results saved to:"
)
print(
    f"  {OUTPUT}"
)

print()
print("=" * 100)
print("S4.5C COMPLETE")
print("=" * 100)

print()
print("IMPORTANT:")
print("No network file was modified.")
print("No mitigation was applied.")
print("No invalid PF values were interpreted.")
print("=" * 100)