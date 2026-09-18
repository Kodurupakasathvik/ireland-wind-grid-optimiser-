"""
====================================================================================================
S4.5D — BASELINE AC POWER-FLOW REPAIR ISOLATION
====================================================================================================

Purpose
-------
Determine why the ORIGINAL baseline AC nonlinear power flow does not converge.

This is a READ-ONLY diagnostic.

The source network is NEVER modified.

Tests:
    A — Raw source PF
    B — Replace NaN generator q_set with 0
    C — Replace NaN load q_set with 0
    D — Replace both generator/load q_set NaNs with 0
    E — D + explicit Slack generator
    F — D + explicit Slack + distributed slack
    G — D + explicit Slack + flat voltage initialization

Target:
    S2_PEAK_DEMAND

No line reinforcement.
No P3 reinforcement.
No reactive compensation.
No network file modification.

====================================================================================================
"""

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

SOURCE = Path(
    r"data\processed\eirgrid_second_reinforced_network.nc"
)

OUTPUT = Path(
    r"data\processed\s4_5d_baseline_pf_repair_isolation.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

V_MIN_VALID = 0.50
V_MAX_VALID = 1.50

PF_X_TOL = 1e-6


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.5D — BASELINE AC POWER-FLOW REPAIR ISOLATION")
print("=" * 100)

print()
print(f"Network  : {SOURCE}")
print(f"Snapshot : {SNAPSHOT}")
print("PF       : AC nonlinear")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Network  : READ-ONLY")
print()

print("=" * 100)
print("PURPOSE")
print("=" * 100)

print(
    """
The original baseline AC power flow failed before any reinforcement was applied.

This stage isolates possible formulation/data problems:

    1. NaN generator reactive setpoints
    2. NaN load reactive setpoints
    3. Missing explicit slack generator
    4. Slack balancing requirement
    5. Initial voltage conditions

No reinforcement is applied.
No reactive device is added.
No source network file is modified.
"""
)


# ==================================================================================================
# BASIC HELPERS
# ==================================================================================================

def is_numeric_finite(value):
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def get_snapshot_value(table, column, snapshot):
    """
    Safely retrieve a static or time-varying PyPSA value.
    """
    if column not in table.columns:
        return None

    value = table[column]

    if isinstance(value, pd.Series):
        return value

    if snapshot in value.index if isinstance(value.index, pd.Index) else False:
        return value.loc[snapshot]

    return value


def finite_voltage_stats(n, snapshot):
    """
    Return voltage statistics only if values are numerically finite.
    """
    if snapshot not in n.buses_t.v_mag_pu.columns:
        return {
            "valid_voltage": False,
            "min_voltage": np.nan,
            "max_voltage": np.nan,
        }

    v = pd.to_numeric(
        n.buses_t.v_mag_pu.loc[snapshot],
        errors="coerce"
    )

    finite = v[np.isfinite(v)]

    if len(finite) == 0:
        return {
            "valid_voltage": False,
            "min_voltage": np.nan,
            "max_voltage": np.nan,
        }

    vmin = float(finite.min())
    vmax = float(finite.max())

    valid = (
        vmin >= V_MIN_VALID
        and vmax <= V_MAX_VALID
    )

    return {
        "valid_voltage": bool(valid),
        "min_voltage": vmin,
        "max_voltage": vmax,
    }


def count_nan_column(table, column):
    if column not in table.columns:
        return 0

    return int(
        pd.to_numeric(
            table[column],
            errors="coerce"
        ).isna().sum()
    )


def choose_slack_generator(n, snapshot):
    """
    Select the largest positive-dispatch generator.

    We do NOT change its active-power setpoint.

    The only change is its PF control mode to Slack.

    This is appropriate for a diagnostic because PyPSA's
    AC PF needs one slack bus to absorb active-power mismatch.
    """

    if len(n.generators) == 0:
        return None

    dispatch = pd.Series(
        0.0,
        index=n.generators.index,
        dtype=float
    )

    if snapshot in n.generators_t.p_set.columns:
        values = pd.to_numeric(
            n.generators_t.p_set.loc[snapshot],
            errors="coerce"
        ).fillna(0.0)

        dispatch.loc[values.index] = values

    positive = dispatch[dispatch > 0]

    if len(positive) == 0:
        return None

    return positive.idxmax()


def calculate_power_balance(n, snapshot):
    """
    Calculate P-setpoint balance before PF.
    """

    gen = 0.0
    load = 0.0

    if snapshot in n.generators_t.p_set.columns:
        gen = float(
            pd.to_numeric(
                n.generators_t.p_set.loc[snapshot],
                errors="coerce"
            ).fillna(0.0).sum()
        )

    if snapshot in n.loads_t.p_set.columns:
        load = float(
            pd.to_numeric(
                n.loads_t.p_set.loc[snapshot],
                errors="coerce"
            ).fillna(0.0).sum()
        )

    return gen, load, gen - load


# ==================================================================================================
# LOAD SOURCE
# ==================================================================================================

print("=" * 100)
print("LOADING SOURCE NETWORK")
print("=" * 100)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Source network not found: {SOURCE}"
    )

source = pypsa.Network(SOURCE)

print(f"Buses        : {len(source.buses)}")
print(f"Lines        : {len(source.lines)}")
print(f"Transformers : {len(source.transformers)}")
print(f"Generators   : {len(source.generators)}")
print(f"Loads        : {len(source.loads)}")

print()

if SNAPSHOT not in source.snapshots:
    raise ValueError(
        f"Snapshot '{SNAPSHOT}' not found."
    )


# ==================================================================================================
# SOURCE SANITY REPORT
# ==================================================================================================

print("=" * 100)
print("SOURCE PF INPUT SANITY")
print("=" * 100)

gen_q_nan = count_nan_column(
    source.generators,
    "q_set"
)

load_q_nan = count_nan_column(
    source.loads,
    "q_set"
)

gen_control_counts = (
    source.generators["control"]
    .value_counts(dropna=False)
    .to_dict()
)

print()
print(f"Generator q_set NaNs : {gen_q_nan}")
print(f"Load q_set NaNs      : {load_q_nan}")
print(f"Generator controls   : {gen_control_counts}")

if "v_mag_pu_set" in source.buses.columns:
    bus_v_nan = count_nan_column(
        source.buses,
        "v_mag_pu_set"
    )
else:
    bus_v_nan = 0

print(f"Bus v_mag_pu_set NaNs: {bus_v_nan}")

print()

gen_p, load_p, balance = calculate_power_balance(
    source,
    SNAPSHOT
)

print(f"Generator P set : {gen_p:.6f} MW")
print(f"Load P set      : {load_p:.6f} MW")
print(f"Generation-load : {balance:.6f} MW")

print()


# ==================================================================================================
# CASE DEFINITIONS
# ==================================================================================================

cases = [
    {
        "name": "A_RAW_SOURCE",
        "gen_q_zero": False,
        "load_q_zero": False,
        "explicit_slack": False,
        "distributed_slack": False,
        "flat_voltage": False,
    },
    {
        "name": "B_GENERATOR_Q_ZERO",
        "gen_q_zero": True,
        "load_q_zero": False,
        "explicit_slack": False,
        "distributed_slack": False,
        "flat_voltage": False,
    },
    {
        "name": "C_LOAD_Q_ZERO",
        "gen_q_zero": False,
        "load_q_zero": True,
        "explicit_slack": False,
        "distributed_slack": False,
        "flat_voltage": False,
    },
    {
        "name": "D_ALL_Q_ZERO",
        "gen_q_zero": True,
        "load_q_zero": True,
        "explicit_slack": False,
        "distributed_slack": False,
        "flat_voltage": False,
    },
    {
        "name": "E_Q_ZERO_EXPLICIT_SLACK",
        "gen_q_zero": True,
        "load_q_zero": True,
        "explicit_slack": True,
        "distributed_slack": False,
        "flat_voltage": False,
    },
    {
        "name": "F_Q_ZERO_SLACK_DISTRIBUTED",
        "gen_q_zero": True,
        "load_q_zero": True,
        "explicit_slack": True,
        "distributed_slack": True,
        "flat_voltage": False,
    },
    {
        "name": "G_Q_ZERO_SLACK_FLAT_VOLTAGE",
        "gen_q_zero": True,
        "load_q_zero": True,
        "explicit_slack": True,
        "distributed_slack": False,
        "flat_voltage": True,
    },
]


# ==================================================================================================
# CASE EXECUTION
# ==================================================================================================

results = []


for case in cases:

    name = case["name"]

    print()
    print("=" * 100)
    print(f"CASE {name}")
    print("=" * 100)

    print()
    print(f"Generator q_set -> 0 : {case['gen_q_zero']}")
    print(f"Load q_set      -> 0 : {case['load_q_zero']}")
    print(f"Explicit slack       : {case['explicit_slack']}")
    print(f"Distributed slack    : {case['distributed_slack']}")
    print(f"Flat voltage         : {case['flat_voltage']}")
    print()

    # ----------------------------------------------------------------------------------------------
    # FRESH COPY
    # ----------------------------------------------------------------------------------------------

    n = copy.deepcopy(source)

    # ----------------------------------------------------------------------------------------------
    # SNAPSHOT ISOLATION
    # ----------------------------------------------------------------------------------------------

    n.set_snapshots([SNAPSHOT])

    # ----------------------------------------------------------------------------------------------
    # REPAIR GENERATOR q_set
    # ----------------------------------------------------------------------------------------------

    if case["gen_q_zero"]:

        if "q_set" not in n.generators.columns:
            n.generators["q_set"] = 0.0

        n.generators["q_set"] = (
            pd.to_numeric(
                n.generators["q_set"],
                errors="coerce"
            )
            .fillna(0.0)
        )

        if SNAPSHOT in n.generators_t.q_set.columns:

            q_series = pd.to_numeric(
                n.generators_t.q_set.loc[SNAPSHOT],
                errors="coerce"
            ).fillna(0.0)

            n.generators_t.q_set.loc[
                SNAPSHOT,
                q_series.index
            ] = q_series.values

    # ----------------------------------------------------------------------------------------------
    # REPAIR LOAD q_set
    # ----------------------------------------------------------------------------------------------

    if case["load_q_zero"]:

        if "q_set" not in n.loads.columns:
            n.loads["q_set"] = 0.0

        n.loads["q_set"] = (
            pd.to_numeric(
                n.loads["q_set"],
                errors="coerce"
            )
            .fillna(0.0)
        )

        if SNAPSHOT in n.loads_t.q_set.columns:

            q_series = pd.to_numeric(
                n.loads_t.q_set.loc[SNAPSHOT],
                errors="coerce"
            ).fillna(0.0)

            n.loads_t.q_set.loc[
                SNAPSHOT,
                q_series.index
            ] = q_series.values

    # ----------------------------------------------------------------------------------------------
    # EXPLICIT SLACK
    # ----------------------------------------------------------------------------------------------

    slack_generator = None

    if case["explicit_slack"]:

        slack_generator = choose_slack_generator(
            n,
            SNAPSHOT
        )

        if slack_generator is None:

            print(
                "WARNING: No positive-dispatch generator available "
                "for explicit slack assignment."
            )

        else:

            print(
                f"Selected slack generator : {slack_generator}"
            )

            print(
                f"Slack bus                : "
                f"{n.generators.at[slack_generator, 'bus']}"
            )

            # Reset all generators to PQ first.
            n.generators["control"] = "PQ"

            # Assign exactly one explicit slack.
            n.generators.at[
                slack_generator,
                "control"
            ] = "Slack"

            # Ensure slack q_set is finite.
            if (
                not is_numeric_finite(
                    n.generators.at[
                        slack_generator,
                        "q_set"
                    ]
                )
            ):
                n.generators.at[
                    slack_generator,
                    "q_set"
                ] = 0.0

    # ----------------------------------------------------------------------------------------------
    # FLAT VOLTAGE INITIALIZATION
    # ----------------------------------------------------------------------------------------------

    if case["flat_voltage"]:

        if "v_mag_pu_set" in n.buses.columns:

            n.buses["v_mag_pu_set"] = (
                pd.to_numeric(
                    n.buses["v_mag_pu_set"],
                    errors="coerce"
                )
                .fillna(1.0)
            )

            n.buses["v_mag_pu_set"] = 1.0

        # Explicitly initialize the PF voltage state.
        n.buses_t.v_mag_pu.loc[
            SNAPSHOT,
            :
        ] = 1.0

        n.buses_t.v_ang.loc[
            SNAPSHOT,
            :
        ] = 0.0

    # ----------------------------------------------------------------------------------------------
    # RE-CHECK q SETPOINTS
    # ----------------------------------------------------------------------------------------------

    final_gen_q_nan = 0
    final_load_q_nan = 0

    if "q_set" in n.generators.columns:
        final_gen_q_nan = int(
            pd.to_numeric(
                n.generators["q_set"],
                errors="coerce"
            ).isna().sum()
        )

    if "q_set" in n.loads.columns:
        final_load_q_nan = int(
            pd.to_numeric(
                n.loads["q_set"],
                errors="coerce"
            ).isna().sum()
        )

    print(
        f"Final generator q_set NaNs : {final_gen_q_nan}"
    )

    print(
        f"Final load q_set NaNs      : {final_load_q_nan}"
    )

    # ----------------------------------------------------------------------------------------------
    # CONTROL STRUCTURE
    # ----------------------------------------------------------------------------------------------

    print()
    print("Generator controls:")

    print(
        n.generators[
            ["bus", "control", "p_set", "q_set"]
        ].to_string()
    )

    print()

    # ----------------------------------------------------------------------------------------------
    # POWER BALANCE
    # ----------------------------------------------------------------------------------------------

    gen_p_case, load_p_case, balance_case = (
        calculate_power_balance(
            n,
            SNAPSHOT
        )
    )

    print()
    print(f"Generation : {gen_p_case:.6f} MW")
    print(f"Load       : {load_p_case:.6f} MW")
    print(f"Mismatch   : {balance_case:.6f} MW")

    # ----------------------------------------------------------------------------------------------
    # POWER FLOW
    # ----------------------------------------------------------------------------------------------

    print()
    print("Running AC nonlinear power flow...")

    converged = False
    pf_error = np.nan
    iterations = np.nan

    try:

        pf_result = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=PF_X_TOL,
            distribute_slack=case["distributed_slack"],
        )

        # ------------------------------------------------------------------
        # Extract convergence information
        # ------------------------------------------------------------------

        if isinstance(pf_result, dict):

            if "converged" in pf_result:

                conv = pf_result["converged"]

                try:
                    converged = bool(
                        conv.loc[SNAPSHOT].iloc[0]
                        if isinstance(
                            conv.loc[SNAPSHOT],
                            pd.Series
                        )
                        else conv.loc[SNAPSHOT]
                    )
                except Exception:

                    try:
                        converged = bool(
                            np.asarray(conv).flatten()[0]
                        )
                    except Exception:
                        converged = False

            if "error" in pf_result:

                try:
                    error_value = pf_result["error"]

                    if isinstance(error_value, pd.DataFrame):
                        pf_error = float(
                            error_value.loc[SNAPSHOT].iloc[0]
                        )

                    elif isinstance(error_value, pd.Series):
                        pf_error = float(
                            error_value.loc[SNAPSHOT]
                        )

                    else:
                        pf_error = float(
                            np.asarray(
                                error_value
                            ).flatten()[0]
                        )

                except Exception:
                    pf_error = np.nan

            if "n_iter" in pf_result:

                try:
                    iter_value = pf_result["n_iter"]

                    if isinstance(iter_value, pd.DataFrame):
                        iterations = float(
                            iter_value.loc[SNAPSHOT].iloc[0]
                        )

                    elif isinstance(iter_value, pd.Series):
                        iterations = float(
                            iter_value.loc[SNAPSHOT]
                        )

                    else:
                        iterations = float(
                            np.asarray(
                                iter_value
                            ).flatten()[0]
                        )

                except Exception:
                    iterations = np.nan

        print("PF returned.")

    except Exception as exc:

        print()
        print("PF EXCEPTION:")
        print(type(exc).__name__)
        print(str(exc))

        pf_error = np.nan

    # ----------------------------------------------------------------------------------------------
    # VOLTAGE VALIDATION
    # ----------------------------------------------------------------------------------------------

    voltage_stats = finite_voltage_stats(
        n,
        SNAPSHOT
    )

    valid_voltage = voltage_stats["valid_voltage"]

    vmin = voltage_stats["min_voltage"]
    vmax = voltage_stats["max_voltage"]

    # ----------------------------------------------------------------------------------------------
    # FINAL PHYSICAL VALIDITY
    # ----------------------------------------------------------------------------------------------

    valid_physical = (
        bool(converged)
        and bool(valid_voltage)
        and final_gen_q_nan == 0
        and final_load_q_nan == 0
    )

    # ----------------------------------------------------------------------------------------------
    # LINE RESULTS
    # ----------------------------------------------------------------------------------------------

    max_line_loading = np.nan
    overloaded_lines = np.nan

    if (
        converged
        and hasattr(n, "lines_t")
        and "p0" in n.lines_t
    ):

        try:

            p0 = pd.to_numeric(
                n.lines_t.p0.loc[SNAPSHOT],
                errors="coerce"
            )

            q0 = pd.to_numeric(
                n.lines_t.q0.loc[SNAPSHOT],
                errors="coerce"
            )

            apparent = np.sqrt(
                p0**2 + q0**2
            )

            s_nom = pd.to_numeric(
                n.lines["s_nom"],
                errors="coerce"
            )

            loading = (
                apparent
                / s_nom
                * 100.0
            )

            finite_loading = loading[
                np.isfinite(loading)
            ]

            if len(finite_loading) > 0:

                max_line_loading = float(
                    finite_loading.max()
                )

                overloaded_lines = int(
                    (finite_loading > 100.0).sum()
                )

        except Exception:
            pass

    # ----------------------------------------------------------------------------------------------
    # RESULT PRINT
    # ----------------------------------------------------------------------------------------------

    print()
    print("-" * 100)
    print(f"CASE RESULT : {name}")
    print("-" * 100)

    print(
        f"Converged                    : {converged}"
    )

    print(
        f"Valid physical solution      : {valid_physical}"
    )

    print(
        f"PF error                     : {pf_error}"
    )

    print(
        f"Iterations                   : {iterations}"
    )

    print(
        f"Minimum voltage              : {vmin}"
    )

    print(
        f"Maximum voltage              : {vmax}"
    )

    print(
        f"Maximum line loading        : "
        f"{max_line_loading}"
    )

    print(
        f"Overloaded lines             : "
        f"{overloaded_lines}"
    )

    # ----------------------------------------------------------------------------------------------
    # STORE RESULT
    # ----------------------------------------------------------------------------------------------

    results.append(
        {
            "case": name,
            "generator_q_zero": case["gen_q_zero"],
            "load_q_zero": case["load_q_zero"],
            "explicit_slack": case["explicit_slack"],
            "distributed_slack": case["distributed_slack"],
            "flat_voltage": case["flat_voltage"],
            "slack_generator": slack_generator,
            "generator_q_nan_count": final_gen_q_nan,
            "load_q_nan_count": final_load_q_nan,
            "generation_mw": gen_p_case,
            "load_mw": load_p_case,
            "generation_minus_load_mw": balance_case,
            "converged": converged,
            "pf_error": pf_error,
            "iterations": iterations,
            "valid_voltage": valid_voltage,
            "valid_physical_solution": valid_physical,
            "min_voltage_pu": vmin,
            "max_voltage_pu": vmax,
            "max_line_loading_pct": max_line_loading,
            "overloaded_lines": overloaded_lines,
        }
    )


# ==================================================================================================
# SUMMARY
# ==================================================================================================

results_df = pd.DataFrame(results)


print()
print("=" * 100)
print("S4.5D — CONVERGENCE REPAIR SUMMARY")
print("=" * 100)

print()

print(
    results_df[
        [
            "case",
            "generator_q_nan_count",
            "load_q_nan_count",
            "explicit_slack",
            "distributed_slack",
            "converged",
            "valid_physical_solution",
            "min_voltage_pu",
            "max_voltage_pu",
            "max_line_loading_pct",
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# INTERPRETATION
# ==================================================================================================

print()
print("=" * 100)
print("S4.5D INTERPRETATION")
print("=" * 100)

successful = results_df[
    results_df["valid_physical_solution"] == True
]

if len(successful) == 0:

    print(
        """
NO VALID BASELINE AC POWER-FLOW FORMULATION WAS FOUND.

This means the problem is deeper than simply NaN reactive setpoints
or the absence of an explicit slack generator.

Next investigation should inspect:

    1. Network topology / connected components
    2. Y-bus conditioning
    3. Branch impedance scaling
    4. Transformer per-unit conversion
    5. Bus voltage-angle initialization
    6. Generator/load sign conventions
    7. Potentially disconnected or weakly connected buses
    8. The exact network construction stage that produced the .nc file
"""
    )

else:

    print()
    print("VALID AC POWER-FLOW FORMULATION FOUND.")

    best = successful.iloc[0]

    print()
    print(
        f"First valid case : {best['case']}"
    )

    print(
        f"Minimum voltage  : "
        f"{best['min_voltage_pu']:.6f} pu"
    )

    print(
        f"Maximum voltage  : "
        f"{best['max_voltage_pu']:.6f} pu"
    )

    print(
        """
IMPORTANT:

The successful case identifies the numerical formulation required
for the subsequent S4.5 voltage-bottleneck analysis.

It does NOT yet establish a physical voltage bottleneck.

The next stage must use the validated formulation consistently.
"""
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False
)


print()
print("=" * 100)
print("SAVING RESULTS")
print("=" * 100)

print()
print(f"Results saved to:")
print(f"  {OUTPUT}")

print()
print("=" * 100)
print("S4.5D COMPLETE")
print("=" * 100)

print()
print("IMPORTANT:")
print("No network file was modified.")
print("No reinforcement was applied.")
print("No reactive compensation was added.")
print("Only diagnostic copies of the source network were modified in memory.")
print("=" * 100)