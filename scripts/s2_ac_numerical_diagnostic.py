"""
S2 AC NUMERICAL DIAGNOSTIC

Purpose:
    Determine whether S2 AC power-flow failure is caused by:
    1. Active-power imbalance
    2. Reactive-power handling
    3. Network electrical structure / parameters

This script is DIAGNOSTIC ONLY.
It does not modify the saved network.

Network:
    data/processed/eirgrid_optimized_network.nc

Snapshot:
    S2_PEAK_DEMAND
"""

from pathlib import Path
import copy

import numpy as np
import pandas as pd
import pypsa


# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

Q_LOAD_FACTOR = 0.0
Q_GENERATOR_FACTOR = 0.0

VOLTAGE_MIN = 0.5
VOLTAGE_MAX = 1.5


# ============================================================
# HELPERS
# ============================================================

def ensure_snapshot_q_tables(n):
    """
    Ensure PyPSA has generators_t.q_set and loads_t.q_set
    with the correct snapshot/component structure.
    """

    # --------------------------------------------------------
    # Generator Q table
    # --------------------------------------------------------
    if (
        not hasattr(n.generators_t, "q_set")
        or n.generators_t.q_set is None
        or len(n.generators_t.q_set.columns) == 0
    ):
        n.generators_t.q_set = pd.DataFrame(
            0.0,
            index=n.snapshots,
            columns=n.generators.index,
        )
    else:
        q = n.generators_t.q_set.copy()

        q = q.reindex(
            index=n.snapshots,
            columns=n.generators.index,
            fill_value=0.0,
        )

        q = q.fillna(0.0)

        n.generators_t.q_set = q

    # --------------------------------------------------------
    # Load Q table
    # --------------------------------------------------------
    if (
        not hasattr(n.loads_t, "q_set")
        or n.loads_t.q_set is None
        or len(n.loads_t.q_set.columns) == 0
    ):
        n.loads_t.q_set = pd.DataFrame(
            0.0,
            index=n.snapshots,
            columns=n.loads.index,
        )
    else:
        q = n.loads_t.q_set.copy()

        q = q.reindex(
            index=n.snapshots,
            columns=n.loads.index,
            fill_value=0.0,
        )

        q = q.fillna(0.0)

        n.loads_t.q_set = q


def set_all_q_zero(n):
    """
    Set all generator and load reactive-power setpoints to zero.
    """

    ensure_snapshot_q_tables(n)

    n.generators_t.q_set.loc[:, :] = 0.0
    n.loads_t.q_set.loc[:, :] = 0.0


def get_p_generation(n):
    """
    Total generator active power at S2.
    """

    if len(n.generators) == 0:
        return 0.0

    return float(
        n.generators_t.p_set.loc[SNAPSHOT].sum()
    )


def get_p_load(n):
    """
    Total load active power at S2.
    """

    if len(n.loads) == 0:
        return 0.0

    return float(
        n.loads_t.p_set.loc[SNAPSHOT].sum()
    )


def print_power_balance(n, label):
    """
    Print active power balance.
    """

    generation = get_p_generation(n)
    load = get_p_load(n)
    difference = generation - load

    print()
    print(label)
    print("-" * 70)
    print(f"Generation : {generation:.6f} MW")
    print(f"Load       : {load:.6f} MW")
    print(f"Difference : {difference:.6f} MW")


def print_voltage_diagnostic(n):
    """
    Print voltage statistics after PF.
    """

    try:
        v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

    except Exception as exc:
        print()
        print("VOLTAGE DIAGNOSTIC")
        print("-" * 70)
        print(f"Unable to read bus voltages: {exc}")
        return

    v = pd.to_numeric(v, errors="coerce")

    finite = v[np.isfinite(v)]

    print()
    print("VOLTAGE DIAGNOSTIC")
    print("-" * 70)

    if len(finite) == 0:
        print("No finite voltage values.")
        return

    print(
        f"Minimum finite voltage : "
        f"{finite.min():.6f} pu"
    )

    print(
        f"Maximum finite voltage : "
        f"{finite.max():.6f} pu"
    )

    suspicious = v[
        (~np.isfinite(v))
        | (v < VOLTAGE_MIN)
        | (v > VOLTAGE_MAX)
    ]

    print(
        f"Suspicious buses       : "
        f"{len(suspicious)}"
    )

    if len(suspicious) > 0:
        print()
        print(
            suspicious
            .sort_values()
            .to_string()
        )


def run_pf_test(n, label):
    """
    Run PyPSA nonlinear AC power flow and print diagnostics.
    """

    print()
    print("=" * 70)
    print(f"RUNNING: {label}")
    print("=" * 70)

    try:
        result = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-6,
            distribute_slack=False,
        )

        print()
        print("PF RETURN")
        print("-" * 70)
        print(result)

        converged = result["converged"]

        print()
        print("CONVERGENCE")
        print("-" * 70)
        print(converged)

        try:
            did_converge = bool(
                converged.loc[SNAPSHOT].all()
            )
        except Exception:
            did_converge = False

        if did_converge:
            print()
            print("RESULT: PASS — AC PF converged.")
        else:
            print()
            print("RESULT: FAIL — AC PF did not converge.")

        print_voltage_diagnostic(n)

        # ----------------------------------------------------
        # Line flow diagnostics
        # ----------------------------------------------------

        if hasattr(n.lines_t, "p0") and len(n.lines):

            try:
                p0 = (
                    n.lines_t.p0
                    .loc[SNAPSHOT]
                    .replace([np.inf, -np.inf], np.nan)
                )

                p1 = (
                    n.lines_t.p1
                    .loc[SNAPSHOT]
                    .replace([np.inf, -np.inf], np.nan)
                )

                if p0.notna().any():
                    print()
                    print("LINE FLOW DIAGNOSTIC")
                    print("-" * 70)

                    print(
                        f"Maximum |P0| : "
                        f"{p0.abs().max():.6f} MW"
                    )

                    print(
                        f"Maximum |P1| : "
                        f"{p1.abs().max():.6f} MW"
                    )

            except Exception as exc:
                print()
                print(
                    f"Line-flow diagnostic unavailable: {exc}"
                )

        # ----------------------------------------------------
        # Generator diagnostics
        # ----------------------------------------------------

        try:
            print()
            print("GENERATOR RESULTS")
            print("-" * 70)

            print(
                n.generators_t.p
                .loc[SNAPSHOT]
                .to_string()
            )

        except Exception as exc:
            print(
                f"Generator result unavailable: {exc}"
            )

        return result

    except Exception as exc:

        print()
        print("POWER FLOW EXCEPTION")
        print("-" * 70)
        print(type(exc).__name__, str(exc))

        return None


def set_slack_generator(n, generator_name):
    """
    Convert selected generator into the slack generator.
    """

    if generator_name not in n.generators.index:
        raise ValueError(
            f"Slack generator not found: {generator_name}"
        )

    # Set all generators to PQ first
    n.generators.control = "PQ"

    # Ensure sufficient nominal capacity for diagnostic slack
    n.generators.at[
        generator_name,
        "p_nom"
    ] = max(
        float(n.generators.at[generator_name, "p_nom"]),
        10000.0,
    )

    n.generators.at[
        generator_name,
        "control"
    ] = "Slack"


def balance_non_wind_generation(n):
    """
    Balance S2 active power by changing only the non-wind
    generator output.

    This is a diagnostic experiment.
    """

    wind = n.generators[
        n.generators.carrier == "wind"
    ].index

    imports = n.generators[
        n.generators.carrier == "interconnector"
    ].index

    non_wind = n.generators[
        n.generators.carrier == "non_wind"
    ].index

    wind_p = float(
        n.generators_t.p_set
        .loc[SNAPSHOT, wind]
        .sum()
    )

    import_p = float(
        n.generators_t.p_set
        .loc[SNAPSHOT, imports]
        .sum()
    )

    load = get_p_load(n)

    required_non_wind = (
        load
        - wind_p
        - import_p
    )

    if len(non_wind) == 0:
        raise RuntimeError(
            "No non_wind generator found."
        )

    # Use the first non-wind generator
    gen = non_wind[0]

    n.generators_t.p_set.loc[
        SNAPSHOT,
        gen
    ] = required_non_wind

    return (
        wind_p,
        import_p,
        required_non_wind,
    )


# ============================================================
# LOAD NETWORK
# ============================================================

print()
print("=" * 70)
print("S2 AC NUMERICAL DIAGNOSTIC")
print("=" * 70)

print()
print(f"Network: {NETWORK_PATH}")
print(f"Snapshot: {SNAPSHOT}")

n = pypsa.Network(NETWORK_PATH)

print()
print("NETWORK")
print("-" * 70)

print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")
print(f"Links        : {len(n.links)}")


# ============================================================
# ORIGINAL OPERATING POINT
# ============================================================

print()
print("ORIGINAL S2 OPERATING POINT")
print("-" * 70)

original_generation = get_p_generation(n)
original_load = get_p_load(n)

print(
    f"Generation : {original_generation:.6f} MW"
)

print(
    f"Load       : {original_load:.6f} MW"
)

print(
    f"Difference : "
    f"{original_generation - original_load:.6f} MW"
)


# ============================================================
# S2 COMPONENT BREAKDOWN
# ============================================================

print()
print("S2 COMPONENTS")
print("-" * 70)

wind_generators = n.generators[
    n.generators.carrier == "wind"
].index

non_wind_generators = n.generators[
    n.generators.carrier == "non_wind"
].index

import_generators = n.generators[
    n.generators.carrier == "interconnector"
].index

wind_p = float(
    n.generators_t.p_set
    .loc[SNAPSHOT, wind_generators]
    .sum()
)

non_wind_p = float(
    n.generators_t.p_set
    .loc[SNAPSHOT, non_wind_generators]
    .sum()
)

imports_p = float(
    n.generators_t.p_set
    .loc[SNAPSHOT, import_generators]
    .sum()
)

print(f"Wind        : {wind_p:.6f} MW")
print(f"Non-wind    : {non_wind_p:.6f} MW")
print(f"Imports     : {imports_p:.6f} MW")
print(f"Load        : {original_load:.6f} MW")


# ============================================================
# BALANCED MAIN-GRID CALCULATION
# ============================================================

required_non_wind = (
    original_load
    - wind_p
    - imports_p
)

print()
print("BALANCED MAIN-GRID CALCULATION")
print("-" * 70)

print(
    f"Required non-wind generation : "
    f"{required_non_wind:.6f} MW"
)

print(
    f"Original non-wind generation : "
    f"{non_wind_p:.6f} MW"
)

print(
    f"Difference                  : "
    f"{required_non_wind - non_wind_p:.6f} MW"
)


# ============================================================
# TEST 0
# ORIGINAL S2 + Q = 0
# ============================================================

n0 = copy.deepcopy(n)

set_all_q_zero(n0)

print_power_balance(
    n0,
    "TEST 0 — ORIGINAL S2 + Q=0"
)

set_slack_generator(
    n0,
    "eirgrid_non_wind_generation"
)

result0 = run_pf_test(
    n0,
    "TEST 0 — ORIGINAL S2 + Q=0"
)


# ============================================================
# TEST 1
# BALANCED P + Q = 0
# ============================================================

n1 = copy.deepcopy(n)

set_all_q_zero(n1)

(
    test1_wind,
    test1_imports,
    test1_required_non_wind,
) = balance_non_wind_generation(n1)

print()
print("TEST 1 — BALANCED P + Q=0")
print("-" * 70)

print(
    f"Generation : "
    f"{get_p_generation(n1):.6f} MW"
)

print(
    f"Load       : "
    f"{get_p_load(n1):.6f} MW"
)

print(
    f"Difference : "
    f"{get_p_generation(n1) - get_p_load(n1):.6f} MW"
)

set_slack_generator(
    n1,
    "eirgrid_non_wind_generation"
)

result1 = run_pf_test(
    n1,
    "TEST 1 — BALANCED P + Q=0"
)


# ============================================================
# TEST 2
# BALANCED P + SMALL REACTIVE LOAD
# ============================================================

n2 = copy.deepcopy(n)

(
    test2_wind,
    test2_imports,
    test2_required_non_wind,
) = balance_non_wind_generation(n2)

ensure_snapshot_q_tables(n2)

# ------------------------------------------------------------
# Create a small realistic diagnostic reactive demand:
#
# Q = 0.20 * P
#
# This is NOT intended to reproduce actual EirGrid Q.
# It simply tests whether nonzero load Q changes behaviour.
# ------------------------------------------------------------

q_load = (
    0.20
    * n2.loads_t.p_set.loc[SNAPSHOT]
)

q_load = q_load.fillna(0.0)

n2.loads_t.q_set.loc[
    SNAPSHOT,
    :
] = q_load

n2.generators_t.q_set.loc[
    SNAPSHOT,
    :
] = 0.0

print()
print("TEST 2 — BALANCED P + SMALL REACTIVE LOAD")
print("-" * 70)

print(
    f"Total Q load : "
    f"{q_load.sum():.6f} MVAr"
)

set_slack_generator(
    n2,
    "eirgrid_non_wind_generation"
)

result2 = run_pf_test(
    n2,
    "TEST 2 — BALANCED P + SMALL REACTIVE LOAD"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

def convergence_status(result):

    if result is None:
        return "EXCEPTION"

    try:
        value = bool(
            result["converged"]
            .loc[SNAPSHOT]
            .all()
        )

        return "PASS" if value else "FAIL"

    except Exception:
        return "UNKNOWN"


print()
print(
    f"TEST 0 — Original P + Q=0       : "
    f"{convergence_status(result0)}"
)

print(
    f"TEST 1 — Balanced P + Q=0       : "
    f"{convergence_status(result1)}"
)

print(
    f"TEST 2 — Balanced P + Q=0.2P    : "
    f"{convergence_status(result2)}"
)

print()
print("INTERPRETATION")
print("-" * 70)

if (
    convergence_status(result0) == "FAIL"
    and convergence_status(result1) == "FAIL"
):

    print(
        "Active-power imbalance is NOT the primary cause."
    )

if convergence_status(result1) == "FAIL":

    print(
        "Balanced active power still fails."
    )

    print(
        "Next investigation: network electrical "
        "parameter/component isolation."
    )

if convergence_status(result2) == "PASS":

    print(
        "Reactive-power modelling materially affects "
        "convergence."
    )

elif convergence_status(result1) == "FAIL":

    print(
        "Changing Q alone did not establish convergence."
    )

    print(
        "Do NOT modify reinforcement yet."
    )

print()
print("=" * 70)
print("S2 AC NUMERICAL DIAGNOSTIC COMPLETE")
print("=" * 70)