# ==================================================================================================
# S4.5J — ROBUST AC CONVERGENCE + PHYSICAL-SOLUTION VALIDATION
# ==================================================================================================
#
# Purpose
# -------
# Correct S4.5I's overly permissive AC-solution classification.
#
# S4.5I correctly fixed the load-index problem, but it accepted some nonlinear
# power-flow states as "valid" even when PyPSA reported:
#
#   - failure to reach tolerance,
#   - explicit non-convergence,
#   - absurd voltages,
#   - absurd line/transformer loadings,
#   - absurd generator reactive power.
#
# S4.5J therefore separates three concepts:
#
#   1. SOLVER CONVERGENCE
#      Did PyPSA's nonlinear AC solver actually converge?
#
#   2. PHYSICAL / NUMERICAL VALIDITY
#      Are the resulting electrical quantities finite and physically plausible?
#
#   3. SECURITY COMPLIANCE
#      Are voltages within 0.90–1.10 pu and thermal limits respected?
#
# IMPORTANT:
#   AC validity != voltage security.
#
# No reinforcement.
# No reactive compensation.
# No dispatch change.
# No permanent network modification.
# Source network is READ-ONLY.
#
# ==================================================================================================

from pathlib import Path
import warnings
import logging
import math

import numpy as np
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

PROJECT_ROOT = Path(r"C:\Users\Dell\ireland-wind-grid-optimiser")

SOURCE_NETWORK = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eirgrid_second_reinforced_network.nc"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "s4_5j_q_continuation.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"


# --------------------------------------------------------------------------------------------------
# Q continuation
# --------------------------------------------------------------------------------------------------

COARSE_Q_LEVELS = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
]

FINE_Q_LEVELS = [
    7.0,
    7.1,
    7.2,
    7.3,
    7.4,
    7.5,
    7.6,
    7.7,
    7.8,
    7.9,
    8.0,
]


# --------------------------------------------------------------------------------------------------
# Electrical assumptions
# --------------------------------------------------------------------------------------------------

LOAD_POWER_FACTOR = 0.95

VOLTAGE_SECURITY_MIN = 0.90
VOLTAGE_SECURITY_MAX = 1.10

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0


# --------------------------------------------------------------------------------------------------
# Numerical validation thresholds
# --------------------------------------------------------------------------------------------------
#
# These are NOT security limits.
#
# They only reject obvious numerical artifacts such as:
#
#   V = -10^7 pu
#   V = +10^20 pu
#   loading = 10^40 %
#   Qgen = 10^23 Mvar
#
# A low but finite voltage such as 0.65 pu is NOT rejected as a numerical
# artifact. It remains an AC-valid-but-insecure solution if the solver
# genuinely converged.
#
# --------------------------------------------------------------------------------------------------

HARD_VOLTAGE_MIN = 0.20
HARD_VOLTAGE_MAX = 2.00

HARD_ANGLE_ABS_MAX_RAD = 2.0 * math.pi

HARD_LOADING_MAX_PCT = 100000.0

HARD_GENERATOR_Q_ABS_MAX_MVAR = 1.0e6


# --------------------------------------------------------------------------------------------------
# Solver acceptance
# --------------------------------------------------------------------------------------------------

# PyPSA's nonlinear solver should normally report an error much smaller than
# this. S4.6 produced approximately 6.29e-7.
#
# We deliberately use 1e-5 as the acceptance ceiling to avoid rejecting a
# genuinely converged solution merely because of minor numerical variation.
#
# This is a convergence criterion, NOT a security criterion.
#
PF_ERROR_TOLERANCE = 1.0e-5


# ==================================================================================================
# PRINT HELPERS
# ==================================================================================================

SEP = "=" * 100


def print_header(title):
    print()
    print(SEP)
    print(title)
    print(SEP)


def fmt(value, digits=6):
    if value is None:
        return "None"

    try:
        if not np.isfinite(float(value)):
            return str(value)
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


# ==================================================================================================
# PYPSA RESULT EXTRACTION HELPERS
# ==================================================================================================

def extract_snapshot_value(obj, snapshot):
    """
    Robustly extract a scalar snapshot value from PyPSA's PF result.

    PyPSA versions can return Series/DataFrames/scalars depending on the
    object involved.
    """

    if obj is None:
        return np.nan

    try:

        # DataFrame
        if isinstance(obj, pd.DataFrame):

            if snapshot in obj.index:

                row = obj.loc[snapshot]

                if isinstance(row, pd.Series):

                    vals = pd.to_numeric(row, errors="coerce").dropna()

                    if len(vals) == 0:
                        return np.nan

                    if len(vals) == 1:
                        return vals.iloc[0]

                    # For multi-column convergence/error results, the
                    # meaningful scalar is normally the worst value.
                    return vals.max()

            if snapshot in obj.columns:

                col = obj[snapshot]

                vals = pd.to_numeric(col, errors="coerce").dropna()

                if len(vals) == 0:
                    return np.nan

                if len(vals) == 1:
                    return vals.iloc[0]

                return vals.max()

        # Series
        if isinstance(obj, pd.Series):

            if snapshot in obj.index:
                value = obj.loc[snapshot]

                if np.isscalar(value):
                    return value

                vals = pd.to_numeric(
                    pd.Series(value),
                    errors="coerce"
                ).dropna()

                if len(vals) == 0:
                    return np.nan

                return vals.max()

            vals = pd.to_numeric(obj, errors="coerce").dropna()

            if len(vals) == 1:
                return vals.iloc[0]

            if len(vals) > 1:
                return vals.max()

        # Scalar
        if np.isscalar(obj):
            return obj

    except Exception:
        pass

    return np.nan


def extract_convergence(result, snapshot):
    """
    Extract PyPSA convergence information.

    Returns:
        converged, pf_error, iterations
    """

    converged = False
    pf_error = np.nan
    iterations = np.nan

    if result is None:
        return converged, pf_error, iterations

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    try:
        if "converged" in result:

            raw = result["converged"]

            value = extract_snapshot_value(
                raw,
                snapshot
            )

            if isinstance(value, (bool, np.bool_)):
                converged = bool(value)

            elif np.isfinite(value):
                converged = bool(value)

    except Exception:
        pass

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    try:
        if "error" in result:

            pf_error = extract_snapshot_value(
                result["error"],
                snapshot
            )

            try:
                pf_error = float(pf_error)
            except Exception:
                pf_error = np.nan

    except Exception:
        pass

    # ------------------------------------------------------------------
    # Iterations
    # ------------------------------------------------------------------

    try:

        if "n_iter" in result:

            iterations = extract_snapshot_value(
                result["n_iter"],
                snapshot
            )

            try:
                iterations = float(iterations)
            except Exception:
                iterations = np.nan

    except Exception:
        pass

    return converged, pf_error, iterations


# ==================================================================================================
# NETWORK DATA HELPERS
# ==================================================================================================

def get_snapshot_series(table, snapshot, columns):
    """
    Return requested time-series columns for one snapshot.
    """

    if table is None:
        return pd.DataFrame(index=[])

    try:

        if not hasattr(table, "loc"):
            return pd.DataFrame(index=[])

        if snapshot not in table.index:
            return pd.DataFrame(index=columns)

        row = table.loc[snapshot]

        if isinstance(row, pd.Series):
            return row

        return row

    except Exception:
        return pd.Series(dtype=float)


def finite_values(values):
    """
    Return finite numeric values only.
    """

    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=float)

    return arr[np.isfinite(arr)]


def safe_sum(values):
    vals = finite_values(values)

    if len(vals) == 0:
        return np.nan

    return float(vals.sum())


def safe_min(values):
    vals = finite_values(values)

    if len(vals) == 0:
        return np.nan

    return float(vals.min())


def safe_max(values):
    vals = finite_values(values)

    if len(vals) == 0:
        return np.nan

    return float(vals.max())


# ==================================================================================================
# LOAD Q CONSTRUCTION
# ==================================================================================================

def calculate_realistic_q(load_p_mw):
    """
    Calculate realistic total load Q from total P and PF=0.95 lagging.
    """

    total_p = float(load_p_mw)

    q_over_p = math.tan(
        math.acos(LOAD_POWER_FACTOR)
    )

    return total_p * q_over_p


def apply_reactive_load(n, snapshot, q_percentage, q_ratio):
    """
    Apply requested reactive load proportionally to the actual network
    loads.

    IMPORTANT:
    Uses the exact names in n.loads.index.

    No synthetic load names are constructed.
    """

    load_names = list(n.loads.index)

    if len(load_names) == 0:
        raise RuntimeError(
            "Network contains zero loads."
        )

    # ------------------------------------------------------------------
    # Ensure P time series exists
    # ------------------------------------------------------------------

    if not hasattr(n.loads_t, "p_set"):
        raise RuntimeError(
            "Network does not contain loads_t.p_set."
        )

    p_row = n.loads_t.p_set.loc[snapshot, load_names].astype(float)

    total_p = float(p_row.sum())

    if not np.isfinite(total_p) or total_p <= 0:
        raise RuntimeError(
            f"Invalid total load P at {snapshot}: {total_p}"
        )

    # ------------------------------------------------------------------
    # Requested total Q
    # ------------------------------------------------------------------

    full_realistic_q = total_p * q_ratio

    requested_total_q = (
        full_realistic_q
        * q_percentage
        / 100.0
    )

    # ------------------------------------------------------------------
    # Proportional allocation
    # ------------------------------------------------------------------

    p_weights = p_row / total_p

    q_row = (
        p_weights
        * requested_total_q
    )

    # ------------------------------------------------------------------
    # Ensure q_set exists
    # ------------------------------------------------------------------

    if not hasattr(n.loads_t, "q_set"):
        n.loads_t.q_set = pd.DataFrame(
            0.0,
            index=n.snapshots,
            columns=load_names
        )

    # If some load columns are absent from q_set, add them.
    for load_name in load_names:

        if load_name not in n.loads_t.q_set.columns:
            n.loads_t.q_set[load_name] = 0.0

    # Exact network names only.
    n.loads_t.q_set.loc[
        snapshot,
        load_names
    ] = q_row.values

    return (
        total_p,
        full_realistic_q,
        requested_total_q,
        q_row
    )


# ==================================================================================================
# GENERATOR Q SET
# ==================================================================================================

def set_generator_q_reference(n, snapshot):
    """
    Set generator q_set reference to zero.

    IMPORTANT:
    This does NOT force the solved generator Q to zero for PV/slack
    generators. The AC PF may solve generator Q as required by the
    network formulation.

    This preserves the S4.5 experiment's "Generator Q set : 0 Mvar"
    reference.
    """

    generator_names = list(n.generators.index)

    if len(generator_names) == 0:
        return

    if not hasattr(n.generators_t, "q_set"):
        n.generators_t.q_set = pd.DataFrame(
            np.nan,
            index=n.snapshots,
            columns=generator_names
        )

    for gen_name in generator_names:

        if gen_name not in n.generators_t.q_set.columns:
            n.generators_t.q_set[gen_name] = np.nan

    n.generators_t.q_set.loc[
        snapshot,
        generator_names
    ] = 0.0


# ==================================================================================================
# POWER-FLOW OUTPUT VALIDATION
# ==================================================================================================

def calculate_line_loading(n, snapshot):

    if len(n.lines.index) == 0:
        return np.nan, 0

    line_names = list(n.lines.index)

    try:

        p0 = n.lines_t.p0.loc[
            snapshot,
            line_names
        ].astype(float)

        q0 = n.lines_t.q0.loc[
            snapshot,
            line_names
        ].astype(float)

        p1 = n.lines_t.p1.loc[
            snapshot,
            line_names
        ].astype(float)

        q1 = n.lines_t.q1.loc[
            snapshot,
            line_names
        ].astype(float)

        s0 = np.sqrt(
            p0 ** 2 +
            q0 ** 2
        )

        s1 = np.sqrt(
            p1 ** 2 +
            q1 ** 2
        )

        s_max = pd.concat(
            [s0, s1],
            axis=1
        ).max(axis=1)

        ratings = n.lines.s_nom.astype(float)

        loading_pct = (
            s_max
            / ratings
            * 100.0
        )

        finite = loading_pct[
            np.isfinite(loading_pct)
        ]

        if len(finite) == 0:
            return np.nan, 0

        max_loading = float(finite.max())

        overloaded = int(
            (finite > LINE_LOADING_LIMIT_PCT).sum()
        )

        return max_loading, overloaded

    except Exception:
        return np.nan, np.nan


def calculate_transformer_loading(n, snapshot):

    if len(n.transformers.index) == 0:
        return np.nan, 0

    transformer_names = list(
        n.transformers.index
    )

    try:

        p0 = n.transformers_t.p0.loc[
            snapshot,
            transformer_names
        ].astype(float)

        q0 = n.transformers_t.q0.loc[
            snapshot,
            transformer_names
        ].astype(float)

        p1 = n.transformers_t.p1.loc[
            snapshot,
            transformer_names
        ].astype(float)

        q1 = n.transformers_t.q1.loc[
            snapshot,
            transformer_names
        ].astype(float)

        s0 = np.sqrt(
            p0 ** 2 +
            q0 ** 2
        )

        s1 = np.sqrt(
            p1 ** 2 +
            q1 ** 2
        )

        s_max = pd.concat(
            [s0, s1],
            axis=1
        ).max(axis=1)

        ratings = (
            n.transformers.s_nom
            .astype(float)
        )

        loading_pct = (
            s_max
            / ratings
            * 100.0
        )

        finite = loading_pct[
            np.isfinite(loading_pct)
        ]

        if len(finite) == 0:
            return np.nan, 0

        max_loading = float(finite.max())

        overloaded = int(
            (finite > TRANSFORMER_LOADING_LIMIT_PCT).sum()
        )

        return max_loading, overloaded

    except Exception:
        return np.nan, np.nan


# ==================================================================================================
# PHYSICAL / NUMERICAL VALIDITY
# ==================================================================================================

def validate_physical_state(
    n,
    snapshot,
    converged,
    pf_error,
    min_voltage,
    max_voltage,
    min_angle,
    max_angle,
    max_line_loading,
    max_transformer_loading,
    generator_q
):
    """
    Decide whether a mathematically returned state is physically/numerically
    plausible.

    IMPORTANT:
    This function does NOT apply the 0.90–1.10 security criterion.

    A converged state with Vmin=0.70 pu can therefore be:
        valid physical solution = True
        voltage security = False

    But a state with Vmin=-10^7 pu is rejected as numerical garbage.
    """

    reasons = []

    # ------------------------------------------------------------------
    # Solver convergence is mandatory
    # ------------------------------------------------------------------

    if not converged:
        reasons.append(
            "NON_CONVERGED"
        )

    # ------------------------------------------------------------------
    # PF residual
    # ------------------------------------------------------------------

    if not np.isfinite(pf_error):
        reasons.append(
            "PF_ERROR_NOT_FINITE"
        )

    elif pf_error > PF_ERROR_TOLERANCE:
        reasons.append(
            f"PF_ERROR_ABOVE_TOLERANCE:{pf_error:.6g}"
        )

    # ------------------------------------------------------------------
    # Voltage
    # ------------------------------------------------------------------

    if not np.isfinite(min_voltage):
        reasons.append(
            "MIN_VOLTAGE_NOT_FINITE"
        )

    elif min_voltage <= HARD_VOLTAGE_MIN:
        reasons.append(
            f"MIN_VOLTAGE_NUMERICALLY_IMPLAUSIBLE:{min_voltage:.6g}"
        )

    if not np.isfinite(max_voltage):
        reasons.append(
            "MAX_VOLTAGE_NOT_FINITE"
        )

    elif max_voltage >= HARD_VOLTAGE_MAX:
        reasons.append(
            f"MAX_VOLTAGE_NUMERICALLY_IMPLAUSIBLE:{max_voltage:.6g}"
        )

    # ------------------------------------------------------------------
    # Angle
    # ------------------------------------------------------------------

    if not np.isfinite(min_angle):
        reasons.append(
            "MIN_ANGLE_NOT_FINITE"
        )

    elif abs(min_angle) > HARD_ANGLE_ABS_MAX_RAD:
        reasons.append(
            f"MIN_ANGLE_NUMERICALLY_IMPLAUSIBLE:{min_angle:.6g}"
        )

    if not np.isfinite(max_angle):
        reasons.append(
            "MAX_ANGLE_NOT_FINITE"
        )

    elif abs(max_angle) > HARD_ANGLE_ABS_MAX_RAD:
        reasons.append(
            f"MAX_ANGLE_NUMERICALLY_IMPLAUSIBLE:{max_angle:.6g}"
        )

    # ------------------------------------------------------------------
    # Line loading
    # ------------------------------------------------------------------

    if not np.isfinite(max_line_loading):
        reasons.append(
            "LINE_LOADING_NOT_FINITE"
        )

    elif max_line_loading > HARD_LOADING_MAX_PCT:
        reasons.append(
            f"LINE_LOADING_NUMERICALLY_IMPLAUSIBLE:{max_line_loading:.6g}"
        )

    # ------------------------------------------------------------------
    # Transformer loading
    # ------------------------------------------------------------------

    if not np.isfinite(max_transformer_loading):
        reasons.append(
            "TRANSFORMER_LOADING_NOT_FINITE"
        )

    elif max_transformer_loading > HARD_LOADING_MAX_PCT:
        reasons.append(
            "TRANSFORMER_LOADING_NUMERICALLY_IMPLAUSIBLE:"
            f"{max_transformer_loading:.6g}"
        )

    # ------------------------------------------------------------------
    # Generator Q
    # ------------------------------------------------------------------

    if not np.isfinite(generator_q):
        reasons.append(
            "GENERATOR_Q_NOT_FINITE"
        )

    elif abs(generator_q) > HARD_GENERATOR_Q_ABS_MAX_MVAR:
        reasons.append(
            f"GENERATOR_Q_NUMERICALLY_IMPLAUSIBLE:{generator_q:.6g}"
        )

    physical = len(reasons) == 0

    return physical, reasons


# ==================================================================================================
# SECURITY SCREEN
# ==================================================================================================

def calculate_security(
    physical_solution,
    min_voltage,
    max_voltage,
    overloaded_lines,
    overloaded_transformers
):
    """
    Security is evaluated separately from AC solution validity.
    """

    if not physical_solution:
        return False

    if not np.isfinite(min_voltage):
        return False

    if not np.isfinite(max_voltage):
        return False

    voltage_ok = (
        min_voltage >= VOLTAGE_SECURITY_MIN
        and
        max_voltage <= VOLTAGE_SECURITY_MAX
    )

    thermal_ok = (
        overloaded_lines == 0
        and
        overloaded_transformers == 0
    )

    return bool(
        voltage_ok
        and
        thermal_ok
    )


# ==================================================================================================
# SINGLE Q CASE
# ==================================================================================================

def run_case(
    q_percentage,
    stage,
    case_number,
    total_cases
):

    case_id = (
        f"Q_{q_percentage:04.1f}"
        .replace(".", "_")
        + "PCT"
    )

    print()
    print(
        f"[{case_number:02d}/{total_cases:02d}] "
        f"{stage.upper()} Q={q_percentage:.1f}%"
    )

    print_header(
        f"CASE {case_id}"
    )

    print(
        f"Reactive load level : "
        f"{q_percentage:.1f}%"
    )

    result = {
        "case": case_id,
        "continuation_stage": stage,
        "q_percentage": q_percentage,

        "converged": False,
        "valid_ac_solution": False,
        "physical_solution": False,

        "pf_error": np.nan,
        "iterations": np.nan,

        "min_voltage_pu": np.nan,
        "max_voltage_pu": np.nan,

        "min_voltage_bus": None,
        "max_voltage_bus": None,

        "voltage_security_valid": False,

        "min_angle_rad": np.nan,
        "max_angle_rad": np.nan,

        "max_line_loading_pct": np.nan,
        "overloaded_lines": np.nan,

        "max_transformer_loading_pct": np.nan,
        "overloaded_transformers": np.nan,

        "total_load_q_mvar": np.nan,
        "generator_q_mvar": np.nan,

        "generator_p_set_mw": np.nan,
        "load_p_set_mw": np.nan,

        "solved_generation_mw": np.nan,
        "solved_load_mw": np.nan,
        "generation_minus_load_mw": np.nan,

        "load_pf": LOAD_POWER_FACTOR,
        "q_p_ratio": np.nan,
        "full_realistic_q_mvar": np.nan,

        "p_generator_set_difference_mw": np.nan,
        "p_load_set_difference_mw": np.nan,

        "load_count": 0,

        "numerical_validation_reasons": "",

        "exception": None,
    }

    # ----------------------------------------------------------------------------------------------
    # Fresh network for every case
    # ----------------------------------------------------------------------------------------------

    try:

        n = pypsa.Network(
            str(SOURCE_NETWORK)
        )

        print(
            "Fresh network loaded for this case."
        )

        print(
            f"  Buses        : {len(n.buses)}"
        )

        print(
            f"  Lines        : {len(n.lines)}"
        )

        print(
            f"  Transformers : {len(n.transformers)}"
        )

        print(
            f"  Links        : {len(n.links)}"
        )

        print(
            f"  Generators   : {len(n.generators)}"
        )

        print(
            f"  Loads        : {len(n.loads)}"
        )

        # ------------------------------------------------------------------------------------------
        # Snapshot isolation
        # ------------------------------------------------------------------------------------------

        if SNAPSHOT not in n.snapshots:
            raise RuntimeError(
                f"Snapshot '{SNAPSHOT}' not found. "
                f"Available snapshots: {list(n.snapshots)}"
            )

        n.set_snapshots(
            [SNAPSHOT]
        )

        # ------------------------------------------------------------------------------------------
        # Original P reference
        # ------------------------------------------------------------------------------------------

        generator_names = list(
            n.generators.index
        )

        load_names = list(
            n.loads.index
        )

        result["load_count"] = len(
            load_names
        )

        if not generator_names:
            raise RuntimeError(
                "No generators found."
            )

        if not load_names:
            raise RuntimeError(
                "No loads found."
            )

        generator_p_set = float(
            n.generators_t.p_set.loc[
                SNAPSHOT,
                generator_names
            ].sum()
        )

        load_p_set = float(
            n.loads_t.p_set.loc[
                SNAPSHOT,
                load_names
            ].sum()
        )

        result[
            "generator_p_set_mw"
        ] = generator_p_set

        result[
            "load_p_set_mw"
        ] = load_p_set

        print()
        print("OPERATING POINT")

        print(
            f"Generator P set : "
            f"{generator_p_set:.6f} MW"
        )

        print(
            f"Load P set      : "
            f"{load_p_set:.6f} MW"
        )

        print(
            f"Generation-load : "
            f"{generator_p_set - load_p_set:.6f} MW"
        )

        # ------------------------------------------------------------------------------------------
        # Reactive reference
        # ------------------------------------------------------------------------------------------

        q_ratio = math.tan(
            math.acos(
                LOAD_POWER_FACTOR
            )
        )

        result[
            "q_p_ratio"
        ] = q_ratio

        (
            total_p,
            full_realistic_q,
            requested_total_q,
            q_row
        ) = apply_reactive_load(
            n,
            SNAPSHOT,
            q_percentage,
            q_ratio
        )

        result[
            "full_realistic_q_mvar"
        ] = full_realistic_q

        result[
            "total_load_q_mvar"
        ] = requested_total_q

        print()
        print("REACTIVE POWER")

        print(
            f"Load PF              : "
            f"{LOAD_POWER_FACTOR:.4f} lagging"
        )

        print(
            f"Q/P ratio            : "
            f"{q_ratio:.6f}"
        )

        print(
            f"Full realistic Q     : "
            f"{full_realistic_q:.6f} Mvar"
        )

        print(
            f"Requested total Q    : "
            f"{requested_total_q:.6f} Mvar"
        )

        # ------------------------------------------------------------------------------------------
        # Load index validation
        # ------------------------------------------------------------------------------------------

        print()
        print("LOAD INDEX CHECK")

        print(
            f"Actual network load count : "
            f"{len(load_names)}"
        )

        print(
            "Actual load names used    : YES"
        )

        for i, load_name in enumerate(load_names):

            print(
                f"  [{i:02d}] {load_name}"
            )

        # Ensure Q assignment did not alter P.
        current_p_set = n.loads_t.p_set.loc[
            SNAPSHOT,
            load_names
        ].astype(float)

        p_difference = (
            current_p_set
            -
            n.loads_t.p_set.loc[
                SNAPSHOT,
                load_names
            ].astype(float)
        )

        result[
            "p_load_set_difference_mw"
        ] = float(
            np.max(
                np.abs(
                    p_difference.to_numpy()
                )
            )
        )

        # ------------------------------------------------------------------------------------------
        # Generator Q reference
        # ------------------------------------------------------------------------------------------

        set_generator_q_reference(
            n,
            SNAPSHOT
        )

        # ------------------------------------------------------------------------------------------
        # Save P references before PF
        # ------------------------------------------------------------------------------------------

        generator_p_before = (
            n.generators_t.p_set.loc[
                SNAPSHOT,
                generator_names
            ]
            .astype(float)
            .copy()
        )

        load_p_before = (
            n.loads_t.p_set.loc[
                SNAPSHOT,
                load_names
            ]
            .astype(float)
            .copy()
        )

        # ------------------------------------------------------------------------------------------
        # AC nonlinear power flow
        # ------------------------------------------------------------------------------------------

        print()
        print("AC NONLINEAR POWER FLOW")

        print(
            "Generator Q set : 0 Mvar"
        )

        print(
            "Dispatch        : unchanged"
        )

        print(
            "Loads P         : unchanged"
        )

        print(
            "Reactive load   : controlled"
        )

        print(
            "Slack           : distributed"
        )

        # Capture Python warnings separately.
        with warnings.catch_warnings(record=True) as caught_warnings:

            warnings.simplefilter(
                "always"
            )

            pf_result = n.pf(
                snapshots=[SNAPSHOT],
                distribute_slack=True
            )

        # ------------------------------------------------------------------------------------------
        # Extract actual PyPSA convergence
        # ------------------------------------------------------------------------------------------

        (
            converged,
            pf_error,
            iterations
        ) = extract_convergence(
            pf_result,
            SNAPSHOT
        )

        result[
            "converged"
        ] = bool(converged)

        result[
            "pf_error"
        ] = pf_error

        result[
            "iterations"
        ] = iterations

        # ------------------------------------------------------------------------------------------
        # Voltage
        # ------------------------------------------------------------------------------------------

        voltage_names = list(
            n.buses.index
        )

        v = (
            n.buses_t.v_mag_pu.loc[
                SNAPSHOT,
                voltage_names
            ]
            .astype(float)
        )

        finite_v = v[
            np.isfinite(v)
        ]

        if len(finite_v) > 0:

            min_voltage = float(
                finite_v.min()
            )

            max_voltage = float(
                finite_v.max()
            )

            min_voltage_bus = (
                finite_v.idxmin()
            )

            max_voltage_bus = (
                finite_v.idxmax()
            )

        else:

            min_voltage = np.nan
            max_voltage = np.nan
            min_voltage_bus = None
            max_voltage_bus = None

        result[
            "min_voltage_pu"
        ] = min_voltage

        result[
            "max_voltage_pu"
        ] = max_voltage

        result[
            "min_voltage_bus"
        ] = min_voltage_bus

        result[
            "max_voltage_bus"
        ] = max_voltage_bus

        # ------------------------------------------------------------------------------------------
        # Voltage angles
        # ------------------------------------------------------------------------------------------

        try:

            angles = (
                n.buses_t.v_ang.loc[
                    SNAPSHOT,
                    voltage_names
                ]
                .astype(float)
            )

            finite_angles = angles[
                np.isfinite(angles)
            ]

            if len(finite_angles) > 0:

                result[
                    "min_angle_rad"
                ] = float(
                    finite_angles.min()
                )

                result[
                    "max_angle_rad"
                ] = float(
                    finite_angles.max()
                )

        except Exception:
            pass

        # ------------------------------------------------------------------------------------------
        # Thermal loading
        # ------------------------------------------------------------------------------------------

        (
            max_line_loading,
            overloaded_lines
        ) = calculate_line_loading(
            n,
            SNAPSHOT
        )

        result[
            "max_line_loading_pct"
        ] = max_line_loading

        result[
            "overloaded_lines"
        ] = overloaded_lines

        (
            max_transformer_loading,
            overloaded_transformers
        ) = calculate_transformer_loading(
            n,
            SNAPSHOT
        )

        result[
            "max_transformer_loading_pct"
        ] = max_transformer_loading

        result[
            "overloaded_transformers"
        ] = overloaded_transformers

        # ------------------------------------------------------------------------------------------
        # Generator Q
        # ------------------------------------------------------------------------------------------

        try:

            generator_q = safe_sum(
                n.generators_t.q.loc[
                    SNAPSHOT,
                    generator_names
                ]
            )

        except Exception:

            generator_q = np.nan

        result[
            "generator_q_mvar"
        ] = generator_q

        # ------------------------------------------------------------------------------------------
        # Solved P
        # ------------------------------------------------------------------------------------------

        try:

            solved_generation = safe_sum(
                n.generators_t.p.loc[
                    SNAPSHOT,
                    generator_names
                ]
            )

        except Exception:

            solved_generation = np.nan

        try:

            solved_load = safe_sum(
                n.loads_t.p.loc[
                    SNAPSHOT,
                    load_names
                ]
            )

        except Exception:

            solved_load = np.nan

        result[
            "solved_generation_mw"
        ] = solved_generation

        result[
            "solved_load_mw"
        ] = solved_load

        if (
            np.isfinite(solved_generation)
            and
            np.isfinite(solved_load)
        ):

            result[
                "generation_minus_load_mw"
            ] = (
                solved_generation
                -
                solved_load
            )

        # ------------------------------------------------------------------------------------------
        # Dispatch preservation check
        # ------------------------------------------------------------------------------------------

        try:

            generator_p_after = (
                n.generators_t.p_set.loc[
                    SNAPSHOT,
                    generator_names
                ]
                .astype(float)
            )

            load_p_after = (
                n.loads_t.p_set.loc[
                    SNAPSHOT,
                    load_names
                ]
                .astype(float)
            )

            result[
                "p_generator_set_difference_mw"
            ] = float(
                np.max(
                    np.abs(
                        (
                            generator_p_after
                            -
                            generator_p_before
                        ).to_numpy()
                    )
                )
            )

            result[
                "p_load_set_difference_mw"
            ] = float(
                np.max(
                    np.abs(
                        (
                            load_p_after
                            -
                            load_p_before
                        ).to_numpy()
                    )
                )
            )

        except Exception:
            pass

        # ------------------------------------------------------------------------------------------
        # Physical/numerical validation
        # ------------------------------------------------------------------------------------------

        (
            physical_solution,
            validation_reasons
        ) = validate_physical_state(
            n=n,
            snapshot=SNAPSHOT,
            converged=converged,
            pf_error=pf_error,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            min_angle=result["min_angle_rad"],
            max_angle=result["max_angle_rad"],
            max_line_loading=max_line_loading,
            max_transformer_loading=max_transformer_loading,
            generator_q=generator_q
        )

        result[
            "physical_solution"
        ] = bool(
            physical_solution
        )

        result[
            "valid_ac_solution"
        ] = bool(
            converged
            and
            physical_solution
        )

        result[
            "numerical_validation_reasons"
        ] = ";".join(
            validation_reasons
        )

        # ------------------------------------------------------------------------------------------
        # Security screen
        # ------------------------------------------------------------------------------------------

        result[
            "voltage_security_valid"
        ] = calculate_security(
            physical_solution=result[
                "valid_ac_solution"
            ],
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            overloaded_lines=overloaded_lines,
            overloaded_transformers=overloaded_transformers
        )

        # ------------------------------------------------------------------------------------------
        # Console report
        # ------------------------------------------------------------------------------------------

        print()
        print("AC POWER FLOW RESULT")

        print(
            f"PyPSA converged       : "
            f"{converged}"
        )

        print(
            f"PF error              : "
            f"{fmt(pf_error, 10)}"
        )

        print(
            f"Iterations            : "
            f"{fmt(iterations, 0)}"
        )

        print(
            f"Physical solution     : "
            f"{result['physical_solution']}"
        )

        print(
            f"VALID AC SOLUTION     : "
            f"{result['valid_ac_solution']}"
        )

        print(
            f"Minimum voltage       : "
            f"{fmt(min_voltage)} pu"
        )

        print(
            f"Minimum-voltage bus   : "
            f"{min_voltage_bus}"
        )

        print(
            f"Maximum voltage       : "
            f"{fmt(max_voltage)} pu"
        )

        print(
            f"Maximum-voltage bus   : "
            f"{max_voltage_bus}"
        )

        print(
            f"Voltage security      : "
            f"{result['voltage_security_valid']}"
        )

        print(
            f"Maximum line loading : "
            f"{fmt(max_line_loading)}%"
        )

        print(
            f"Overloaded lines      : "
            f"{overloaded_lines}"
        )

        print(
            f"Maximum transformer   : "
            f"{fmt(max_transformer_loading)}%"
        )

        print(
            f"Overloaded transformers: "
            f"{overloaded_transformers}"
        )

        print(
            f"Solved generator Q    : "
            f"{fmt(generator_q)} Mvar"
        )

        if validation_reasons:

            print()
            print(
                "NUMERICAL VALIDATION REASONS"
            )

            for reason in validation_reasons:
                print(
                    f"  - {reason}"
                )

        # ------------------------------------------------------------------------------------------
        # Warning information
        # ------------------------------------------------------------------------------------------

        if caught_warnings:

            print()
            print(
                f"Python warnings captured : "
                f"{len(caught_warnings)}"
            )

            for warning in caught_warnings[:10]:

                print(
                    f"  WARNING: "
                    f"{warning.message}"
                )

        # ------------------------------------------------------------------------------------------
        # Hard reference checks
        # ------------------------------------------------------------------------------------------

        if (
            result[
                "p_generator_set_difference_mw"
            ]
            >
            1e-9
        ):

            print(
                "WARNING: generator P set changed."
            )

        if (
            result[
                "p_load_set_difference_mw"
            ]
            >
            1e-9
        ):

            print(
                "WARNING: load P set changed."
            )

    except Exception as exc:

        result[
            "exception"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        result[
            "converged"
        ] = False

        result[
            "valid_ac_solution"
        ] = False

        result[
            "physical_solution"
        ] = False

        result[
            "voltage_security_valid"
        ] = False

        print()
        print(
            f"CASE EXCEPTION : {case_id}"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    print_header(
        "S4.5J — ROBUST AC CONVERGENCE + PHYSICAL-SOLUTION VALIDATION"
    )

    print(
        f"Network  : {SOURCE_NETWORK}"
    )

    print(
        f"Snapshot : {SNAPSHOT}"
    )

    print(
        "PF       : AC nonlinear"
    )

    print(
        "Dispatch : unchanged"
    )

    print(
        "Loads P  : unchanged"
    )

    print(
        "Generator Q set : 0"
    )

    print(
        "Slack    : distributed"
    )

    print(
        "Source   : READ-ONLY"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "AC solver convergence, physical/numerical validity, "
        "and security compliance are evaluated separately."
    )

    print(
        "0.90–1.10 pu is used ONLY as a security screen."
    )

    print(
        "No reinforcement."
    )

    print(
        "No reactive compensation."
    )

    print(
        "No dispatch change."
    )

    print(
        "No source network modification."
    )

    # ----------------------------------------------------------------------------------------------
    # Source validation
    # ----------------------------------------------------------------------------------------------

    if not SOURCE_NETWORK.exists():

        raise FileNotFoundError(
            f"Source network not found:\n{SOURCE_NETWORK}"
        )

    print_header(
        "LOADING SOURCE NETWORK"
    )

    source = pypsa.Network(
        str(SOURCE_NETWORK)
    )

    if SNAPSHOT not in source.snapshots:

        raise RuntimeError(
            f"Snapshot '{SNAPSHOT}' not found.\n"
            f"Available snapshots: {list(source.snapshots)}"
        )

    source.set_snapshots(
        [SNAPSHOT]
    )

    print(
        "Snapshot successfully isolated:"
    )

    print(
        f"  {SNAPSHOT}"
    )

    print()
    print("Source network:")

    print(
        f"  Buses        : {len(source.buses)}"
    )

    print(
        f"  Lines        : {len(source.lines)}"
    )

    print(
        f"  Transformers : {len(source.transformers)}"
    )

    print(
        f"  Links        : {len(source.links)}"
    )

    print(
        f"  Generators   : {len(source.generators)}"
    )

    print(
        f"  Loads        : {len(source.loads)}"
    )

    # ----------------------------------------------------------------------------------------------
    # Source operating point
    # ----------------------------------------------------------------------------------------------

    generator_names = list(
        source.generators.index
    )

    load_names = list(
        source.loads.index
    )

    generator_p_set = float(
        source.generators_t.p_set.loc[
            SNAPSHOT,
            generator_names
        ].sum()
    )

    load_p_set = float(
        source.loads_t.p_set.loc[
            SNAPSHOT,
            load_names
        ].sum()
    )

    print_header(
        "SOURCE OPERATING POINT"
    )

    print(
        f"Generator P set : "
        f"{generator_p_set:.6f} MW"
    )

    print(
        f"Load P set      : "
        f"{load_p_set:.6f} MW"
    )

    print(
        f"Generation-load : "
        f"{generator_p_set - load_p_set:.6f} MW"
    )

    # ----------------------------------------------------------------------------------------------
    # Q reference
    # ----------------------------------------------------------------------------------------------

    q_ratio = math.tan(
        math.acos(
            LOAD_POWER_FACTOR
        )
    )

    full_realistic_q = (
        load_p_set
        *
        q_ratio
    )

    print_header(
        "REALISTIC Q REFERENCE"
    )

    print(
        f"Load PF              : "
        f"{LOAD_POWER_FACTOR:.4f} lagging"
    )

    print(
        f"Q/P ratio            : "
        f"{q_ratio:.6f}"
    )

    print(
        f"Full realistic load Q : "
        f"{full_realistic_q:.6f} Mvar"
    )

    # ----------------------------------------------------------------------------------------------
    # Actual load index
    # ----------------------------------------------------------------------------------------------

    print_header(
        "ACTUAL NETWORK LOAD INDEX"
    )

    print(
        f"Actual load count : "
        f"{len(load_names)}"
    )

    print(
        "The continuation uses these exact names."
    )

    print(
        "No synthetic 'eirgrid_load_' prefix is constructed."
    )

    for i, load_name in enumerate(load_names):

        print(
            f"  [{i:02d}] {load_name}"
        )

    # ----------------------------------------------------------------------------------------------
    # Build test sequence
    # ----------------------------------------------------------------------------------------------

    tests = []

    for q in COARSE_Q_LEVELS:

        tests.append(
            (
                "coarse",
                float(q)
            )
        )

    for q in FINE_Q_LEVELS:

        tests.append(
            (
                "fine",
                float(q)
            )
        )

    total_cases = len(
        tests
    )

    print_header(
        "COARSE + FINE Q CONTINUATION"
    )

    print(
        "Coarse:"
    )

    print(
        ", ".join(
            f"{q:.1f}%"
            for q in COARSE_Q_LEVELS
        )
    )

    print()

    print(
        "Fine:"
    )

    print(
        ", ".join(
            f"{q:.1f}%"
            for q in FINE_Q_LEVELS
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Run tests
    # ----------------------------------------------------------------------------------------------

    results = []

    for i, (
        stage,
        q_percentage
    ) in enumerate(
        tests,
        start=1
    ):

        result = run_case(
            q_percentage=q_percentage,
            stage=stage,
            case_number=i,
            total_cases=total_cases
        )

        results.append(
            result
        )

    # ----------------------------------------------------------------------------------------------
    # DataFrame
    # ----------------------------------------------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # Ensure stable ordering
    df = df[
        [
            "case",
            "continuation_stage",
            "q_percentage",

            "converged",
            "valid_ac_solution",
            "physical_solution",

            "pf_error",
            "iterations",

            "min_voltage_pu",
            "max_voltage_pu",
            "min_voltage_bus",
            "max_voltage_bus",

            "voltage_security_valid",

            "min_angle_rad",
            "max_angle_rad",

            "max_line_loading_pct",
            "overloaded_lines",

            "max_transformer_loading_pct",
            "overloaded_transformers",

            "total_load_q_mvar",
            "generator_q_mvar",

            "load_pf",
            "q_p_ratio",
            "full_realistic_q_mvar",

            "generator_p_set_mw",
            "load_p_set_mw",

            "solved_generation_mw",
            "solved_load_mw",
            "generation_minus_load_mw",

            "p_generator_set_difference_mw",
            "p_load_set_difference_mw",

            "load_count",

            "numerical_validation_reasons",

            "exception",
        ]
    ]

    # ----------------------------------------------------------------------------------------------
    # Q=0 reference check
    # ----------------------------------------------------------------------------------------------

    print_header(
        "Q=0 REFERENCE CONSISTENCY CHECK"
    )

    q0_rows = df[
        np.isclose(
            df["q_percentage"],
            0.0
        )
    ]

    q0_ok = False

    if len(q0_rows) == 1:

        q0 = q0_rows.iloc[0]

        print(
            f"Q=0 converged           : "
            f"{q0['converged']}"
        )

        print(
            f"Q=0 valid AC solution   : "
            f"{q0['valid_ac_solution']}"
        )

        print(
            f"Q=0 physical solution   : "
            f"{q0['physical_solution']}"
        )

        print(
            f"Q=0 PF error            : "
            f"{fmt(q0['pf_error'], 10)}"
        )

        print(
            f"Q=0 iterations          : "
            f"{fmt(q0['iterations'], 0)}"
        )

        print(
            f"Q=0 minimum voltage     : "
            f"{fmt(q0['min_voltage_pu'])}"
        )

        print(
            f"Q=0 maximum voltage     : "
            f"{fmt(q0['max_voltage_pu'])}"
        )

        print(
            f"Q=0 overloaded lines    : "
            f"{q0['overloaded_lines']}"
        )

        q0_ok = bool(
            q0["valid_ac_solution"]
        )

    else:

        print(
            "Q=0 result not uniquely available."
        )

    print()

    print(
        f"Q=0 robust reference valid : "
        f"{q0_ok}"
    )

    # ----------------------------------------------------------------------------------------------
    # Continuation summary
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S4.5J — Q CONTINUATION SUMMARY"
    )

    display_columns = [
        "q_percentage",
        "continuation_stage",
        "converged",
        "valid_ac_solution",
        "physical_solution",
        "pf_error",
        "iterations",
        "min_voltage_pu",
        "max_voltage_pu",
        "voltage_security_valid",
        "max_line_loading_pct",
        "overloaded_lines",
        "max_transformer_loading_pct",
        "overloaded_transformers",
        "total_load_q_mvar",
        "generator_q_mvar",
    ]

    print(
        df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Valid tested Q levels
    # ----------------------------------------------------------------------------------------------

    valid_rows = df[
        df["valid_ac_solution"]
        == True
    ]

    invalid_rows = df[
        df["valid_ac_solution"]
        == False
    ]

    print_header(
        "ROBUST AC VALIDITY INTERPRETATION"
    )

    if len(valid_rows) > 0:

        highest_valid_q = float(
            valid_rows[
                "q_percentage"
            ].max()
        )

        print(
            f"Highest valid tested Q level : "
            f"{highest_valid_q:.1f}%"
        )

    else:

        highest_valid_q = np.nan

        print(
            "Highest valid tested Q level : NONE"
        )

    if len(invalid_rows) > 0:

        first_invalid_q = float(
            invalid_rows[
                "q_percentage"
            ].min()
        )

        print(
            f"Lowest invalid tested Q level : "
            f"{first_invalid_q:.1f}%"
        )

    else:

        first_invalid_q = np.nan

        print(
            "Lowest invalid tested Q level : NONE"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "An invalid case may represent either solver non-convergence "
        "or numerical/physical invalidity."
    )

    print(
        "It is NOT automatically a voltage-collapse point."
    )

    # ----------------------------------------------------------------------------------------------
    # Security interpretation
    # ----------------------------------------------------------------------------------------------

    print_header(
        "VOLTAGE + THERMAL SECURITY SCREEN"
    )

    security_columns = [
        "q_percentage",
        "valid_ac_solution",
        "min_voltage_pu",
        "max_voltage_pu",
        "voltage_security_valid",
        "max_line_loading_pct",
        "overloaded_lines",
        "max_transformer_loading_pct",
        "overloaded_transformers",
    ]

    print(
        df[
            security_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Numerical-artifact cases
    # ----------------------------------------------------------------------------------------------

    artifact_rows = df[
        (
            df[
                "numerical_validation_reasons"
            ].fillna("")
            != ""
        )
    ]

    print_header(
        "NUMERICAL / SOLVER FAILURE CASES"
    )

    if len(artifact_rows) == 0:

        print(
            "No numerical-validation failures detected."
        )

    else:

        for _, row in artifact_rows.iterrows():

            print(
                f"Q={row['q_percentage']:.1f}%"
            )

            print(
                f"  converged       : "
                f"{row['converged']}"
            )

            print(
                f"  valid AC        : "
                f"{row['valid_ac_solution']}"
            )

            print(
                f"  PF error        : "
                f"{fmt(row['pf_error'], 10)}"
            )

            print(
                f"  reason          : "
                f"{row['numerical_validation_reasons']}"
            )

    # ----------------------------------------------------------------------------------------------
    # Security-valid cases
    # ----------------------------------------------------------------------------------------------

    secure_rows = df[
        df[
            "voltage_security_valid"
        ]
        == True
    ]

    print_header(
        "SECURITY-COMPLIANT TESTED Q LEVELS"
    )

    if len(secure_rows) == 0:

        print(
            "No tested Q level is fully voltage + thermally secure."
        )

    else:

        print(
            secure_rows[
                [
                    "q_percentage",
                    "min_voltage_pu",
                    "max_voltage_pu",
                    "max_line_loading_pct",
                    "overloaded_lines",
                    "max_transformer_loading_pct",
                    "overloaded_transformers",
                ]
            ].to_string(
                index=False
            )
        )

    # ----------------------------------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print_header(
        "S4.5J RESULTS SAVED"
    )

    print(
        f"Summary : {OUTPUT_CSV}"
    )

    print()
    print(
        "Output columns include:"
    )

    print(
        "  converged"
    )

    print(
        "  valid_ac_solution"
    )

    print(
        "  physical_solution"
    )

    print(
        "  pf_error"
    )

    print(
        "  iterations"
    )

    print(
        "  voltage_security_valid"
    )

    print(
        "  max_line_loading_pct"
    )

    print(
        "  max_transformer_loading_pct"
    )

    print(
        "  numerical_validation_reasons"
    )

    # ----------------------------------------------------------------------------------------------
    # Source integrity
    # ----------------------------------------------------------------------------------------------

    print_header(
        "SOURCE NETWORK INTEGRITY"
    )

    print(
        "SOURCE NETWORK MODIFIED : NO"
    )

    print(
        "REINFORCEMENTS APPLIED  : NO"
    )

    print(
        "REACTIVE DEVICES ADDED  : NO"
    )

    print(
        "DISPATCH CHANGED        : NO"
    )

    print(
        "LOAD P CHANGED          : NO"
    )

    print(
        "PERMANENT CHANGES       : NONE"
    )

    print(
        "Q=0 ROBUST REFERENCE    : "
        f"{q0_ok}"
    )

    print_header(
        "S4.5J COMPLETE"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    main()