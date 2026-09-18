# ==================================================================================================
# S5.1 — STAGE 5 BASELINE CONFIRMATION
# ==================================================================================================
#
# Purpose
# -------
# Establish the authoritative Stage 5 baseline operating condition before
# any Stage 5 reinforcement / voltage-support experiments.
#
# IMPORTANT CORRECTION
# --------------------
# Earlier S5.1 output incorrectly reported:
#
#   PyPSA converged : False
#   PF error        : nan
#   Valid AC solution : False
#
# even though the RAW PyPSA result showed:
#
#   converged = True
#   error     = 4.042721e-11
#   n_iter    = 7
#
# This version uses robust extraction of PyPSA's DataFrame-based PF result.
#
# S5.1 therefore separates:
#
#   1. SOLVER CONVERGENCE
#   2. PHYSICAL / NUMERICAL VALIDITY
#   3. VOLTAGE SECURITY
#   4. THERMAL SECURITY
#
# IMPORTANT:
#
#   AC validity != voltage security
#
# A converged AC solution with:
#
#   Vmin = 0.738929 pu
#
# is a valid solved AC state, but it is NOT voltage secure.
#
# No reinforcement.
# No reactive compensation.
# No dispatch change.
# No load modification.
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

PROJECT_ROOT = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser"
)

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
    / "s5_1_stage5_baseline.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"


# ==================================================================================================
# SECURITY LIMITS
# ==================================================================================================

VOLTAGE_SECURITY_MIN = 0.90
VOLTAGE_SECURITY_MAX = 1.10

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0


# ==================================================================================================
# NUMERICAL VALIDATION LIMITS
# ==================================================================================================
#
# These are NOT security limits.
#
# They only reject obviously nonsensical numerical states.
#
# A voltage such as 0.7389 pu is NOT rejected here.
# It is a physically meaningful but insecure voltage.
#
# ==================================================================================================

HARD_VOLTAGE_MIN = 0.20
HARD_VOLTAGE_MAX = 2.00

HARD_ANGLE_ABS_MAX_RAD = 2.0 * math.pi

HARD_LOADING_MAX_PCT = 100000.0

HARD_GENERATOR_Q_ABS_MAX_MVAR = 1.0e6


# ==================================================================================================
# SOLVER ACCEPTANCE
# ==================================================================================================
#
# PyPSA nonlinear PF error should normally be very small.
#
# The observed S5.1 result:
#
#   4.042721e-11
#
# is comfortably below this threshold.
#
# ==================================================================================================

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
# ROBUST PYPSA RESULT EXTRACTION
# ==================================================================================================

def extract_snapshot_value(obj, snapshot):
    """
    Robustly extract a scalar value from PyPSA's PF result.

    PyPSA may return:
        - DataFrame
        - Series
        - scalar
        - numpy scalar

    IMPORTANT:
    For DataFrame results such as:

                      0    1    2
        S2_PEAK_DEMAND  7    0    0

    the snapshot is stored in the INDEX.

    For convergence:
        True / True / True

    any False value means the corresponding subnetwork did not converge.

    For numerical error:
        the WORST / maximum finite error is used.
    """

    if obj is None:
        return np.nan

    try:

        # ------------------------------------------------------------------------------------------
        # DataFrame
        # ------------------------------------------------------------------------------------------

        if isinstance(obj, pd.DataFrame):

            # Snapshot in index
            if snapshot in obj.index:

                row = obj.loc[snapshot]

                if isinstance(row, pd.Series):

                    # --------------------------------------------------
                    # Boolean result
                    # --------------------------------------------------

                    if all(
                        isinstance(v, (bool, np.bool_))
                        for v in row.dropna().tolist()
                    ):

                        values = row.dropna().astype(bool)

                        # For convergence, all subnetworks must converge.
                        return bool(values.all())

                    # --------------------------------------------------
                    # Numeric result
                    # --------------------------------------------------

                    vals = pd.to_numeric(
                        row,
                        errors="coerce"
                    ).dropna()

                    if len(vals) == 0:
                        return np.nan

                    if len(vals) == 1:
                        return vals.iloc[0]

                    # For error:
                    # use worst value.
                    return vals.max()

            # Snapshot in columns
            if snapshot in obj.columns:

                col = obj[snapshot]

                # Boolean column
                if all(
                    isinstance(v, (bool, np.bool_))
                    for v in col.dropna().tolist()
                ):

                    values = col.dropna().astype(bool)

                    return bool(values.all())

                vals = pd.to_numeric(
                    col,
                    errors="coerce"
                ).dropna()

                if len(vals) == 0:
                    return np.nan

                if len(vals) == 1:
                    return vals.iloc[0]

                return vals.max()

        # ------------------------------------------------------------------------------------------
        # Series
        # ------------------------------------------------------------------------------------------

        if isinstance(obj, pd.Series):

            if snapshot in obj.index:

                value = obj.loc[snapshot]

                if isinstance(
                    value,
                    (bool, np.bool_)
                ):

                    return bool(value)

                if np.isscalar(value):

                    return value

                vals = pd.to_numeric(
                    pd.Series(value),
                    errors="coerce"
                ).dropna()

                if len(vals) == 0:
                    return np.nan

                return vals.max()

            # Boolean Series
            if all(
                isinstance(v, (bool, np.bool_))
                for v in obj.dropna().tolist()
            ):

                return bool(
                    obj.dropna().astype(bool).all()
                )

            vals = pd.to_numeric(
                obj,
                errors="coerce"
            ).dropna()

            if len(vals) == 0:
                return np.nan

            if len(vals) == 1:
                return vals.iloc[0]

            return vals.max()

        # ------------------------------------------------------------------------------------------
        # Scalar
        # ------------------------------------------------------------------------------------------

        if isinstance(
            obj,
            (bool, np.bool_)
        ):

            return bool(obj)

        if np.isscalar(obj):

            return obj

    except Exception:

        pass

    return np.nan


def extract_convergence(result, snapshot):
    """
    Extract:
        converged
        PF error
        iterations

    from PyPSA's nonlinear PF result.
    """

    converged = False
    pf_error = np.nan
    iterations = np.nan

    if result is None:
        return converged, pf_error, iterations

    # ----------------------------------------------------------------------------------------------
    # CONVERGENCE
    # ----------------------------------------------------------------------------------------------

    try:

        if "converged" in result:

            raw_converged = result["converged"]

            value = extract_snapshot_value(
                raw_converged,
                snapshot
            )

            if isinstance(
                value,
                (bool, np.bool_)
            ):

                converged = bool(value)

            elif np.isfinite(value):

                converged = bool(value)

    except Exception:

        converged = False

    # ----------------------------------------------------------------------------------------------
    # PF ERROR
    # ----------------------------------------------------------------------------------------------

    try:

        if "error" in result:

            raw_error = result["error"]

            value = extract_snapshot_value(
                raw_error,
                snapshot
            )

            try:

                pf_error = float(value)

            except Exception:

                pf_error = np.nan

    except Exception:

        pf_error = np.nan

    # ----------------------------------------------------------------------------------------------
    # ITERATIONS
    # ----------------------------------------------------------------------------------------------

    try:

        if "n_iter" in result:

            raw_iterations = result["n_iter"]

            value = extract_snapshot_value(
                raw_iterations,
                snapshot
            )

            try:

                iterations = float(value)

            except Exception:

                iterations = np.nan

    except Exception:

        iterations = np.nan

    return (
        converged,
        pf_error,
        iterations
    )


# ==================================================================================================
# NUMERIC HELPERS
# ==================================================================================================

def finite_values(values):

    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    return arr[
        np.isfinite(arr)
    ]


def safe_sum(values):

    vals = finite_values(values)

    if len(vals) == 0:
        return np.nan

    return float(
        vals.sum()
    )


# ==================================================================================================
# LINE THERMAL CHECK
# ==================================================================================================

def calculate_line_loading(
    n,
    snapshot
):

    if len(n.lines.index) == 0:

        return np.nan, 0

    line_names = list(
        n.lines.index
    )

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
            p0 ** 2
            +
            q0 ** 2
        )

        s1 = np.sqrt(
            p1 ** 2
            +
            q1 ** 2
        )

        s_max = pd.concat(
            [s0, s1],
            axis=1
        ).max(
            axis=1
        )

        ratings = (
            n.lines.s_nom
            .astype(float)
        )

        loading_pct = (
            s_max
            /
            ratings
            *
            100.0
        )

        finite = loading_pct[
            np.isfinite(
                loading_pct
            )
        ]

        if len(finite) == 0:

            return np.nan, 0

        max_loading = float(
            finite.max()
        )

        overloaded = int(
            (
                finite
                >
                LINE_LOADING_LIMIT_PCT
            ).sum()
        )

        return (
            max_loading,
            overloaded
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


# ==================================================================================================
# TRANSFORMER THERMAL CHECK
# ==================================================================================================

def calculate_transformer_loading(
    n,
    snapshot
):

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
            p0 ** 2
            +
            q0 ** 2
        )

        s1 = np.sqrt(
            p1 ** 2
            +
            q1 ** 2
        )

        s_max = pd.concat(
            [s0, s1],
            axis=1
        ).max(
            axis=1
        )

        ratings = (
            n.transformers.s_nom
            .astype(float)
        )

        loading_pct = (
            s_max
            /
            ratings
            *
            100.0
        )

        finite = loading_pct[
            np.isfinite(
                loading_pct
            )
        ]

        if len(finite) == 0:

            return np.nan, 0

        max_loading = float(
            finite.max()
        )

        overloaded = int(
            (
                finite
                >
                TRANSFORMER_LOADING_LIMIT_PCT
            ).sum()
        )

        return (
            max_loading,
            overloaded
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


# ==================================================================================================
# PHYSICAL / NUMERICAL VALIDATION
# ==================================================================================================

def validate_physical_state(
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

    reasons = []

    # ----------------------------------------------------------------------------------------------
    # Solver convergence
    # ----------------------------------------------------------------------------------------------

    if not converged:

        reasons.append(
            "NON_CONVERGED"
        )

    # ----------------------------------------------------------------------------------------------
    # PF error
    # ----------------------------------------------------------------------------------------------

    if not np.isfinite(pf_error):

        reasons.append(
            "PF_ERROR_NOT_FINITE"
        )

    elif pf_error > PF_ERROR_TOLERANCE:

        reasons.append(
            f"PF_ERROR_ABOVE_TOLERANCE:{pf_error:.6g}"
        )

    # ----------------------------------------------------------------------------------------------
    # Voltage
    # ----------------------------------------------------------------------------------------------

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

    # ----------------------------------------------------------------------------------------------
    # Voltage angles
    # ----------------------------------------------------------------------------------------------

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

    # ----------------------------------------------------------------------------------------------
    # Line loading
    # ----------------------------------------------------------------------------------------------

    if not np.isfinite(max_line_loading):

        reasons.append(
            "LINE_LOADING_NOT_FINITE"
        )

    elif max_line_loading > HARD_LOADING_MAX_PCT:

        reasons.append(
            f"LINE_LOADING_NUMERICALLY_IMPLAUSIBLE:{max_line_loading:.6g}"
        )

    # ----------------------------------------------------------------------------------------------
    # Transformer loading
    # ----------------------------------------------------------------------------------------------

    if not np.isfinite(max_transformer_loading):

        reasons.append(
            "TRANSFORMER_LOADING_NOT_FINITE"
        )

    elif max_transformer_loading > HARD_LOADING_MAX_PCT:

        reasons.append(
            "TRANSFORMER_LOADING_NUMERICALLY_IMPLAUSIBLE:"
            f"{max_transformer_loading:.6g}"
        )

    # ----------------------------------------------------------------------------------------------
    # Generator Q
    # ----------------------------------------------------------------------------------------------

    if not np.isfinite(generator_q):

        reasons.append(
            "GENERATOR_Q_NOT_FINITE"
        )

    elif abs(generator_q) > HARD_GENERATOR_Q_ABS_MAX_MVAR:

        reasons.append(
            f"GENERATOR_Q_NUMERICALLY_IMPLAUSIBLE:{generator_q:.6g}"
        )

    physical_solution = (
        len(reasons) == 0
    )

    return (
        physical_solution,
        reasons
    )


# ==================================================================================================
# SECURITY CHECKS
# ==================================================================================================

def voltage_security_check(
    min_voltage,
    max_voltage
):

    if not np.isfinite(min_voltage):
        return False

    if not np.isfinite(max_voltage):
        return False

    return bool(
        min_voltage >= VOLTAGE_SECURITY_MIN
        and
        max_voltage <= VOLTAGE_SECURITY_MAX
    )


def thermal_security_check(
    overloaded_lines,
    overloaded_transformers
):

    if not np.isfinite(
        overloaded_lines
    ):

        return False

    if not np.isfinite(
        overloaded_transformers
    ):

        return False

    return bool(
        overloaded_lines == 0
        and
        overloaded_transformers == 0
    )


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    print_header(
        "S5.1 — STAGE 5 BASELINE CONFIRMATION"
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
        "Slack    : distributed"
    )

    print(
        "Dispatch : unchanged"
    )

    print(
        "Loads    : unchanged"
    )

    print(
        "Source   : READ-ONLY"
    )

    # ----------------------------------------------------------------------------------------------
    # SOURCE VALIDATION
    # ----------------------------------------------------------------------------------------------

    if not SOURCE_NETWORK.exists():

        raise FileNotFoundError(
            f"Source network not found:\n{SOURCE_NETWORK}"
        )

    print_header(
        "LOADING SOURCE NETWORK"
    )

    n = pypsa.Network(
        str(SOURCE_NETWORK)
    )

    if SNAPSHOT not in n.snapshots:

        raise RuntimeError(
            f"Snapshot '{SNAPSHOT}' not found.\n"
            f"Available snapshots: {list(n.snapshots)}"
        )

    n.set_snapshots(
        [SNAPSHOT]
    )

    print(
        "Snapshot successfully isolated:"
    )

    print(
        f"  {SNAPSHOT}"
    )

    print()

    print(
        "SOURCE NETWORK"
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

    # ----------------------------------------------------------------------------------------------
    # OPERATING POINT
    # ----------------------------------------------------------------------------------------------

    generator_names = list(
        n.generators.index
    )

    load_names = list(
        n.loads.index
    )

    if len(generator_names) == 0:

        raise RuntimeError(
            "Network contains no generators."
        )

    if len(load_names) == 0:

        raise RuntimeError(
            "Network contains no loads."
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

    generation_minus_load = (
        generator_p_set
        -
        load_p_set
    )

    print_header(
        "OPERATING POINT"
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
        f"{generation_minus_load:.6f} MW"
    )

    # ----------------------------------------------------------------------------------------------
    # P REFERENCES
    # ----------------------------------------------------------------------------------------------

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

    # ----------------------------------------------------------------------------------------------
    # AC POWER FLOW
    # ----------------------------------------------------------------------------------------------

    print_header(
        "AC POWER FLOW"
    )

    print(
        "Running PyPSA nonlinear AC power flow..."
    )

    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:

        warnings.simplefilter(
            "always"
        )

        pf_result = n.pf(
            snapshots=[SNAPSHOT],
            distribute_slack=True
        )

    # ----------------------------------------------------------------------------------------------
    # RAW RESULT
    # ----------------------------------------------------------------------------------------------

    print()

    print(
        "Raw PyPSA PF result:"
    )

    print(
        pf_result
    )

    # ----------------------------------------------------------------------------------------------
    # ROBUST CONVERGENCE EXTRACTION
    # ----------------------------------------------------------------------------------------------

    (
        converged,
        pf_error,
        iterations
    ) = extract_convergence(
        pf_result,
        SNAPSHOT
    )

    # ----------------------------------------------------------------------------------------------
    # VOLTAGE
    # ----------------------------------------------------------------------------------------------

    print_header(
        "VOLTAGE CHECK"
    )

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

    if len(finite_v) == 0:

        min_voltage = np.nan
        max_voltage = np.nan
        min_voltage_bus = None
        max_voltage_bus = None

    else:

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

    under_voltage_buses = int(
        (
            finite_v
            <
            VOLTAGE_SECURITY_MIN
        ).sum()
    )

    over_voltage_buses = int(
        (
            finite_v
            >
            VOLTAGE_SECURITY_MAX
        ).sum()
    )

    print(
        f"Minimum voltage : "
        f"{fmt(min_voltage)} pu"
    )

    print(
        f"Minimum bus     : "
        f"{min_voltage_bus}"
    )

    print(
        f"Maximum voltage : "
        f"{fmt(max_voltage)} pu"
    )

    print(
        f"Maximum bus     : "
        f"{max_voltage_bus}"
    )

    print()

    print(
        f"Under-voltage buses (< {VOLTAGE_SECURITY_MIN:.2f} pu) : "
        f"{under_voltage_buses}"
    )

    print(
        f"Over-voltage buses  (> {VOLTAGE_SECURITY_MAX:.2f} pu) : "
        f"{over_voltage_buses}"
    )

    print()

    print(
        "Worst under-voltage buses:"
    )

    print(
        finite_v.sort_values().head(
            10
        ).to_string()
    )

    # ----------------------------------------------------------------------------------------------
    # LINE THERMAL
    # ----------------------------------------------------------------------------------------------

    print_header(
        "LINE THERMAL CHECK"
    )

    (
        max_line_loading,
        overloaded_lines
    ) = calculate_line_loading(
        n,
        SNAPSHOT
    )

    print(
        f"Maximum line loading : "
        f"{fmt(max_line_loading)}%"
    )

    print(
        f"Overloaded lines     : "
        f"{overloaded_lines}"
    )

    try:

        line_names = list(
            n.lines.index
        )

        p0 = n.lines_t.p0.loc[
            SNAPSHOT,
            line_names
        ].astype(float)

        q0 = n.lines_t.q0.loc[
            SNAPSHOT,
            line_names
        ].astype(float)

        p1 = n.lines_t.p1.loc[
            SNAPSHOT,
            line_names
        ].astype(float)

        q1 = n.lines_t.q1.loc[
            SNAPSHOT,
            line_names
        ].astype(float)

        s0 = np.sqrt(
            p0 ** 2
            +
            q0 ** 2
        )

        s1 = np.sqrt(
            p1 ** 2
            +
            q1 ** 2
        )

        s_max = pd.concat(
            [s0, s1],
            axis=1
        ).max(
            axis=1
        )

        ratings = (
            n.lines.s_nom
            .astype(float)
        )

        loading = (
            s_max
            /
            ratings
            *
            100.0
        )

        print()

        print(
            "Top 10 line loadings:"
        )

        print(
            loading.sort_values(
                ascending=False
            ).head(
                10
            ).to_string()
        )

    except Exception:

        pass

    # ----------------------------------------------------------------------------------------------
    # TRANSFORMER THERMAL
    # ----------------------------------------------------------------------------------------------

    print_header(
        "TRANSFORMER THERMAL CHECK"
    )

    (
        max_transformer_loading,
        overloaded_transformers
    ) = calculate_transformer_loading(
        n,
        SNAPSHOT
    )

    print(
        f"Maximum transformer loading : "
        f"{fmt(max_transformer_loading)}%"
    )

    print(
        f"Overloaded transformers     : "
        f"{overloaded_transformers}"
    )

    try:

        transformer_names = list(
            n.transformers.index
        )

        p0 = n.transformers_t.p0.loc[
            SNAPSHOT,
            transformer_names
        ].astype(float)

        q0 = n.transformers_t.q0.loc[
            SNAPSHOT,
            transformer_names
        ].astype(float)

        p1 = n.transformers_t.p1.loc[
            SNAPSHOT,
            transformer_names
        ].astype(float)

        q1 = n.transformers_t.q1.loc[
            SNAPSHOT,
            transformer_names
        ].astype(float)

        s0 = np.sqrt(
            p0 ** 2
            +
            q0 ** 2
        )

        s1 = np.sqrt(
            p1 ** 2
            +
            q1 ** 2
        )

        s_max = pd.concat(
            [s0, s1],
            axis=1
        ).max(
            axis=1
        )

        ratings = (
            n.transformers.s_nom
            .astype(float)
        )

        transformer_loading = (
            s_max
            /
            ratings
            *
            100.0
        )

        print()

        print(
            "Transformer loadings:"
        )

        print(
            transformer_loading
            .sort_values(
                ascending=False
            )
            .to_string()
        )

    except Exception:

        pass

    # ----------------------------------------------------------------------------------------------
    # GENERATOR Q
    # ----------------------------------------------------------------------------------------------

    print_header(
        "GENERATOR REACTIVE POWER"
    )

    try:

        generator_q = safe_sum(
            n.generators_t.q.loc[
                SNAPSHOT,
                generator_names
            ]
        )

    except Exception:

        generator_q = np.nan

    print(
        f"Total generator Q : "
        f"{fmt(generator_q)} Mvar"
    )

    # ----------------------------------------------------------------------------------------------
    # VOLTAGE ANGLES
    # ----------------------------------------------------------------------------------------------

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

            min_angle = float(
                finite_angles.min()
            )

            max_angle = float(
                finite_angles.max()
            )

        else:

            min_angle = np.nan
            max_angle = np.nan

    except Exception:

        min_angle = np.nan
        max_angle = np.nan

    # ----------------------------------------------------------------------------------------------
    # SOLVED P
    # ----------------------------------------------------------------------------------------------

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

    if (
        np.isfinite(solved_generation)
        and
        np.isfinite(solved_load)
    ):

        solved_generation_minus_load = (
            solved_generation
            -
            solved_load
        )

    else:

        solved_generation_minus_load = np.nan

    # ----------------------------------------------------------------------------------------------
    # DISPATCH INTEGRITY
    # ----------------------------------------------------------------------------------------------

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

    generator_p_difference = float(
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

    load_p_difference = float(
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

    dispatch_unchanged = bool(
        generator_p_difference <= 1e-9
        and
        load_p_difference <= 1e-9
    )

    # ----------------------------------------------------------------------------------------------
    # PHYSICAL / NUMERICAL VALIDATION
    # ----------------------------------------------------------------------------------------------

    (
        physical_solution,
        validation_reasons
    ) = validate_physical_state(
        converged=converged,
        pf_error=pf_error,
        min_voltage=min_voltage,
        max_voltage=max_voltage,
        min_angle=min_angle,
        max_angle=max_angle,
        max_line_loading=max_line_loading,
        max_transformer_loading=max_transformer_loading,
        generator_q=generator_q
    )

    valid_ac_solution = bool(
        converged
        and
        physical_solution
    )

    # ----------------------------------------------------------------------------------------------
    # SECURITY
    # ----------------------------------------------------------------------------------------------

    voltage_security = voltage_security_check(
        min_voltage,
        max_voltage
    )

    thermal_security = thermal_security_check(
        overloaded_lines,
        overloaded_transformers
    )

    overall_security = bool(
        valid_ac_solution
        and
        voltage_security
        and
        thermal_security
    )

    # ----------------------------------------------------------------------------------------------
    # ROBUST VALIDITY REPORT
    # ----------------------------------------------------------------------------------------------

    print_header(
        "ROBUST VALIDITY"
    )

    print(
        f"PyPSA converged         : "
        f"{converged}"
    )

    print(
        f"PF error                : "
        f"{fmt(pf_error, 12)}"
    )

    print(
        f"Iterations              : "
        f"{fmt(iterations, 0)}"
    )

    print(
        f"Finite voltages         : "
        f"{len(finite_v) == len(n.buses)}"
    )

    print(
        f"Physical voltage state  : "
        f"{np.isfinite(min_voltage) and np.isfinite(max_voltage)}"
    )

    print(
        f"PF residual valid       : "
        f"{np.isfinite(pf_error) and pf_error <= PF_ERROR_TOLERANCE}"
    )

    print(
        f"Physical solution       : "
        f"{physical_solution}"
    )

    print(
        f"Valid AC solution       : "
        f"{valid_ac_solution}"
    )

    if validation_reasons:

        print()

        print(
            "Validation reasons:"
        )

        for reason in validation_reasons:

            print(
                f"  - {reason}"
            )

    else:

        print()

        print(
            "Validation reasons      : NONE"
        )

    # ----------------------------------------------------------------------------------------------
    # SECURITY
    # ----------------------------------------------------------------------------------------------

    print_header(
        "SECURITY"
    )

    print(
        f"Voltage security        : "
        f"{voltage_security}"
    )

    print(
        f"Thermal security        : "
        f"{thermal_security}"
    )

    print(
        f"Overall security        : "
        f"{overall_security}"
    )

    # ----------------------------------------------------------------------------------------------
    # DISPATCH INTEGRITY
    # ----------------------------------------------------------------------------------------------

    print_header(
        "DISPATCH INTEGRITY"
    )

    print(
        f"Generator P set change : "
        f"{generator_p_difference:.12f} MW"
    )

    print(
        f"Load P set change      : "
        f"{load_p_difference:.12f} MW"
    )

    print(
        f"Dispatch unchanged     : "
        f"{dispatch_unchanged}"
    )

    # ----------------------------------------------------------------------------------------------
    # RESULT DATA
    # ----------------------------------------------------------------------------------------------

    result = {

        "snapshot": SNAPSHOT,

        "source_network": str(
            SOURCE_NETWORK
        ),

        "buses": len(n.buses),
        "lines": len(n.lines),
        "transformers": len(n.transformers),
        "links": len(n.links),
        "generators": len(n.generators),
        "loads": len(n.loads),

        "generator_p_set_mw":
            generator_p_set,

        "load_p_set_mw":
            load_p_set,

        "generation_minus_load_mw":
            generation_minus_load,

        "converged":
            converged,

        "pf_error":
            pf_error,

        "iterations":
            iterations,

        "physical_solution":
            physical_solution,

        "valid_ac_solution":
            valid_ac_solution,

        "min_voltage_pu":
            min_voltage,

        "max_voltage_pu":
            max_voltage,

        "min_voltage_bus":
            min_voltage_bus,

        "max_voltage_bus":
            max_voltage_bus,

        "undervoltage_buses":
            under_voltage_buses,

        "overvoltage_buses":
            over_voltage_buses,

        "min_angle_rad":
            min_angle,

        "max_angle_rad":
            max_angle,

        "max_line_loading_pct":
            max_line_loading,

        "overloaded_lines":
            overloaded_lines,

        "max_transformer_loading_pct":
            max_transformer_loading,

        "overloaded_transformers":
            overloaded_transformers,

        "generator_q_mvar":
            generator_q,

        "solved_generation_mw":
            solved_generation,

        "solved_load_mw":
            solved_load,

        "solved_generation_minus_load_mw":
            solved_generation_minus_load,

        "generator_p_set_change_mw":
            generator_p_difference,

        "load_p_set_change_mw":
            load_p_difference,

        "dispatch_unchanged":
            dispatch_unchanged,

        "voltage_security":
            voltage_security,

        "thermal_security":
            thermal_security,

        "overall_security":
            overall_security,

        "numerical_validation_reasons":
            ";".join(
                validation_reasons
            ),
    }

    df = pd.DataFrame(
        [result]
    )

    # ----------------------------------------------------------------------------------------------
    # SAVE
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
        "S5.1 BASELINE SAVED"
    )

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    # ----------------------------------------------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.1 FINAL STATUS"
    )

    print(
        f"AC BASELINE : "
        f"{'PASS' if valid_ac_solution else 'FAIL'}"
    )

    print(
        f"VOLTAGE SECURITY : "
        f"{'PASS' if voltage_security else 'FAIL'}"
    )

    print(
        f"THERMAL SECURITY : "
        f"{'PASS' if thermal_security else 'FAIL'}"
    )

    print(
        f"OVERALL BASELINE SECURITY : "
        f"{'PASS' if overall_security else 'FAIL'}"
    )

    print()

    print(
        "SOURCE NETWORK MODIFIED : NO"
    )

    print(
        "REINFORCEMENTS APPLIED   : NO"
    )

    print(
        "REACTIVE SUPPORT ADDED   : NO"
    )

    print(
        "DISPATCH CHANGED         : NO"
    )

    print(
        "PERMANENT CHANGES        : NONE"
    )

    # ----------------------------------------------------------------------------------------------
    # WARNINGS
    # ----------------------------------------------------------------------------------------------

    if caught_warnings:

        print()

        print(
            f"Python warnings captured : "
            f"{len(caught_warnings)}"
        )

        for warning in caught_warnings[:10]:

            print(
                f"  WARNING: {warning.message}"
            )

    print_header(
        "S5.1 COMPLETE"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":

    main()