# ==================================================================================================
# S5.4 — JOINT VOLTAGE + THERMAL SECURITY EVALUATION
# ==================================================================================================
#
# Purpose
# -------
# S5.2 established that 300 MVAr local reactive support restores voltage security.
#
# S5.3 established the thermal reinforcement required to eliminate the remaining
# transmission-line overloads while retaining the fixed 300 MVAr operating point.
#
# S5.4 combines BOTH temporary controls:
#
#       1. Local reactive support
#              Q support = 300 MVAr
#
#       2. Temporary transmission thermal reinforcement
#              line s_nom multiplier
#
# The experiment asks:
#
#   "Under the voltage-secure 300 MVAr condition, what thermal reinforcement
#    multiplier is required to achieve complete voltage + thermal security?"
#
# IMPORTANT
# ---------
# Every case starts from a fresh in-memory copy of the SOURCE_NETWORK.
#
# The source .nc file is READ-ONLY.
#
# No:
#   - generator dispatch optimisation
#   - load shedding
#   - topology changes
#   - permanent reinforcement
#   - source-network modification
#
# is performed.
#
# ==================================================================================================

from pathlib import Path
import warnings
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
    / "s5_4_joint_security.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

# Exact S5.2 weak/support bus used by the supplied S5.3 script.
WEAK_BUS = "way/104388595-220"

# ==================================================================================================
# FIXED S5.2 OPERATING CONDITION
# ==================================================================================================

Q_SUPPORT_MVAR = 300.0

VOLTAGE_SECURITY_MIN = 0.90
VOLTAGE_SECURITY_MAX = 1.10

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0

# ==================================================================================================
# THERMAL SWEEP
# ==================================================================================================

THERMAL_MULTIPLIERS = [
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.75,
    2.00,
]

# ==================================================================================================
# S5.2 REFERENCE FINGERPRINT
# ==================================================================================================
#
# These values come directly from the supplied S5.3 script.
#
# At Q = 300 MVAr and NO thermal reinforcement:
#
#   Vmin              = 0.917237 pu
#   UV buses          = 0
#   Max line loading  = 166.234091 %
#   Overloaded lines  = 9
#   Max transformer   = 33.071461 %
#   Overloaded trafo  = 0
#
# ==================================================================================================

REFERENCE_VMIN = 0.917237
REFERENCE_UV_COUNT = 0
REFERENCE_MAX_LINE_LOADING = 166.234091
REFERENCE_OVERLOADED_LINES = 9

REFERENCE_MAX_TRANSFORMER_LOADING = 33.071461
REFERENCE_OVERLOADED_TRANSFORMERS = 0

REFERENCE_PF_ERROR = 1.073222e-08

VMIN_TOL = 5e-5
LINE_LOADING_TOL = 0.05
TRANSFORMER_LOADING_TOL = 0.05
PF_ERROR_TOL = 1e-6

# ==================================================================================================
# NUMERICAL VALIDITY LIMITS
# ==================================================================================================

PF_ERROR_TOLERANCE = 1.0e-5

HARD_VOLTAGE_MIN = 0.20
HARD_VOLTAGE_MAX = 2.00

HARD_LOADING_MAX_PCT = 100000.0

HARD_GENERATOR_Q_ABS_MAX_MVAR = 1.0e6


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

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{digits}f}"

    except Exception:

        return str(value)


def fmt_iterations(value):

    if value is None:
        return "N/A"

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{int(round(value))}"

    except Exception:

        return "N/A"


# ==================================================================================================
# PYPSA RESULT EXTRACTION
# ==================================================================================================

def extract_snapshot_value(obj, snapshot):

    if obj is None:
        return np.nan

    try:

        if isinstance(obj, pd.DataFrame):

            if snapshot in obj.index:

                row = obj.loc[snapshot]

                if isinstance(row, pd.Series):

                    numeric = pd.to_numeric(
                        row,
                        errors="coerce"
                    ).dropna()

                    if len(numeric) == 0:
                        return np.nan

                    if len(numeric) == 1:
                        return numeric.iloc[0]

                    return numeric.max()

            if snapshot in obj.columns:

                col = obj[snapshot]

                numeric = pd.to_numeric(
                    col,
                    errors="coerce"
                ).dropna()

                if len(numeric) == 0:
                    return np.nan

                if len(numeric) == 1:
                    return numeric.iloc[0]

                return numeric.max()

        if isinstance(obj, pd.Series):

            if snapshot in obj.index:

                value = obj.loc[snapshot]

                if np.isscalar(value):
                    return value

                numeric = pd.to_numeric(
                    pd.Series(value),
                    errors="coerce"
                ).dropna()

                if len(numeric) == 0:
                    return np.nan

                return numeric.max()

            numeric = pd.to_numeric(
                obj,
                errors="coerce"
            ).dropna()

            if len(numeric) == 0:
                return np.nan

            if len(numeric) == 1:
                return numeric.iloc[0]

            return numeric.max()

        if np.isscalar(obj):
            return obj

    except Exception:
        pass

    return np.nan


def extract_convergence(result, snapshot):

    converged = False
    pf_error = np.nan
    iterations = np.nan

    if result is None:
        return converged, pf_error, iterations

    try:

        if "converged" in result:

            raw = extract_snapshot_value(
                result["converged"],
                snapshot
            )

            if isinstance(raw, (bool, np.bool_)):

                converged = bool(raw)

            elif np.isfinite(raw):

                converged = bool(raw)

    except Exception:
        pass

    try:

        if "error" in result:

            pf_error = extract_snapshot_value(
                result["error"],
                snapshot
            )

            pf_error = float(pf_error)

    except Exception:

        pf_error = np.nan

    try:

        if "n_iter" in result:

            iterations = extract_snapshot_value(
                result["n_iter"],
                snapshot
            )

            iterations = float(iterations)

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
    ).to_numpy(dtype=float)

    return arr[np.isfinite(arr)]


def safe_sum(values):

    arr = finite_values(values)

    if len(arr) == 0:
        return np.nan

    return float(arr.sum())


# ==================================================================================================
# EXACT S5.3 REACTIVE SUPPORT IMPLEMENTATION
# ==================================================================================================

def apply_q_support(
    n,
    snapshot,
    bus_name,
    q_support_mvar
):
    """
    EXACT reactive-support implementation used by the supplied S5.3 script.

    A temporary PQ generator is added in memory.

        p_nom   = 0 MW
        control = PQ
        p_set   = 0 MW
        q_set   = +300 MVAr

    The source network is never modified.
    """

    if bus_name not in n.buses.index:

        raise RuntimeError(
            f"Support bus not found: {bus_name}"
        )

    support_name = "S5_4_TEMP_Q_SUPPORT"

    if support_name in n.generators.index:

        raise RuntimeError(
            f"Temporary support generator already exists: "
            f"{support_name}"
        )

    n.add(
        "Generator",
        support_name,
        bus=bus_name,
        p_nom=0.0,
        control="PQ",
        carrier="S5_4_TEMP_REACTIVE_SUPPORT"
    )

    if support_name not in n.generators_t.p_set.columns:

        n.generators_t.p_set[
            support_name
        ] = 0.0

    if support_name not in n.generators_t.q_set.columns:

        n.generators_t.q_set[
            support_name
        ] = 0.0

    n.generators_t.p_set.loc[
        snapshot,
        support_name
    ] = 0.0

    n.generators_t.q_set.loc[
        snapshot,
        support_name
    ] = q_support_mvar

    return support_name


# ==================================================================================================
# LINE THERMAL LOADING
# ==================================================================================================

def calculate_line_loading(n, snapshot):

    if len(n.lines.index) == 0:

        return (
            np.nan,
            0,
            pd.Series(dtype=float)
        )

    names = list(
        n.lines.index
    )

    try:

        p0 = n.lines_t.p0.loc[
            snapshot,
            names
        ].astype(float)

        q0 = n.lines_t.q0.loc[
            snapshot,
            names
        ].astype(float)

        p1 = n.lines_t.p1.loc[
            snapshot,
            names
        ].astype(float)

        q1 = n.lines_t.q1.loc[
            snapshot,
            names
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

        loading = (
            s_max
            / ratings
            * 100.0
        )

        finite = loading[
            np.isfinite(loading)
        ]

        if len(finite) == 0:

            return (
                np.nan,
                0,
                pd.Series(dtype=float)
            )

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
            overloaded,
            finite.sort_values(
                ascending=False
            )
        )

    except Exception:

        return (
            np.nan,
            np.nan,
            pd.Series(dtype=float)
        )


# ==================================================================================================
# TRANSFORMER THERMAL LOADING
# ==================================================================================================

def calculate_transformer_loading(
    n,
    snapshot
):

    if len(n.transformers.index) == 0:

        return (
            np.nan,
            0
        )

    names = list(
        n.transformers.index
    )

    try:

        p0 = n.transformers_t.p0.loc[
            snapshot,
            names
        ].astype(float)

        q0 = n.transformers_t.q0.loc[
            snapshot,
            names
        ].astype(float)

        p1 = n.transformers_t.p1.loc[
            snapshot,
            names
        ].astype(float)

        q1 = n.transformers_t.q1.loc[
            snapshot,
            names
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

        loading = (
            s_max
            / ratings
            * 100.0
        )

        finite = loading[
            np.isfinite(loading)
        ]

        if len(finite) == 0:

            return (
                np.nan,
                0
            )

        return (
            float(finite.max()),
            int(
                (
                    finite
                    >
                    TRANSFORMER_LOADING_LIMIT_PCT
                ).sum()
            )
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


# ==================================================================================================
# VOLTAGE
# ==================================================================================================

def calculate_voltage(
    n,
    snapshot
):

    names = list(
        n.buses.index
    )

    try:

        v = (
            n.buses_t.v_mag_pu.loc[
                snapshot,
                names
            ].astype(float)
        )

    except Exception:

        return (
            np.nan,
            np.nan,
            None,
            None,
            np.nan
        )

    finite = v[
        np.isfinite(v)
    ]

    if len(finite) == 0:

        return (
            np.nan,
            np.nan,
            None,
            None,
            0
        )

    min_v = float(
        finite.min()
    )

    max_v = float(
        finite.max()
    )

    min_bus = finite.idxmin()

    max_bus = finite.idxmax()

    undervoltage_count = int(
        (
            finite
            <
            VOLTAGE_SECURITY_MIN
        ).sum()
    )

    return (
        min_v,
        max_v,
        min_bus,
        max_bus,
        undervoltage_count
    )


# ==================================================================================================
# GENERATOR Q
# ==================================================================================================

def calculate_generator_q(
    n,
    snapshot
):

    names = list(
        n.generators.index
    )

    try:

        return safe_sum(
            n.generators_t.q.loc[
                snapshot,
                names
            ]
        )

    except Exception:

        return np.nan


# ==================================================================================================
# PHYSICAL VALIDITY
# ==================================================================================================

def validate_physical_state(
    converged,
    pf_error,
    min_voltage,
    max_voltage,
    max_line_loading,
    max_transformer_loading,
    generator_q
):

    reasons = []

    if not converged:

        reasons.append(
            "NON_CONVERGED"
        )

    if not np.isfinite(pf_error):

        reasons.append(
            "PF_ERROR_NOT_FINITE"
        )

    elif pf_error > PF_ERROR_TOLERANCE:

        reasons.append(
            f"PF_ERROR_ABOVE_TOLERANCE:{pf_error:.6g}"
        )

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

    if not np.isfinite(max_line_loading):

        reasons.append(
            "LINE_LOADING_NOT_FINITE"
        )

    elif max_line_loading > HARD_LOADING_MAX_PCT:

        reasons.append(
            f"LINE_LOADING_NUMERICALLY_IMPLAUSIBLE:{max_line_loading:.6g}"
        )

    if not np.isfinite(max_transformer_loading):

        reasons.append(
            "TRANSFORMER_LOADING_NOT_FINITE"
        )

    elif max_transformer_loading > HARD_LOADING_MAX_PCT:

        reasons.append(
            f"TRANSFORMER_LOADING_NUMERICALLY_IMPLAUSIBLE:{max_transformer_loading:.6g}"
        )

    if not np.isfinite(generator_q):

        reasons.append(
            "GENERATOR_Q_NOT_FINITE"
        )

    elif abs(generator_q) > HARD_GENERATOR_Q_ABS_MAX_MVAR:

        reasons.append(
            f"GENERATOR_Q_NUMERICALLY_IMPLAUSIBLE:{generator_q:.6g}"
        )

    return (
        len(reasons) == 0,
        reasons
    )


# ==================================================================================================
# SECURITY
# ==================================================================================================

def calculate_security(
    valid_ac_solution,
    min_voltage,
    max_voltage,
    overloaded_lines,
    overloaded_transformers
):

    if not valid_ac_solution:

        return (
            False,
            False,
            False
        )

    voltage_security = (
        np.isfinite(min_voltage)
        and
        np.isfinite(max_voltage)
        and
        min_voltage >= VOLTAGE_SECURITY_MIN
        and
        max_voltage <= VOLTAGE_SECURITY_MAX
    )

    thermal_security = (
        np.isfinite(overloaded_lines)
        and
        np.isfinite(overloaded_transformers)
        and
        overloaded_lines == 0
        and
        overloaded_transformers == 0
    )

    overall_security = (
        voltage_security
        and
        thermal_security
    )

    return (
        bool(voltage_security),
        bool(thermal_security),
        bool(overall_security)
    )


# ==================================================================================================
# DISPATCH INTEGRITY
# ==================================================================================================

def calculate_dispatch_integrity(
    n,
    snapshot,
    generator_names,
    load_names,
    generator_p_before,
    load_p_before
):

    generator_p_after = (
        n.generators_t.p_set.loc[
            snapshot,
            generator_names
        ]
        .astype(float)
    )

    load_p_after = (
        n.loads_t.p_set.loc[
            snapshot,
            load_names
        ]
        .astype(float)
    )

    generator_delta = (
        generator_p_after
        -
        generator_p_before
    ).to_numpy()

    load_delta = (
        load_p_after
        -
        load_p_before
    ).to_numpy()

    generator_p_change = float(
        np.max(
            np.abs(generator_delta)
        )
    )

    load_p_change = float(
        np.max(
            np.abs(load_delta)
        )
    )

    unchanged = (
        generator_p_change <= 1e-9
        and
        load_p_change <= 1e-9
    )

    return (
        generator_p_change,
        load_p_change,
        bool(unchanged)
    )


# ==================================================================================================
# S5.2 REFERENCE REPRODUCTION
# ==================================================================================================

def reproduce_s52_reference():

    print_header(
        "S5.2 REFERENCE GATE"
    )

    print(
        "The established S5.2 300 MVAr operating point is reproduced"
    )

    print(
        "using the same reactive-support implementation supplied in S5.3."
    )

    n = pypsa.Network(
        str(SOURCE_NETWORK)
    )

    if SNAPSHOT not in n.snapshots:

        raise RuntimeError(
            f"Snapshot not found: {SNAPSHOT}"
        )

    n.set_snapshots(
        [SNAPSHOT]
    )

    generator_names = list(
        n.generators.index
    )

    load_names = list(
        n.loads.index
    )

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

    apply_q_support(
        n=n,
        snapshot=SNAPSHOT,
        bus_name=WEAK_BUS,
        q_support_mvar=Q_SUPPORT_MVAR
    )

    print()
    print("AC NONLINEAR POWER FLOW")

    with warnings.catch_warnings():

        warnings.simplefilter(
            "always"
        )

        pf_result = n.pf(
            snapshots=[SNAPSHOT],
            distribute_slack=True
        )

    (
        converged,
        pf_error,
        iterations
    ) = extract_convergence(
        pf_result,
        SNAPSHOT
    )

    (
        min_voltage,
        max_voltage,
        weakest_bus,
        maximum_voltage_bus,
        undervoltage_buses
    ) = calculate_voltage(
        n,
        SNAPSHOT
    )

    (
        max_line_loading,
        overloaded_lines,
        _
    ) = calculate_line_loading(
        n,
        SNAPSHOT
    )

    (
        max_transformer_loading,
        overloaded_transformers
    ) = calculate_transformer_loading(
        n,
        SNAPSHOT
    )

    generator_q = calculate_generator_q(
        n,
        SNAPSHOT
    )

    (
        generator_p_change,
        load_p_change,
        dispatch_unchanged
    ) = calculate_dispatch_integrity(
        n,
        SNAPSHOT,
        generator_names,
        load_names,
        generator_p_before,
        load_p_before
    )

    print()
    print("S5.2 REFERENCE OBSERVED")

    print(
        f"Minimum V              : "
        f"{fmt(min_voltage)} pu"
    )

    print(
        f"Weakest bus            : "
        f"{weakest_bus}"
    )

    print(
        f"Undervoltage buses     : "
        f"{undervoltage_buses}"
    )

    print(
        f"Maximum line loading   : "
        f"{fmt(max_line_loading)}%"
    )

    print(
        f"Overloaded lines       : "
        f"{overloaded_lines}"
    )

    print(
        f"Maximum transformer    : "
        f"{fmt(max_transformer_loading)}%"
    )

    print(
        f"Overloaded transformers: "
        f"{overloaded_transformers}"
    )

    print(
        f"PF converged           : "
        f"{converged}"
    )

    print(
        f"PF error               : "
        f"{fmt(pf_error, 12)}"
    )

    print(
        f"Iterations             : "
        f"{fmt_iterations(iterations)}"
    )

    print(
        f"Generator Q total      : "
        f"{fmt(generator_q)} MVAr"
    )

    print()
    print("REFERENCE CHECKS")

    checks = {}

    checks["minimum_voltage"] = (
        np.isfinite(min_voltage)
        and
        abs(
            min_voltage
            -
            REFERENCE_VMIN
        )
        <= VMIN_TOL
    )

    checks["undervoltage_count"] = (
        undervoltage_buses
        ==
        REFERENCE_UV_COUNT
    )

    checks["max_line_loading"] = (
        np.isfinite(max_line_loading)
        and
        abs(
            max_line_loading
            -
            REFERENCE_MAX_LINE_LOADING
        )
        <= LINE_LOADING_TOL
    )

    checks["overloaded_lines"] = (
        overloaded_lines
        ==
        REFERENCE_OVERLOADED_LINES
    )

    checks["max_transformer_loading"] = (
        np.isfinite(max_transformer_loading)
        and
        abs(
            max_transformer_loading
            -
            REFERENCE_MAX_TRANSFORMER_LOADING
        )
        <= TRANSFORMER_LOADING_TOL
    )

    checks["overloaded_transformers"] = (
        overloaded_transformers
        ==
        REFERENCE_OVERLOADED_TRANSFORMERS
    )

    checks["pf_converged"] = bool(
        converged
    )

    checks["pf_error"] = (
        np.isfinite(pf_error)
        and
        pf_error <= PF_ERROR_TOL
    )

    checks["dispatch_unchanged"] = bool(
        dispatch_unchanged
    )

    for name, passed in checks.items():

        print(
            f"  {name:<28} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

    reference_pass = all(
        checks.values()
    )

    print()

    print(
        f"S5.2 300 MVAr REFERENCE : "
        f"{'PASS' if reference_pass else 'FAIL'}"
    )

    if not reference_pass:

        print()
        print(
            "EXPECTED S5.2 FINGERPRINT"
        )

        print(
            f"  Vmin                 = "
            f"{REFERENCE_VMIN:.6f} pu"
        )

        print(
            f"  Max line loading     = "
            f"{REFERENCE_MAX_LINE_LOADING:.6f}%"
        )

        print(
            f"  Overloaded lines     = "
            f"{REFERENCE_OVERLOADED_LINES}"
        )

        print(
            f"  Max transformer      = "
            f"{REFERENCE_MAX_TRANSFORMER_LOADING:.6f}%"
        )

        print(
            f"  Overloaded transform= "
            f"{REFERENCE_OVERLOADED_TRANSFORMERS}"
        )

    return {
        "pass": reference_pass,
        "min_voltage": min_voltage,
        "max_line_loading": max_line_loading,
        "overloaded_lines": overloaded_lines,
        "max_transformer_loading": max_transformer_loading,
        "overloaded_transformers": overloaded_transformers,
        "converged": converged,
        "pf_error": pf_error,
        "iterations": iterations,
        "dispatch_unchanged": dispatch_unchanged,
        "generator_p_change": generator_p_change,
        "load_p_change": load_p_change,
    }


# ==================================================================================================
# SINGLE JOINT CASE
# ==================================================================================================

def run_joint_case(
    multiplier,
    case_number,
    total_cases
):

    case_id = (
        f"JOINT_Q{int(Q_SUPPORT_MVAR):04d}_T"
        f"{int(round(multiplier * 100)):03d}"
    )

    print()
    print(
        f"[{case_number:02d}/{total_cases:02d}] "
        f"JOINT Q={Q_SUPPORT_MVAR:.0f} MVAr × "
        f"T={multiplier:.2f}x"
    )

    print_header(
        f"CASE {case_id}"
    )

    result = {

        "case": case_id,

        "q_support_mvar": Q_SUPPORT_MVAR,

        "thermal_multiplier": multiplier,

        "support_bus": WEAK_BUS,

        "converged": False,

        "pf_error": np.nan,

        "iterations": np.nan,

        "physical_solution": False,

        "valid_ac_solution": False,

        "min_voltage_pu": np.nan,

        "max_voltage_pu": np.nan,

        "weakest_bus": None,

        "maximum_voltage_bus": None,

        "undervoltage_buses": np.nan,

        "max_line_loading_pct": np.nan,

        "overloaded_lines": np.nan,

        "max_transformer_loading_pct": np.nan,

        "overloaded_transformers": np.nan,

        "generator_q_mvar": np.nan,

        "voltage_security": False,

        "thermal_security": False,

        "overall_security": False,

        "generator_p_set_change_mw": np.nan,

        "load_p_set_change_mw": np.nan,

        "dispatch_unchanged": False,

        "numerical_validation_reasons": "",

        "exception": None,
    }

    try:

        # ------------------------------------------------------------------------------------------
        # FRESH SOURCE NETWORK
        # ------------------------------------------------------------------------------------------

        n = pypsa.Network(
            str(SOURCE_NETWORK)
        )

        if SNAPSHOT not in n.snapshots:

            raise RuntimeError(
                f"Snapshot not found: {SNAPSHOT}"
            )

        n.set_snapshots(
            [SNAPSHOT]
        )

        generator_names = list(
            n.generators.index
        )

        load_names = list(
            n.loads.index
        )

        # ------------------------------------------------------------------------------------------
        # P-set fingerprints
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
        # EXACT S5.3 Q SUPPORT
        # ------------------------------------------------------------------------------------------

        apply_q_support(
            n=n,
            snapshot=SNAPSHOT,
            bus_name=WEAK_BUS,
            q_support_mvar=Q_SUPPORT_MVAR
        )

        # ------------------------------------------------------------------------------------------
        # EXACT S5.3 THERMAL MODIFICATION
        # ------------------------------------------------------------------------------------------

        original_s_nom = (
            n.lines.s_nom
            .astype(float)
            .copy()
        )

        n.lines.loc[
            original_s_nom.index,
            "s_nom"
        ] = (
            original_s_nom
            *
            multiplier
        )

        print()
        print("TEMPORARY JOINT CONTROL")

        print(
            f"Q support              : "
            f"{Q_SUPPORT_MVAR:.3f} MVAr"
        )

        print(
            f"Support bus            : "
            f"{WEAK_BUS}"
        )

        print(
            f"Thermal multiplier     : "
            f"{multiplier:.2f}x"
        )

        print(
            "Topology changed       : NO"
        )

        print(
            "Generator dispatch     : unchanged"
        )

        print(
            "Permanent change       : NO"
        )

        # ------------------------------------------------------------------------------------------
        # AC POWER FLOW
        # ------------------------------------------------------------------------------------------

        print()
        print("AC NONLINEAR POWER FLOW")

        with warnings.catch_warnings():

            warnings.simplefilter(
                "always"
            )

            pf_result = n.pf(
                snapshots=[SNAPSHOT],
                distribute_slack=True
            )

        (
            converged,
            pf_error,
            iterations
        ) = extract_convergence(
            pf_result,
            SNAPSHOT
        )

        result["converged"] = bool(
            converged
        )

        result["pf_error"] = pf_error

        result["iterations"] = iterations

        # ------------------------------------------------------------------------------------------
        # VOLTAGE
        # ------------------------------------------------------------------------------------------

        (
            min_voltage,
            max_voltage,
            weakest_bus,
            maximum_voltage_bus,
            undervoltage_buses
        ) = calculate_voltage(
            n,
            SNAPSHOT
        )

        result["min_voltage_pu"] = min_voltage
        result["max_voltage_pu"] = max_voltage
        result["weakest_bus"] = weakest_bus
        result["maximum_voltage_bus"] = maximum_voltage_bus
        result["undervoltage_buses"] = undervoltage_buses

        # ------------------------------------------------------------------------------------------
        # LINE THERMAL
        # ------------------------------------------------------------------------------------------

        (
            max_line_loading,
            overloaded_lines,
            _
        ) = calculate_line_loading(
            n,
            SNAPSHOT
        )

        result["max_line_loading_pct"] = (
            max_line_loading
        )

        result["overloaded_lines"] = (
            overloaded_lines
        )

        # ------------------------------------------------------------------------------------------
        # TRANSFORMER THERMAL
        # ------------------------------------------------------------------------------------------

        (
            max_transformer_loading,
            overloaded_transformers
        ) = calculate_transformer_loading(
            n,
            SNAPSHOT
        )

        result["max_transformer_loading_pct"] = (
            max_transformer_loading
        )

        result["overloaded_transformers"] = (
            overloaded_transformers
        )

        # ------------------------------------------------------------------------------------------
        # GENERATOR Q
        # ------------------------------------------------------------------------------------------

        generator_q = calculate_generator_q(
            n,
            SNAPSHOT
        )

        result["generator_q_mvar"] = (
            generator_q
        )

        # ------------------------------------------------------------------------------------------
        # PHYSICAL VALIDITY
        # ------------------------------------------------------------------------------------------

        (
            physical_solution,
            validation_reasons
        ) = validate_physical_state(
            converged=converged,
            pf_error=pf_error,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            max_line_loading=max_line_loading,
            max_transformer_loading=max_transformer_loading,
            generator_q=generator_q
        )

        result["physical_solution"] = bool(
            physical_solution
        )

        result["valid_ac_solution"] = bool(
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
        # SECURITY
        # ------------------------------------------------------------------------------------------

        (
            voltage_security,
            thermal_security,
            overall_security
        ) = calculate_security(
            valid_ac_solution=result[
                "valid_ac_solution"
            ],
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            overloaded_lines=overloaded_lines,
            overloaded_transformers=overloaded_transformers
        )

        result["voltage_security"] = (
            voltage_security
        )

        result["thermal_security"] = (
            thermal_security
        )

        result["overall_security"] = (
            overall_security
        )

        # ------------------------------------------------------------------------------------------
        # DISPATCH INTEGRITY
        # ------------------------------------------------------------------------------------------

        (
            generator_p_change,
            load_p_change,
            dispatch_unchanged
        ) = calculate_dispatch_integrity(
            n,
            SNAPSHOT,
            generator_names,
            load_names,
            generator_p_before,
            load_p_before
        )

        result[
            "generator_p_set_change_mw"
        ] = generator_p_change

        result[
            "load_p_set_change_mw"
        ] = load_p_change

        result[
            "dispatch_unchanged"
        ] = dispatch_unchanged

        # ------------------------------------------------------------------------------------------
        # CONSOLE OUTPUT
        # ------------------------------------------------------------------------------------------

        print()
        print("RESULT")

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
            f"{fmt_iterations(iterations)}"
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
            f"Weakest bus           : "
            f"{weakest_bus}"
        )

        print(
            f"Undervoltage buses    : "
            f"{undervoltage_buses}"
        )

        print(
            f"Maximum line loading  : "
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
            f"Generator Q total     : "
            f"{fmt(generator_q)} MVAr"
        )

        print()
        print("SECURITY")

        print(
            f"Voltage security      : "
            f"{voltage_security}"
        )

        print(
            f"Thermal security      : "
            f"{thermal_security}"
        )

        print(
            f"Overall security      : "
            f"{overall_security}"
        )

        print()
        print("DISPATCH INTEGRITY")

        print(
            f"Generator P-set change : "
            f"{generator_p_change:.12f} MW"
        )

        print(
            f"Load P-set change      : "
            f"{load_p_change:.12f} MW"
        )

        print(
            f"Dispatch unchanged     : "
            f"{dispatch_unchanged}"
        )

        if validation_reasons:

            print()
            print("VALIDATION REASONS")

            for reason in validation_reasons:

                print(
                    f"  - {reason}"
                )

    except Exception as exc:

        result["exception"] = (
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print("CASE EXCEPTION")

        print(
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    print_header(
        "S5.4 — JOINT VOLTAGE + THERMAL SECURITY EVALUATION"
    )

    print(
        f"Network                       : "
        f"{SOURCE_NETWORK}"
    )

    print(
        f"Snapshot                      : "
        f"{SNAPSHOT}"
    )

    print(
        "PF                            : "
        "AC nonlinear"
    )

    print(
        "Slack                         : "
        "distributed"
    )

    print(
        "Dispatch                      : "
        "unchanged"
    )

    print(
        "Loads P                       : "
        "unchanged"
    )

    print(
        f"Weak bus                      : "
        f"{WEAK_BUS}"
    )

    print(
        "Source                        : "
        "READ-ONLY"
    )

    print()
    print(
        "S5.4 evaluates temporary Q support "
        "× thermal reinforcement."
    )

    print(
        "Every case uses a fresh in-memory network."
    )

    print(
        "No permanent reinforcement is applied."
    )

    print(
        "No dispatch optimisation is performed."
    )

    # ----------------------------------------------------------------------------------------------
    # Source validation
    # ----------------------------------------------------------------------------------------------

    if not SOURCE_NETWORK.exists():

        raise FileNotFoundError(
            f"Source network not found:\n"
            f"{SOURCE_NETWORK}"
        )

    source = pypsa.Network(
        str(SOURCE_NETWORK)
    )

    if SNAPSHOT not in source.snapshots:

        raise RuntimeError(
            f"Snapshot not found: {SNAPSHOT}"
        )

    source.set_snapshots(
        [SNAPSHOT]
    )

    if WEAK_BUS not in source.buses.index:

        raise RuntimeError(
            f"Weak/support bus not found: {WEAK_BUS}"
        )

    print_header(
        "SOURCE NETWORK"
    )

    print(
        f"Buses        : "
        f"{len(source.buses)}"
    )

    print(
        f"Lines        : "
        f"{len(source.lines)}"
    )

    print(
        f"Transformers : "
        f"{len(source.transformers)}"
    )

    print(
        f"Links        : "
        f"{len(source.links)}"
    )

    print(
        f"Generators   : "
        f"{len(source.generators)}"
    )

    print(
        f"Loads        : "
        f"{len(source.loads)}"
    )

    # ----------------------------------------------------------------------------------------------
    # S5.2 reference gate
    # ----------------------------------------------------------------------------------------------

    reference = reproduce_s52_reference()

    if not reference["pass"]:

        print_header(
            "S5.4 GATE"
        )

        print(
            "S5.2 300 MVAr reference : FAILED"
        )

        print(
            "S5.4 STATUS             : LOCKED"
        )

        print()
        print(
            "Joint sweep NOT started."
        )

        print(
            "The S5.2 reference must reproduce "
            "the supplied S5.3 fingerprint first."
        )

        return

    # ----------------------------------------------------------------------------------------------
    # S5.3 1.75x analytical fingerprint
    # ----------------------------------------------------------------------------------------------

    expected_175_line_loading = (
        REFERENCE_MAX_LINE_LOADING
        / 1.75
    )

    print_header(
        "S5.3 REFERENCE GATE"
    )

    print(
        "S5.2 300 MVAr reference : CONFIRMED"
    )

    print(
        "S5.4 STATUS             : UNLOCKED"
    )

    print()
    print(
        "Expected S5.3 1.75x line loading:"
    )

    print(
        f"  166.234091 / 1.75 = "
        f"{expected_175_line_loading:.6f}%"
    )

    print()
    print(
        "At fixed dispatch and fixed Q support, "
        "thermal reinforcement changes ratings only."
    )

    print(
        "Therefore the AC operating point should remain "
        "the same while calculated line loading decreases."
    )

    # ----------------------------------------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.4 JOINT SECURITY SWEEP"
    )

    print(
        "Fixed Q support:"
    )

    print(
        f"  {Q_SUPPORT_MVAR:.1f} MVAr"
    )

    print()
    print(
        "Thermal multipliers:"
    )

    print(
        ", ".join(
            f"{x:.2f}x"
            for x in THERMAL_MULTIPLIERS
        )
    )

    results = []

    total_cases = len(
        THERMAL_MULTIPLIERS
    )

    for i, multiplier in enumerate(
        THERMAL_MULTIPLIERS,
        start=1
    ):

        result = run_joint_case(
            multiplier=multiplier,
            case_number=i,
            total_cases=total_cases
        )

        results.append(
            result
        )

    df = pd.DataFrame(
        results
    )

    # ----------------------------------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.4 — JOINT SECURITY SUMMARY"
    )

    summary_columns = [

        "case",

        "q_support_mvar",

        "thermal_multiplier",

        "converged",

        "valid_ac_solution",

        "pf_error",

        "iterations",

        "min_voltage_pu",

        "weakest_bus",

        "undervoltage_buses",

        "max_line_loading_pct",

        "overloaded_lines",

        "max_transformer_loading_pct",

        "overloaded_transformers",

        "voltage_security",

        "thermal_security",

        "overall_security",
    ]

    print(
        df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Thermal secure cases
    # ----------------------------------------------------------------------------------------------

    thermal_secure_rows = df[
        df["thermal_security"]
        == True
    ]

    overall_secure_rows = df[
        df["overall_security"]
        == True
    ]

    print_header(
        "THERMAL SECURITY RESULTS"
    )

    print(
        f"Thermal-secure tested cases : "
        f"{len(thermal_secure_rows)}"
    )

    if len(thermal_secure_rows) > 0:

        print()

        print(
            thermal_secure_rows[
                [
                    "thermal_multiplier",
                    "max_line_loading_pct",
                    "overloaded_lines",
                    "max_transformer_loading_pct",
                    "overloaded_transformers",
                ]
            ].to_string(
                index=False
            )
        )

        minimum_thermal_multiplier = float(
            thermal_secure_rows[
                "thermal_multiplier"
            ].min()
        )

        print()

        print(
            f"Lowest tested thermal-secure multiplier : "
            f"{minimum_thermal_multiplier:.2f}x"
        )

    else:

        minimum_thermal_multiplier = np.nan

        print()

        print(
            "No tested multiplier eliminates all "
            "thermal overloads."
        )

    # ----------------------------------------------------------------------------------------------
    # Overall security
    # ----------------------------------------------------------------------------------------------

    print_header(
        "OVERALL SECURITY RESULTS"
    )

    print(
        f"Overall-secure tested cases : "
        f"{len(overall_secure_rows)}"
    )

    if len(overall_secure_rows) > 0:

        print()

        print(
            overall_secure_rows[
                [
                    "thermal_multiplier",
                    "min_voltage_pu",
                    "undervoltage_buses",
                    "max_line_loading_pct",
                    "overloaded_lines",
                    "max_transformer_loading_pct",
                    "overloaded_transformers",
                ]
            ].to_string(
                index=False
            )
        )

        minimum_overall_multiplier = float(
            overall_secure_rows[
                "thermal_multiplier"
            ].min()
        )

        print()

        print(
            f"Lowest tested overall-secure multiplier : "
            f"{minimum_overall_multiplier:.2f}x"
        )

    else:

        minimum_overall_multiplier = np.nan

        print()

        print(
            "No tested case achieves complete "
            "voltage + thermal security."
        )

    # ----------------------------------------------------------------------------------------------
    # Transition analysis
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.4 SECURITY TRANSITION ANALYSIS"
    )

    valid_rows = df[
        df["valid_ac_solution"]
        == True
    ]

    if len(valid_rows) > 0:

        print(
            f"Valid AC cases              : "
            f"{len(valid_rows)}"
        )

        print(
            f"Lowest valid multiplier     : "
            f"{valid_rows['thermal_multiplier'].min():.2f}x"
        )

        print(
            f"Highest valid multiplier    : "
            f"{valid_rows['thermal_multiplier'].max():.2f}x"
        )

    if len(thermal_secure_rows) > 0:

        first_thermal = (
            thermal_secure_rows
            .sort_values(
                "thermal_multiplier"
            )
            .iloc[0]
        )

        print()
        print(
            "FIRST THERMAL-SECURE POINT"
        )

        print(
            f"  Thermal multiplier : "
            f"{first_thermal['thermal_multiplier']:.2f}x"
        )

        print(
            f"  Max line loading   : "
            f"{first_thermal['max_line_loading_pct']:.6f}%"
        )

        print(
            f"  Overloaded lines   : "
            f"{int(first_thermal['overloaded_lines'])}"
        )

        print(
            f"  Max transformer    : "
            f"{first_thermal['max_transformer_loading_pct']:.6f}%"
        )

    if len(overall_secure_rows) > 0:

        first_overall = (
            overall_secure_rows
            .sort_values(
                "thermal_multiplier"
            )
            .iloc[0]
        )

        print()
        print(
            "FIRST OVERALL-SECURE POINT"
        )

        print(
            f"  Thermal multiplier : "
            f"{first_overall['thermal_multiplier']:.2f}x"
        )

        print(
            f"  Minimum voltage    : "
            f"{first_overall['min_voltage_pu']:.6f} pu"
        )

        print(
            f"  Max line loading   : "
            f"{first_overall['max_line_loading_pct']:.6f}%"
        )

        print(
            f"  Overloaded lines   : "
            f"{int(first_overall['overloaded_lines'])}"
        )

    # ----------------------------------------------------------------------------------------------
    # Expected 1.75x diagnostic
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.3 1.75x CONSISTENCY DIAGNOSTIC"
    )

    row_175 = df[
        np.isclose(
            df["thermal_multiplier"],
            1.75
        )
    ]

    if len(row_175) == 1:

        row = row_175.iloc[0]

        observed_175 = float(
            row["max_line_loading_pct"]
        )

        expected_175 = (
            REFERENCE_MAX_LINE_LOADING
            / 1.75
        )

        difference = (
            observed_175
            -
            expected_175
        )

        print(
            f"Expected max line loading : "
            f"{expected_175:.6f}%"
        )

        print(
            f"Observed max line loading : "
            f"{observed_175:.6f}%"
        )

        print(
            f"Difference                 : "
            f"{difference:.6f} percentage points"
        )

        if abs(difference) <= LINE_LOADING_TOL:

            print(
                "1.75x consistency          : PASS"
            )

        else:

            print(
                "1.75x consistency          : CHECK REQUIRED"
            )

            print()
            print(
                "This indicates that the S5.4 operating point "
                "is not reproducing the supplied S5.3 condition."
            )

    else:

        print(
            "No 1.75x case found in the sweep."
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
        "TEMPORARY Q SUPPORT     : YES — IN MEMORY ONLY"
    )

    print(
        "TEMPORARY LINE RATING   : YES — IN MEMORY ONLY"
    )

    print(
        "PERMANENT REINFORCEMENT : NO"
    )

    print(
        "GENERATOR DISPATCH      : NO"
    )

    print(
        "LOAD P CHANGED          : NO"
    )

    print(
        "TOPOLOGY CHANGED        : NO"
    )

    print(
        "SOURCE .NC OVERWRITTEN  : NO"
    )

    # ----------------------------------------------------------------------------------------------
    # Save CSV
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
        "S5.4 RESULTS SAVED"
    )

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    # ----------------------------------------------------------------------------------------------
    # Final status
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.4 FINAL STATUS"
    )

    if len(thermal_secure_rows) > 0:

        print(
            "THERMAL SECURITY : ACHIEVED "
            "IN TESTED RANGE"
        )

        print(
            f"Minimum tested thermal multiplier : "
            f"{minimum_thermal_multiplier:.2f}x"
        )

    else:

        print(
            "THERMAL SECURITY : NOT ACHIEVED "
            "IN TESTED RANGE"
        )

    if len(overall_secure_rows) > 0:

        print(
            "OVERALL SECURITY : ACHIEVED "
            "IN TESTED RANGE"
        )

        print(
            f"Minimum tested overall multiplier : "
            f"{minimum_overall_multiplier:.2f}x"
        )

    else:

        print(
            "OVERALL SECURITY : NOT ACHIEVED "
            "IN TESTED RANGE"
        )

    print()
    print(
        "S5.4 COMPLETE"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":

    main()