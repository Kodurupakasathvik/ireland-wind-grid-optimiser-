# ==================================================================================================
# S5.3 — CONTROLLED THERMAL REINFORCEMENT EVALUATION
# ==================================================================================================
#
# Purpose
# -------
# S5.2 established that local reactive support can restore voltage security:
#
#   Q support = 300 MVAr
#   Vmin      = ~0.917237 pu
#   UV buses  = 0
#
# However, thermal security remained FAILED:
#
#   Maximum line loading ~= 166.234091 %
#   Overloaded lines     = 9
#
# S5.3 therefore isolates the remaining thermal-security problem.
#
# CONTROLLED EXPERIMENT
# ---------------------
# Temporary line-rating reinforcement is applied to fresh in-memory copies
# of the source network.
#
# The operating point is preserved:
#
#   Generator P : unchanged
#   Load P      : unchanged
#   Q support   : fixed at 300 MVAr
#   Topology    : unchanged
#
# Only line thermal ratings are temporarily multiplied.
#
# IMPORTANT
# ---------
# Source network is READ-ONLY.
# No permanent reinforcement is written to the source .nc file.
# No generator dispatch optimization is performed.
# No load shedding is performed.
# No topology changes are made.
#
# The experiment answers:
#
#   "What temporary transmission-capacity multiplier is required
#    to eliminate the remaining thermal overloads while retaining
#    the voltage-secure 300 MVAr operating condition?"
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
    / "s5_3_thermal_reinforcement.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

WEAK_BUS_S5_2 = "way/104388595-220"

# --------------------------------------------------------------------------------------------------
# Fixed S5.2 voltage-support condition
# --------------------------------------------------------------------------------------------------

Q_SUPPORT_MVAR = 300.0

VOLTAGE_SECURITY_MIN = 0.90
VOLTAGE_SECURITY_MAX = 1.10

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0

# --------------------------------------------------------------------------------------------------
# Thermal reinforcement sweep
# --------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------
# S5.2 reference fingerprint
# --------------------------------------------------------------------------------------------------
#
# This is the observed S5.2 300 MVAr result supplied by the completed run.
#
# We use tolerances rather than exact floating-point equality.
#

REFERENCE_VMIN = 0.917237
REFERENCE_UV_COUNT = 0
REFERENCE_MAX_LINE_LOADING = 166.234091
REFERENCE_OVERLOADED_LINES = 9

REFERENCE_MAX_TRANSFORMER_LOADING = 33.071461
REFERENCE_OVERLOADED_TRANSFORMERS = 0

REFERENCE_PF_ITERATIONS = 5
REFERENCE_PF_ERROR = 1.073222e-08

VMIN_TOL = 5e-5
LINE_LOADING_TOL = 0.05
TRANSFORMER_LOADING_TOL = 0.05
PF_ERROR_TOL = 1e-6

# --------------------------------------------------------------------------------------------------
# Robust AC validity
# --------------------------------------------------------------------------------------------------

PF_ERROR_TOLERANCE = 1.0e-5

HARD_VOLTAGE_MIN = 0.20
HARD_VOLTAGE_MAX = 2.00

HARD_ANGLE_ABS_MAX_RAD = 2.0 * math.pi

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

        if not np.isfinite(float(value)):
            return str(value)

        return f"{float(value):.{digits}f}"

    except Exception:

        return str(value)


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

    return converged, pf_error, iterations


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
# REACTIVE SUPPORT
# ==================================================================================================

def apply_q_support(n, snapshot, bus_name, q_support_mvar):
    """
    Add temporary reactive support using a fresh in-memory generator.

    The generator has:
        p_set = 0 MW
        q_set = +Q MVAr

    It is NOT written to the source network.
    """

    if bus_name not in n.buses.index:

        raise RuntimeError(
            f"Support bus not found: {bus_name}"
        )

    support_name = "S5_3_TEMP_Q_SUPPORT"

    if support_name in n.generators.index:

        raise RuntimeError(
            f"Temporary support generator already exists: {support_name}"
        )

    # Create a temporary generator.
    n.add(
        "Generator",
        support_name,
        bus=bus_name,
        p_nom=0.0,
        control="PQ",
        carrier="S5_3_TEMP_REACTIVE_SUPPORT"
    )

    # Ensure time-series columns exist.
    if support_name not in n.generators_t.p_set.columns:
        n.generators_t.p_set[support_name] = 0.0

    if support_name not in n.generators_t.q_set.columns:
        n.generators_t.q_set[support_name] = 0.0

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
# THERMAL CALCULATIONS
# ==================================================================================================

def calculate_line_loading(n, snapshot):

    if len(n.lines.index) == 0:
        return np.nan, 0, pd.Series(dtype=float)

    names = list(n.lines.index)

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


def calculate_transformer_loading(n, snapshot):

    if len(n.transformers.index) == 0:
        return np.nan, 0

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

            return np.nan, 0

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

def calculate_voltage(n, snapshot):

    names = list(
        n.buses.index
    )

    v = (
        n.buses_t.v_mag_pu.loc[
            snapshot,
            names
        ].astype(float)
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

def calculate_generator_q(n, snapshot):

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
        return False, False, False

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
# SINGLE THERMAL CASE
# ==================================================================================================

def run_case(
    multiplier,
    case_number,
    total_cases
):

    case_id = (
        f"THERMAL_{multiplier:.2f}X"
        .replace(".", "_")
    )

    print()
    print(
        f"[{case_number:02d}/{total_cases:02d}] "
        f"THERMAL MULTIPLIER = {multiplier:.2f}x"
    )

    print_header(
        f"CASE {case_id}"
    )

    result = {

        "case": case_id,

        "thermal_multiplier": multiplier,

        "q_support_mvar": Q_SUPPORT_MVAR,

        "converged": False,

        "valid_ac_solution": False,

        "physical_solution": False,

        "pf_error": np.nan,

        "iterations": np.nan,

        "min_voltage_pu": np.nan,

        "max_voltage_pu": np.nan,

        "weakest_bus": None,

        "maximum_voltage_bus": None,

        "undervoltage_buses": np.nan,

        "voltage_improvement_pu": np.nan,

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
        # Fresh source copy
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
        # P references
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
        # Temporary 300 MVAr support
        # ------------------------------------------------------------------------------------------

        support_name = apply_q_support(
            n=n,
            snapshot=SNAPSHOT,
            bus_name=WEAK_BUS_S5_2,
            q_support_mvar=Q_SUPPORT_MVAR
        )

        # ------------------------------------------------------------------------------------------
        # Temporary thermal reinforcement
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
        print("TEMPORARY THERMAL REINFORCEMENT")

        print(
            f"Line rating multiplier : {multiplier:.2f}x"
        )

        print(
            f"Q support              : {Q_SUPPORT_MVAR:.3f} MVAr"
        )

        print(
            f"Support bus            : {WEAK_BUS_S5_2}"
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
        # AC PF
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

        result[
            "min_voltage_pu"
        ] = min_voltage

        result[
            "max_voltage_pu"
        ] = max_voltage

        result[
            "weakest_bus"
        ] = weakest_bus

        result[
            "maximum_voltage_bus"
        ] = maximum_voltage_bus

        result[
            "undervoltage_buses"
        ] = undervoltage_buses

        result[
            "voltage_improvement_pu"
        ] = (
            min_voltage
            -
            REFERENCE_VMIN
        )

        # ------------------------------------------------------------------------------------------
        # Thermal
        # ------------------------------------------------------------------------------------------

        (
            max_line_loading,
            overloaded_lines,
            sorted_loadings
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

        generator_q = calculate_generator_q(
            n,
            SNAPSHOT
        )

        result[
            "generator_q_mvar"
        ] = generator_q

        # ------------------------------------------------------------------------------------------
        # Physical validity
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
        # Security
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

        result[
            "voltage_security"
        ] = voltage_security

        result[
            "thermal_security"
        ] = thermal_security

        result[
            "overall_security"
        ] = overall_security

        # ------------------------------------------------------------------------------------------
        # Dispatch integrity
        # ------------------------------------------------------------------------------------------

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

        generator_p_change = float(
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

        load_p_change = float(
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

        result[
            "generator_p_set_change_mw"
        ] = generator_p_change

        result[
            "load_p_set_change_mw"
        ] = load_p_change

        result[
            "dispatch_unchanged"
        ] = bool(
            generator_p_change <= 1e-9
            and
            load_p_change <= 1e-9
        )

        # ------------------------------------------------------------------------------------------
        # Console result
        # ------------------------------------------------------------------------------------------

        print()
        print("RESULT")

        print(
            f"PyPSA converged       : {converged}"
        )

        print(
            f"PF error              : {fmt(pf_error, 10)}"
        )

        print(
            f"Iterations            : {fmt(iterations, 0)}"
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
            f"{fmt(generator_q)} Mvar"
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
            f"{result['dispatch_unchanged']}"
        )

        if validation_reasons:

            print()
            print(
                "VALIDATION REASONS"
            )

            for reason in validation_reasons:

                print(
                    f"  - {reason}"
                )

    except Exception as exc:

        result[
            "exception"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "CASE EXCEPTION"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ==================================================================================================
# S5.2 REFERENCE CHECK
# ==================================================================================================

def run_s5_2_reference():

    print_header(
        "S5.3 — S5.2 300 MVAr REFERENCE CHECK"
    )

    print(
        "Before the thermal sweep, the 300 MVAr voltage-support"
    )

    print(
        "condition is independently reproduced."
    )

    n = pypsa.Network(
        str(SOURCE_NETWORK)
    )

    if SNAPSHOT not in n.snapshots:

        raise RuntimeError(
            f"Snapshot '{SNAPSHOT}' not found."
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
        bus_name=WEAK_BUS_S5_2,
        q_support_mvar=Q_SUPPORT_MVAR
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

    generator_p_change = float(
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

    load_p_change = float(
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

    print()
    print(
        "S5.2 REFERENCE OBSERVED"
    )

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
        f"PF iterations          : "
        f"{fmt(iterations, 0)}"
    )

    print(
        f"PF residual            : "
        f"{fmt(pf_error, 12)}"
    )

    print()
    print(
        "REFERENCE CHECKS"
    )

    checks = {}

    checks[
        "minimum_voltage"
    ] = (
        np.isfinite(min_voltage)
        and
        abs(
            min_voltage
            -
            REFERENCE_VMIN
        )
        <= VMIN_TOL
    )

    checks[
        "undervoltage_count"
    ] = (
        undervoltage_buses
        ==
        REFERENCE_UV_COUNT
    )

    checks[
        "max_line_loading"
    ] = (
        np.isfinite(max_line_loading)
        and
        abs(
            max_line_loading
            -
            REFERENCE_MAX_LINE_LOADING
        )
        <= LINE_LOADING_TOL
    )

    checks[
        "overloaded_lines"
    ] = (
        overloaded_lines
        ==
        REFERENCE_OVERLOADED_LINES
    )

    checks[
        "max_transformer_loading"
    ] = (
        np.isfinite(max_transformer_loading)
        and
        abs(
            max_transformer_loading
            -
            REFERENCE_MAX_TRANSFORMER_LOADING
        )
        <= TRANSFORMER_LOADING_TOL
    )

    checks[
        "overloaded_transformers"
    ] = (
        overloaded_transformers
        ==
        REFERENCE_OVERLOADED_TRANSFORMERS
    )

    checks[
        "pf_converged"
    ] = bool(
        converged
    )

    checks[
        "pf_error"
    ] = (
        np.isfinite(pf_error)
        and
        pf_error
        <= PF_ERROR_TOL
    )

    checks[
        "dispatch_unchanged"
    ] = (
        generator_p_change <= 1e-9
        and
        load_p_change <= 1e-9
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

    return reference_pass


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    print_header(
        "S5.3 — CONTROLLED THERMAL REINFORCEMENT EVALUATION"
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
        "Loads P  : unchanged"
    )

    print(
        f"Q support: {Q_SUPPORT_MVAR:.1f} MVAr temporary"
    )

    print(
        "Source   : READ-ONLY"
    )

    print()
    print(
        "S5.3 tests temporary thermal reinforcement only."
    )

    print(
        "No permanent network modification is performed."
    )

    # ----------------------------------------------------------------------------------------------
    # Source check
    # ----------------------------------------------------------------------------------------------

    if not SOURCE_NETWORK.exists():

        raise FileNotFoundError(
            f"Source network not found:\n{SOURCE_NETWORK}"
        )

    source = pypsa.Network(
        str(SOURCE_NETWORK)
    )

    if SNAPSHOT not in source.snapshots:

        raise RuntimeError(
            f"Snapshot '{SNAPSHOT}' not found."
        )

    source.set_snapshots(
        [SNAPSHOT]
    )

    if WEAK_BUS_S5_2 not in source.buses.index:

        raise RuntimeError(
            f"S5.2 weak bus not found: {WEAK_BUS_S5_2}"
        )

    print_header(
        "SOURCE NETWORK"
    )

    print(
        f"Buses        : {len(source.buses)}"
    )

    print(
        f"Lines        : {len(source.lines)}"
    )

    print(
        f"Transformers : {len(source.transformers)}"
    )

    print(
        f"Links        : {len(source.links)}"
    )

    print(
        f"Generators   : {len(source.generators)}"
    )

    print(
        f"Loads        : {len(source.loads)}"
    )

    # ----------------------------------------------------------------------------------------------
    # S5.2 reference gate
    # ----------------------------------------------------------------------------------------------

    reference_pass = run_s5_2_reference()

    if not reference_pass:

        print_header(
            "S5.3 GATE"
        )

        print(
            "S5.2 300 MVAr reference : FAILED"
        )

        print(
            "S5.3 STATUS             : LOCKED"
        )

        print()
        print(
            "Thermal reinforcement sweep NOT started."
        )

        print(
            "Investigate S5.2 reproduction before continuing."
        )

        return

    print_header(
        "S5.3 GATE"
    )

    print(
        "S5.2 300 MVAr reference : CONFIRMED"
    )

    print(
        "S5.3 STATUS             : UNLOCKED"
    )

    # ----------------------------------------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------------------------------------

    print_header(
        "CONTROLLED THERMAL REINFORCEMENT SWEEP"
    )

    print(
        "Temporary line-capacity multipliers:"
    )

    print(
        ", ".join(
            f"{x:.2f}x"
            for x in THERMAL_MULTIPLIERS
        )
    )

    print()
    print(
        f"Fixed temporary Q support : "
        f"{Q_SUPPORT_MVAR:.1f} MVAr"
    )

    print(
        f"Support bus                : "
        f"{WEAK_BUS_S5_2}"
    )

    results = []

    total_cases = len(
        THERMAL_MULTIPLIERS
    )

    for i, multiplier in enumerate(
        THERMAL_MULTIPLIERS,
        start=1
    ):

        result = run_case(
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
        "S5.3 — THERMAL REINFORCEMENT SUMMARY"
    )

    summary_columns = [

        "thermal_multiplier",

        "q_support_mvar",

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
    # Thermal-secure cases
    # ----------------------------------------------------------------------------------------------

    thermal_secure_rows = df[
        df[
            "thermal_security"
        ]
        == True
    ]

    overall_secure_rows = df[
        df[
            "overall_security"
        ]
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
                    "min_voltage_pu",
                    "undervoltage_buses",
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

        print(
            "No tested thermal multiplier eliminates "
            "all line overloads."
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

        print(
            "No tested case achieves complete "
            "voltage + thermal security."
        )

    # ----------------------------------------------------------------------------------------------
    # First thermal-secure transition
    # ----------------------------------------------------------------------------------------------

    print_header(
        "THERMAL TRANSITION ANALYSIS"
    )

    valid_rows = df[
        df[
            "valid_ac_solution"
        ]
        == True
    ]

    if len(valid_rows) > 0:

        print(
            f"Highest tested valid multiplier : "
            f"{valid_rows['thermal_multiplier'].max():.2f}x"
        )

        print(
            f"Lowest tested valid multiplier  : "
            f"{valid_rows['thermal_multiplier'].min():.2f}x"
        )

    if len(thermal_secure_rows) > 0:

        first_secure = thermal_secure_rows.iloc[0]

        print()

        print(
            "First tested thermal-secure point:"
        )

        print(
            f"  Multiplier       : "
            f"{first_secure['thermal_multiplier']:.2f}x"
        )

        print(
            f"  Maximum loading  : "
            f"{first_secure['max_line_loading_pct']:.6f}%"
        )

        print(
            f"  Overloaded lines : "
            f"{int(first_secure['overloaded_lines'])}"
        )

        print(
            f"  Minimum voltage  : "
            f"{first_secure['min_voltage_pu']:.6f} pu"
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
        "PERMANENT CHANGES       : NONE"
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
        "S5.3 RESULTS SAVED"
    )

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print_header(
        "S5.3 FINAL STATUS"
    )

    if len(thermal_secure_rows) > 0:

        print(
            "THERMAL SECURITY : ACHIEVED "
            "AT TESTED MULTIPLIER"
        )

    else:

        print(
            "THERMAL SECURITY : NOT ACHIEVED "
            "IN TESTED RANGE"
        )

    if len(overall_secure_rows) > 0:

        print(
            "OVERALL SECURITY : ACHIEVED "
            "AT TESTED MULTIPLIER"
        )

    else:

        print(
            "OVERALL SECURITY : NOT ACHIEVED "
            "IN TESTED RANGE"
        )

    print()
    print(
        "S5.3 COMPLETE"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":

    main()