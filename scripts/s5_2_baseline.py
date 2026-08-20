# ==================================================================================================
# S5.2 — CONTROLLED VOLTAGE SUPPORT EVALUATION
# ==================================================================================================
#
# Purpose
# -------
# Evaluate whether temporary reactive power support at the Stage-5 weakest
# bus can improve the AC voltage state.
#
# AUTHORITATIVE S5.1 BASELINE / S4.5J Q=0 FINGERPRINT
# ----------------------------------------------------
# Minimum voltage          : 0.738929 pu
# Weakest bus              : way/104388595-220
# Undervoltage buses       : 26
# Maximum line loading     : 174.416721 %
# Overloaded lines         : 9
# Maximum transformer      : 34.917640 %
# Overloaded transformers  : 0
# PF iterations            : 6
# PF residual              : 6.286177e-7
#
# EXPERIMENT
# ----------
# Temporary reactive injection is applied at the weakest bus only.
#
# No:
#   - permanent network modification
#   - line reinforcement
#   - transformer reinforcement
#   - generator dispatch change
#   - load P change
#   - topology change
#
# Every test case starts from a fresh copy of the source network.
#
# IMPORTANT
# ---------
# AC validity, voltage security and thermal security are separate gates.
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
    / "s5_2_voltage_support.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

# Authoritative S5.1 weak bus
WEAK_BUS = "way/104388595-220"


# ==================================================================================================
# SUPPORT SWEEP
# ==================================================================================================
#
# Positive Q means reactive power injection into the network at WEAK_BUS.
#
# The support is temporary and exists only in the in-memory test network.
#
# ==================================================================================================

Q_SUPPORT_LEVELS_MVAR = [
    0.0,
    25.0,
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
    200.0,
    250.0,
    300.0,
]


# ==================================================================================================
# SECURITY LIMITS
# ==================================================================================================

VOLTAGE_SECURITY_MIN = 0.90
VOLTAGE_SECURITY_MAX = 1.10

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0


# ==================================================================================================
# HARD NUMERICAL VALIDITY LIMITS
# ==================================================================================================

HARD_VOLTAGE_MIN = 0.20
HARD_VOLTAGE_MAX = 2.00

HARD_ANGLE_ABS_MAX_RAD = 2.0 * math.pi

HARD_LOADING_MAX_PCT = 100000.0

HARD_GENERATOR_Q_ABS_MAX_MVAR = 1.0e6

PF_ERROR_TOLERANCE = 1.0e-5


# ==================================================================================================
# AUTHORITATIVE S5.1 BASELINE FINGERPRINT
# ==================================================================================================

BASELINE_VMIN = 0.738929
BASELINE_WEAK_BUS = "way/104388595-220"
BASELINE_UNDERVOLTAGE_COUNT = 26

BASELINE_MAX_LINE_LOADING = 174.416721
BASELINE_OVERLOADED_LINES = 9

BASELINE_MAX_TRANSFORMER_LOADING = 34.917640
BASELINE_OVERLOADED_TRANSFORMERS = 0

BASELINE_PF_ITERATIONS = 6
BASELINE_PF_ERROR = 6.286177e-7


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
            return str(value)

        return f"{value:.{digits}f}"

    except Exception:

        return str(value)


# ==================================================================================================
# ROBUST PYPSA RESULT EXTRACTION
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

            value = extract_snapshot_value(
                result["converged"],
                snapshot
            )

            if isinstance(value, (bool, np.bool_)):

                converged = bool(value)

            elif np.isfinite(value):

                converged = bool(value)

    except Exception:
        pass

    try:

        if "error" in result:

            value = extract_snapshot_value(
                result["error"],
                snapshot
            )

            try:
                pf_error = float(value)
            except Exception:
                pf_error = np.nan

    except Exception:
        pass

    try:

        if "n_iter" in result:

            value = extract_snapshot_value(
                result["n_iter"],
                snapshot
            )

            try:
                iterations = float(value)
            except Exception:
                iterations = np.nan

    except Exception:
        pass

    return converged, pf_error, iterations


# ==================================================================================================
# SAFE NUMERIC HELPERS
# ==================================================================================================

def finite_values(values):

    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=float)

    return numeric[np.isfinite(numeric)]


def safe_sum(values):

    vals = finite_values(values)

    if len(vals) == 0:
        return np.nan

    return float(vals.sum())


# ==================================================================================================
# BASELINE FINGERPRINT
# ==================================================================================================

def check_baseline_fingerprint():

    print_header(
        "S5.2 — S5.1 BASELINE FINGERPRINT CHECK"
    )

    print(
        "Checking the source network against the established "
        "S4.5J Q=0 / S5.1 fingerprint."
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

    # ----------------------------------------------------------------------------------------------
    # Operating point
    # ----------------------------------------------------------------------------------------------

    generator_names = list(
        n.generators.index
    )

    load_names = list(
        n.loads.index
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

    # ----------------------------------------------------------------------------------------------
    # AC PF
    # ----------------------------------------------------------------------------------------------

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
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

    # ----------------------------------------------------------------------------------------------
    # Voltage
    # ----------------------------------------------------------------------------------------------

    voltage = (
        n.buses_t.v_mag_pu.loc[
            SNAPSHOT
        ]
        .astype(float)
    )

    finite_voltage = voltage[
        np.isfinite(voltage)
    ]

    if len(finite_voltage) == 0:

        raise RuntimeError(
            "Baseline PF produced no finite bus voltages."
        )

    min_voltage = float(
        finite_voltage.min()
    )

    max_voltage = float(
        finite_voltage.max()
    )

    weak_bus = finite_voltage.idxmin()

    undervoltage_count = int(
        (
            finite_voltage
            < VOLTAGE_SECURITY_MIN
        ).sum()
    )

    # ----------------------------------------------------------------------------------------------
    # Line loading
    # ----------------------------------------------------------------------------------------------

    (
        max_line_loading,
        overloaded_lines
    ) = calculate_line_loading(
        n,
        SNAPSHOT
    )

    # ----------------------------------------------------------------------------------------------
    # Transformer loading
    # ----------------------------------------------------------------------------------------------

    (
        max_transformer_loading,
        overloaded_transformers
    ) = calculate_transformer_loading(
        n,
        SNAPSHOT
    )

    print()
    print("S5.1 BASELINE OBSERVED")

    print(
        f"Minimum V              : "
        f"{min_voltage:.6f} pu"
    )

    print(
        f"Weakest bus            : "
        f"{weak_bus}"
    )

    print(
        f"Undervoltage buses     : "
        f"{undervoltage_count}"
    )

    print(
        f"Maximum line loading   : "
        f"{max_line_loading:.6f}%"
    )

    print(
        f"Overloaded lines       : "
        f"{overloaded_lines}"
    )

    print(
        f"Maximum transformer    : "
        f"{max_transformer_loading:.6f}%"
    )

    print(
        f"Overloaded transformers: "
        f"{overloaded_transformers}"
    )

    print(
        f"PF iterations          : "
        f"{iterations:.0f}"
    )

    print(
        f"PF residual            : "
        f"{pf_error:.10g}"
    )

    # ----------------------------------------------------------------------------------------------
    # Fingerprint comparisons
    # ----------------------------------------------------------------------------------------------

    checks = {

        "minimum_voltage":
            abs(
                min_voltage
                -
                BASELINE_VMIN
            ) <= 1e-5,

        "weakest_bus":
            weak_bus
            ==
            BASELINE_WEAK_BUS,

        "undervoltage_count":
            undervoltage_count
            ==
            BASELINE_UNDERVOLTAGE_COUNT,

        "max_line_loading":
            abs(
                max_line_loading
                -
                BASELINE_MAX_LINE_LOADING
            ) <= 1e-5,

        "overloaded_lines":
            overloaded_lines
            ==
            BASELINE_OVERLOADED_LINES,

        "max_transformer_loading":
            abs(
                max_transformer_loading
                -
                BASELINE_MAX_TRANSFORMER_LOADING
            ) <= 1e-5,

        "overloaded_transformers":
            overloaded_transformers
            ==
            BASELINE_OVERLOADED_TRANSFORMERS,

        "pf_iterations":
            int(iterations)
            ==
            BASELINE_PF_ITERATIONS,

        "pf_error":
            abs(
                pf_error
                -
                BASELINE_PF_ERROR
            ) <= 1e-9,

        "pf_converged":
            bool(converged),
    }

    print()
    print("FINGERPRINT CHECKS")

    all_ok = True

    for name, passed in checks.items():

        print(
            f"  {name:<28} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_ok = False

    print()

    print(
        f"S5.1 BASELINE FINGERPRINT : "
        f"{'PASS' if all_ok else 'FAIL'}"
    )

    if not all_ok:

        raise RuntimeError(
            "S5.1 baseline fingerprint does not match the "
            "established S4.5J Q=0 reference. "
            "S5.2 is LOCKED."
        )

    return {
        "generator_p_set_mw": generator_p_set,
        "load_p_set_mw": load_p_set,
        "converged": converged,
        "pf_error": pf_error,
        "iterations": iterations,
        "min_voltage_pu": min_voltage,
        "max_voltage_pu": max_voltage,
        "weak_bus": weak_bus,
        "undervoltage_count": undervoltage_count,
        "max_line_loading_pct": max_line_loading,
        "overloaded_lines": overloaded_lines,
        "max_transformer_loading_pct":
            max_transformer_loading,
        "overloaded_transformers":
            overloaded_transformers,
    }


# ==================================================================================================
# LINE THERMAL CHECK
# ==================================================================================================

def calculate_line_loading(n, snapshot):

    if len(n.lines.index) == 0:
        return np.nan, 0

    line_names = list(
        n.lines.index
    )

    try:

        p0 = (
            n.lines_t.p0.loc[
                snapshot,
                line_names
            ]
            .astype(float)
        )

        q0 = (
            n.lines_t.q0.loc[
                snapshot,
                line_names
            ]
            .astype(float)
        )

        p1 = (
            n.lines_t.p1.loc[
                snapshot,
                line_names
            ]
            .astype(float)
        )

        q1 = (
            n.lines_t.q1.loc[
                snapshot,
                line_names
            ]
            .astype(float)
        )

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
        ).max(axis=1)

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

        finite = loading[
            np.isfinite(loading)
        ]

        if len(finite) == 0:
            return np.nan, np.nan

        return (
            float(finite.max()),
            int(
                (
                    finite
                    >
                    LINE_LOADING_LIMIT_PCT
                ).sum()
            )
        )

    except Exception:

        return np.nan, np.nan


# ==================================================================================================
# TRANSFORMER THERMAL CHECK
# ==================================================================================================

def calculate_transformer_loading(n, snapshot):

    if len(n.transformers.index) == 0:
        return np.nan, 0

    transformer_names = list(
        n.transformers.index
    )

    try:

        p0 = (
            n.transformers_t.p0.loc[
                snapshot,
                transformer_names
            ]
            .astype(float)
        )

        q0 = (
            n.transformers_t.q0.loc[
                snapshot,
                transformer_names
            ]
            .astype(float)
        )

        p1 = (
            n.transformers_t.p1.loc[
                snapshot,
                transformer_names
            ]
            .astype(float)
        )

        q1 = (
            n.transformers_t.q1.loc[
                snapshot,
                transformer_names
            ]
            .astype(float)
        )

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
        ).max(axis=1)

        ratings = (
            n.transformers.s_nom
            .astype(float)
        )

        loading = (
            s_max
            /
            ratings
            *
            100.0
        )

        finite = loading[
            np.isfinite(loading)
        ]

        if len(finite) == 0:
            return np.nan, np.nan

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

        return np.nan, np.nan


# ==================================================================================================
# REACTIVE SUPPORT IMPLEMENTATION
# ==================================================================================================

def apply_reactive_support(
    n,
    snapshot,
    weak_bus,
    q_support_mvar
):
    """
    Add a temporary shunt-like reactive injection at the weak bus.

    Implementation:
        A temporary generator is created with:
            p_set = 0 MW
            q_set = Q support
            control = PQ

    This is NOT written to the source network.

    The temporary generator exists only in the fresh in-memory case.
    """

    if weak_bus not in n.buses.index:

        raise RuntimeError(
            f"Weak bus '{weak_bus}' does not exist "
            f"in the source network."
        )

    temporary_generator = (
        "S5_2_TEMP_Q_SUPPORT"
    )

    if temporary_generator in n.generators.index:

        raise RuntimeError(
            "Temporary S5.2 support generator already exists."
        )

    n.add(
        "Generator",
        temporary_generator,
        bus=weak_bus,
        control="PQ",
        p_nom=max(
            abs(float(q_support_mvar)),
            1.0
        ),
        p_set=0.0,
        q_set=float(q_support_mvar),
    )

    return temporary_generator


# ==================================================================================================
# SINGLE SUPPORT CASE
# ==================================================================================================

def run_case(
    q_support_mvar,
    case_number,
    total_cases,
    baseline
):

    case_id = (
        f"QSUP_{q_support_mvar:06.1f}"
        .replace(".", "_")
        + "MVAR"
    )

    print()
    print(
        f"[{case_number:02d}/{total_cases:02d}] "
        f"Q SUPPORT = "
        f"{q_support_mvar:.1f} Mvar"
    )

    print_header(
        f"CASE {case_id}"
    )

    result = {

        "case":
            case_id,

        "q_support_mvar":
            q_support_mvar,

        "weak_bus":
            WEAK_BUS,

        "converged":
            False,

        "valid_ac_solution":
            False,

        "physical_solution":
            False,

        "pf_error":
            np.nan,

        "iterations":
            np.nan,

        "min_voltage_pu":
            np.nan,

        "max_voltage_pu":
            np.nan,

        "weakest_bus":
            None,

        "undervoltage_buses":
            np.nan,

        "voltage_improvement_pu":
            np.nan,

        "max_line_loading_pct":
            np.nan,

        "overloaded_lines":
            np.nan,

        "max_transformer_loading_pct":
            np.nan,

        "overloaded_transformers":
            np.nan,

        "generator_q_total_mvar":
            np.nan,

        "generator_p_set_mw":
            np.nan,

        "load_p_set_mw":
            np.nan,

        "solved_generation_mw":
            np.nan,

        "solved_load_mw":
            np.nan,

        "generation_minus_load_mw":
            np.nan,

        "generator_p_set_change_mw":
            np.nan,

        "load_p_set_change_mw":
            np.nan,

        "dispatch_unchanged":
            False,

        "temporary_support_present":
            False,

        "voltage_security":
            False,

        "thermal_security":
            False,

        "overall_security":
            False,

        "validation_reasons":
            "",

        "exception":
            None,
    }

    try:

        # ------------------------------------------------------------------------------------------
        # Fresh source network
        # ------------------------------------------------------------------------------------------

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

        generator_names_before = list(
            n.generators.index
        )

        load_names = list(
            n.loads.index
        )

        generator_p_before = (
            n.generators_t.p_set.loc[
                SNAPSHOT,
                generator_names_before
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

        generator_p_set = float(
            generator_p_before.sum()
        )

        load_p_set = float(
            load_p_before.sum()
        )

        result[
            "generator_p_set_mw"
        ] = generator_p_set

        result[
            "load_p_set_mw"
        ] = load_p_set

        # ------------------------------------------------------------------------------------------
        # Add temporary Q support
        # ------------------------------------------------------------------------------------------

        temporary_generator = apply_reactive_support(
            n=n,
            snapshot=SNAPSHOT,
            weak_bus=WEAK_BUS,
            q_support_mvar=q_support_mvar
        )

        result[
            "temporary_support_present"
        ] = (
            temporary_generator
            in
            n.generators.index
        )

        print()
        print(
            "TEMPORARY SUPPORT"
        )

        print(
            f"Bus                 : {WEAK_BUS}"
        )

        print(
            f"Q support           : "
            f"{q_support_mvar:.3f} Mvar"
        )

        print(
            "P support            : 0.000 MW"
        )

        print(
            "Topology changed     : NO"
        )

        print(
            "Permanent change     : NO"
        )

        # ------------------------------------------------------------------------------------------
        # AC PF
        # ------------------------------------------------------------------------------------------

        print()
        print(
            "AC NONLINEAR POWER FLOW"
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

        voltage = (
            n.buses_t.v_mag_pu.loc[
                SNAPSHOT
            ]
            .astype(float)
        )

        finite_voltage = voltage[
            np.isfinite(voltage)
        ]

        if len(finite_voltage) == 0:

            raise RuntimeError(
                "No finite bus voltages returned."
            )

        min_voltage = float(
            finite_voltage.min()
        )

        max_voltage = float(
            finite_voltage.max()
        )

        weakest_bus = (
            finite_voltage.idxmin()
        )

        undervoltage_buses = int(
            (
                finite_voltage
                <
                VOLTAGE_SECURITY_MIN
            ).sum()
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
            "undervoltage_buses"
        ] = undervoltage_buses

        result[
            "voltage_improvement_pu"
        ] = (
            min_voltage
            -
            baseline["min_voltage_pu"]
        )

        # ------------------------------------------------------------------------------------------
        # Line loading
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

        # ------------------------------------------------------------------------------------------
        # Transformer loading
        # ------------------------------------------------------------------------------------------

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

        all_generator_names = list(
            n.generators.index
        )

        try:

            generator_q_total = safe_sum(
                n.generators_t.q.loc[
                    SNAPSHOT,
                    all_generator_names
                ]
            )

        except Exception:

            generator_q_total = np.nan

        result[
            "generator_q_total_mvar"
        ] = generator_q_total

        # ------------------------------------------------------------------------------------------
        # Solved P
        # ------------------------------------------------------------------------------------------

        try:

            solved_generation = safe_sum(
                n.generators_t.p.loc[
                    SNAPSHOT,
                    all_generator_names
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
        # Dispatch integrity
        # ------------------------------------------------------------------------------------------

        generator_p_after = (
            n.generators_t.p_set.loc[
                SNAPSHOT,
                generator_names_before
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
        # Physical validity
        # ------------------------------------------------------------------------------------------

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
                "TRANSFORMER_LOADING_NUMERICALLY_IMPLAUSIBLE"
            )

        if not np.isfinite(generator_q_total):

            reasons.append(
                "GENERATOR_Q_NOT_FINITE"
            )

        elif abs(generator_q_total) > HARD_GENERATOR_Q_ABS_MAX_MVAR:

            reasons.append(
                "GENERATOR_Q_NUMERICALLY_IMPLAUSIBLE"
            )

        if not result[
            "dispatch_unchanged"
        ]:

            reasons.append(
                "DISPATCH_CHANGED"
            )

        result[
            "physical_solution"
        ] = (
            len(reasons) == 0
        )

        result[
            "valid_ac_solution"
        ] = bool(
            converged
            and
            result["physical_solution"]
        )

        result[
            "validation_reasons"
        ] = ";".join(
            reasons
        )

        # ------------------------------------------------------------------------------------------
        # Security
        # ------------------------------------------------------------------------------------------

        voltage_security = bool(
            result["valid_ac_solution"]
            and
            min_voltage >= VOLTAGE_SECURITY_MIN
            and
            max_voltage <= VOLTAGE_SECURITY_MAX
        )

        thermal_security = bool(
            result["valid_ac_solution"]
            and
            overloaded_lines == 0
            and
            overloaded_transformers == 0
        )

        overall_security = bool(
            voltage_security
            and
            thermal_security
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
        # Console output
        # ------------------------------------------------------------------------------------------

        print()
        print(
            "RESULT"
        )

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
            f"Weakest bus           : "
            f"{weakest_bus}"
        )

        print(
            f"Undervoltage buses    : "
            f"{undervoltage_buses}"
        )

        print(
            f"Voltage improvement   : "
            f"{fmt(result['voltage_improvement_pu'])} pu"
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
            f"{fmt(generator_q_total)} Mvar"
        )

        print()
        print(
            "SECURITY"
        )

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

        if reasons:

            print()
            print(
                "VALIDATION REASONS"
            )

            for reason in reasons:

                print(
                    f"  - {reason}"
                )

        if caught_warnings:

            print()
            print(
                f"Python warnings captured : "
                f"{len(caught_warnings)}"
            )

            for warning in caught_warnings[:5]:

                print(
                    f"  WARNING: "
                    f"{warning.message}"
                )

    except Exception as exc:

        result[
            "exception"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        result[
            "validation_reasons"
        ] = (
            f"EXCEPTION:{type(exc).__name__}:{exc}"
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
# MAIN
# ==================================================================================================

def main():

    print_header(
        "S5.2 — CONTROLLED VOLTAGE SUPPORT EVALUATION"
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
        f"Weak bus : {WEAK_BUS}"
    )

    print(
        "Source   : READ-ONLY"
    )

    print()
    print(
        "Temporary Q support is applied only to fresh "
        "in-memory test networks."
    )

    print(
        "No permanent reinforcement is applied."
    )

    print(
        "No dispatch optimization is performed."
    )

    # ----------------------------------------------------------------------------------------------
    # Source validation
    # ----------------------------------------------------------------------------------------------

    if not SOURCE_NETWORK.exists():

        raise FileNotFoundError(
            f"Source network not found:\n{SOURCE_NETWORK}"
        )

    # ----------------------------------------------------------------------------------------------
    # S5.1 gate
    # ----------------------------------------------------------------------------------------------

    baseline = check_baseline_fingerprint()

    print_header(
        "S5.2 GATE"
    )

    print(
        "S5.1 BASELINE : CONFIRMED"
    )

    print(
        "S5.2 STATUS   : UNLOCKED"
    )

    # ----------------------------------------------------------------------------------------------
    # Test definition
    # ----------------------------------------------------------------------------------------------

    print_header(
        "CONTROLLED Q SUPPORT SWEEP"
    )

    print(
        "Temporary reactive support levels:"
    )

    print(
        ", ".join(
            f"{q:.1f} Mvar"
            for q in Q_SUPPORT_LEVELS_MVAR
        )
    )

    print()
    print(
        "Support bus:"
    )

    print(
        f"  {WEAK_BUS}"
    )

    # ----------------------------------------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------------------------------------

    results = []

    total_cases = len(
        Q_SUPPORT_LEVELS_MVAR
    )

    for i, q_support in enumerate(
        Q_SUPPORT_LEVELS_MVAR,
        start=1
    ):

        result = run_case(
            q_support_mvar=float(q_support),
            case_number=i,
            total_cases=total_cases,
            baseline=baseline
        )

        results.append(
            result
        )

    df = pd.DataFrame(
        results
    )

    # ----------------------------------------------------------------------------------------------
    # Stable column order
    # ----------------------------------------------------------------------------------------------

    columns = [
        "case",
        "q_support_mvar",
        "weak_bus",

        "converged",
        "valid_ac_solution",
        "physical_solution",

        "pf_error",
        "iterations",

        "min_voltage_pu",
        "max_voltage_pu",
        "weakest_bus",
        "undervoltage_buses",
        "voltage_improvement_pu",

        "max_line_loading_pct",
        "overloaded_lines",

        "max_transformer_loading_pct",
        "overloaded_transformers",

        "generator_q_total_mvar",

        "generator_p_set_mw",
        "load_p_set_mw",

        "solved_generation_mw",
        "solved_load_mw",
        "generation_minus_load_mw",

        "generator_p_set_change_mw",
        "load_p_set_change_mw",
        "dispatch_unchanged",

        "temporary_support_present",

        "voltage_security",
        "thermal_security",
        "overall_security",

        "validation_reasons",
        "exception",
    ]

    df = df[
        columns
    ]

    # ----------------------------------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------------------------------

    print_header(
        "S5.2 — CONTROLLED VOLTAGE SUPPORT SUMMARY"
    )

    summary_columns = [
        "q_support_mvar",
        "converged",
        "valid_ac_solution",
        "pf_error",
        "iterations",
        "min_voltage_pu",
        "weakest_bus",
        "undervoltage_buses",
        "voltage_improvement_pu",
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
    # Highest valid support
    # ----------------------------------------------------------------------------------------------

    valid_rows = df[
        df["valid_ac_solution"]
        ==
        True
    ]

    print_header(
        "VALID AC SUPPORT RANGE"
    )

    if len(valid_rows) > 0:

        highest_valid_support = float(
            valid_rows[
                "q_support_mvar"
            ].max()
        )

        print(
            f"Highest tested valid Q support : "
            f"{highest_valid_support:.1f} Mvar"
        )

    else:

        print(
            "No tested Q support level produced "
            "a valid AC solution."
        )

    # ----------------------------------------------------------------------------------------------
    # Voltage improvement
    # ----------------------------------------------------------------------------------------------

    print_header(
        "VOLTAGE IMPROVEMENT"
    )

    valid_improvement_rows = valid_rows.copy()

    if len(valid_improvement_rows) > 0:

        best_row = valid_improvement_rows.loc[
            valid_improvement_rows[
                "min_voltage_pu"
            ].idxmax()
        ]

        print(
            f"Best tested valid support : "
            f"{best_row['q_support_mvar']:.1f} Mvar"
        )

        print(
            f"Baseline Vmin             : "
            f"{baseline['min_voltage_pu']:.6f} pu"
        )

        print(
            f"Best Vmin                 : "
            f"{best_row['min_voltage_pu']:.6f} pu"
        )

        print(
            f"Improvement               : "
            f"{best_row['voltage_improvement_pu']:.6f} pu"
        )

        print(
            f"Remaining weak bus       : "
            f"{best_row['weakest_bus']}"
        )

        print(
            f"Remaining undervoltage   : "
            f"{best_row['undervoltage_buses']}"
        )

    else:

        print(
            "No valid AC case available for "
            "voltage-improvement assessment."
        )

    # ----------------------------------------------------------------------------------------------
    # Security cases
    # ----------------------------------------------------------------------------------------------

    secure_rows = df[
        df["overall_security"]
        ==
        True
    ]

    voltage_secure_rows = df[
        df["voltage_security"]
        ==
        True
    ]

    thermal_secure_rows = df[
        df["thermal_security"]
        ==
        True
    ]

    print_header(
        "S5.2 SECURITY RESULTS"
    )

    print(
        f"Voltage-secure tested cases : "
        f"{len(voltage_secure_rows)}"
    )

    print(
        f"Thermal-secure tested cases : "
        f"{len(thermal_secure_rows)}"
    )

    print(
        f"Overall-secure tested cases : "
        f"{len(secure_rows)}"
    )

    if len(voltage_secure_rows) > 0:

        print()
        print(
            "Voltage-secure Q support:"
        )

        print(
            voltage_secure_rows[
                [
                    "q_support_mvar",
                    "min_voltage_pu",
                    "max_voltage_pu",
                    "undervoltage_buses",
                ]
            ].to_string(
                index=False
            )
        )

    if len(secure_rows) > 0:

        print()
        print(
            "Fully secure Q support:"
        )

        print(
            secure_rows[
                [
                    "q_support_mvar",
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
    # Dispatch integrity
    # ----------------------------------------------------------------------------------------------

    print_header(
        "DISPATCH INTEGRITY"
    )

    max_generator_p_change = safe_sum(
        []
    )

    if "generator_p_set_change_mw" in df:

        finite_changes = pd.to_numeric(
            df[
                "generator_p_set_change_mw"
            ],
            errors="coerce"
        ).dropna()

        if len(finite_changes) > 0:

            max_generator_p_change = float(
                finite_changes.max()
            )

    finite_load_changes = pd.to_numeric(
        df[
            "load_p_set_change_mw"
        ],
        errors="coerce"
    ).dropna()

    if len(finite_load_changes) > 0:

        max_load_p_change = float(
            finite_load_changes.max()
        )

    else:

        max_load_p_change = np.nan

    print(
        f"Maximum generator P-set change : "
        f"{fmt(max_generator_p_change, 12)} MW"
    )

    print(
        f"Maximum load P-set change      : "
        f"{fmt(max_load_p_change, 12)} MW"
    )

    print(
        f"All dispatch checks unchanged   : "
        f"{bool(df['dispatch_unchanged'].all())}"
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
        "S5.2 RESULTS SAVED"
    )

    print(
        f"CSV : {OUTPUT_CSV}"
    )

    print()
    print(
        "Source network modified : NO"
    )

    print(
        "Permanent reinforcement : NO"
    )

    print(
        "Permanent Q device      : NO"
    )

    print(
        "Generator dispatch      : NO"
    )

    print(
        "Load P changed          : NO"
    )

    print(
        "Topology changed        : NO"
    )

    print_header(
        "S5.2 COMPLETE"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    main()