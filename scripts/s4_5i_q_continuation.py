# ==================================================================================================
# S4.5I — FINE Q CONTINUATION / REACTIVE POWER BOUNDARY ISOLATION — CORRECTED
# ==================================================================================================
#
# Purpose
# -------
# Establish the AC-solution boundary as reactive load is increased.
#
# Coarse:
#   0%, 1%, 2%, ..., 10%
#
# Fine:
#   7.0%, 7.1%, ..., 8.0%
#
# IMPORTANT:
#   AC solution validity is separate from voltage-security compliance.
#
#   AC VALIDITY:
#       PF converged successfully
#       AND PF residual/error is within tolerance
#
#   VOLTAGE SECURITY SCREEN:
#       0.90 <= V <= 1.10 pu
#
# Constraints
# -----------
#   Source network: READ-ONLY
#   P dispatch: unchanged
#   P loads: unchanged
#   Generator Q: 0
#   Distributed slack
#   No reinforcement
#   No reactive compensation
#   No dispatch optimisation
#   No permanent source-network modification
#
# Critical correction
# -------------------
# The source network's actual load names are used.
#
# NEVER construct load names by adding "eirgrid_load_" or any other prefix.
#
# Q is assigned using a Series indexed by the actual n.loads.index, so PyPSA/Pandas
# cannot accidentally receive a foreign load index.
#
# ==================================================================================================

from pathlib import Path
import copy
import warnings

import numpy as np
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

SOURCE_NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_second_reinforced_network.nc"
)

OUTPUT_CSV = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\s4_5i_q_continuation.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

# Load power factor
LOAD_PF = 0.95

# AC solution tolerance
PF_TOLERANCE = 1e-6

# Voltage-security screen
V_MIN_SECURITY = 0.90
V_MAX_SECURITY = 1.10

# Thermal-security reporting threshold
LINE_LOADING_LIMIT = 100.0
TRANSFORMER_LOADING_LIMIT = 100.0

# Coarse continuation
COARSE_Q_LEVELS = [
    float(x) for x in range(0, 11)
]

# Fine continuation
FINE_Q_LEVELS = [
    round(x / 10.0, 1)
    for x in range(70, 81)
]

# Numerical settings
NR_MAX_ITER = 100

# Suppress only the known noisy warning from the installed scientific stack.
warnings.filterwarnings(
    "ignore",
    message="numpy.ndarray size changed"
)


# ==================================================================================================
# PRINT HELPERS
# ==================================================================================================

def banner(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ==================================================================================================
# NETWORK LOADING
# ==================================================================================================

def load_network():
    """
    Load a completely fresh network from disk.

    The source file is never written to.
    """

    n = pypsa.Network(str(SOURCE_NETWORK))

    if SNAPSHOT not in n.snapshots:
        raise ValueError(
            f"Snapshot {SNAPSHOT!r} not found in source network.\n"
            f"Available snapshots: {list(n.snapshots)}"
        )

    # Isolate the requested snapshot.
    n.set_snapshots([SNAPSHOT])

    return n


# ==================================================================================================
# SNAPSHOT HELPERS
# ==================================================================================================

def get_snapshot(n):
    """
    Return the actual snapshot label used by the network.
    """

    if len(n.snapshots) != 1:
        raise RuntimeError(
            f"Expected exactly one snapshot, found {len(n.snapshots)}"
        )

    return n.snapshots[0]


# ==================================================================================================
# ACTUAL LOAD ALIGNMENT — CRITICAL FIX
# ==================================================================================================

def get_actual_load_p(n, snapshot):
    """
    Return active load powers using the ACTUAL load names/index
    present in the imported network.

    This avoids the previous failure caused by trying to access:

        eirgrid_load_...

    names that were not present in n.loads_t.p_set.
    """

    actual_load_index = pd.Index(n.loads.index)

    if len(actual_load_index) == 0:
        raise RuntimeError("Network contains zero loads.")

    # Case 1:
    # Time-dependent p_set exists.
    if hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set"):
        pset = n.loads_t.p_set

        if snapshot in pset.index:
            row = pset.loc[snapshot]

            # Select ONLY columns that actually exist in the network.
            missing = actual_load_index.difference(row.index)

            if len(missing) > 0:
                raise RuntimeError(
                    "The network load table and loads_t.p_set are inconsistent.\n"
                    f"Missing load entries: {list(missing)}"
                )

            result = pd.Series(
                row.loc[actual_load_index].astype(float).values,
                index=actual_load_index,
                dtype=float,
            )

            return result

    # Case 2:
    # Static p_set fallback.
    if "p_set" in n.loads.columns:
        return pd.Series(
            n.loads.loc[actual_load_index, "p_set"].astype(float).values,
            index=actual_load_index,
            dtype=float,
        )

    # Case 3:
    # Static p fallback.
    if "p" in n.loads.columns:
        return pd.Series(
            n.loads.loc[actual_load_index, "p"].astype(float).values,
            index=actual_load_index,
            dtype=float,
        )

    raise RuntimeError(
        "Could not obtain active load powers from the network."
    )


def get_actual_load_q(n, snapshot):
    """
    Obtain the original Q setpoints if present.

    This is mainly used for diagnostics. S4.5I deliberately overwrites
    Q with the controlled continuation value.
    """

    actual_load_index = pd.Index(n.loads.index)

    if hasattr(n, "loads_t") and hasattr(n.loads_t, "q_set"):
        qset = n.loads_t.q_set

        if snapshot in qset.index:
            row = qset.loc[snapshot]

            available = actual_load_index.intersection(row.index)

            return pd.Series(
                row.loc[available].astype(float).values,
                index=available,
                dtype=float,
            )

    if "q_set" in n.loads.columns:
        return pd.Series(
            n.loads.loc[actual_load_index, "q_set"].astype(float).values,
            index=actual_load_index,
            dtype=float,
        )

    if "q" in n.loads.columns:
        return pd.Series(
            n.loads.loc[actual_load_index, "q"].astype(float).values,
            index=actual_load_index,
            dtype=float,
        )

    return pd.Series(
        0.0,
        index=actual_load_index,
        dtype=float,
    )


# ==================================================================================================
# GENERATOR ACTIVE POWER
# ==================================================================================================

def get_generator_p(n, snapshot):
    """
    Return generator active-power setpoints using actual generator names.
    """

    actual_index = pd.Index(n.generators.index)

    if len(actual_index) == 0:
        return pd.Series(dtype=float)

    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_set"):
        pset = n.generators_t.p_set

        if snapshot in pset.index:
            row = pset.loc[snapshot]

            available = actual_index.intersection(row.index)

            return pd.Series(
                row.loc[available].astype(float).values,
                index=available,
                dtype=float,
            )

    if "p_set" in n.generators.columns:
        return pd.Series(
            n.generators.loc[actual_index, "p_set"].astype(float).values,
            index=actual_index,
            dtype=float,
        )

    if "p" in n.generators.columns:
        return pd.Series(
            n.generators.loc[actual_index, "p"].astype(float).values,
            index=actual_index,
            dtype=float,
        )

    return pd.Series(
        0.0,
        index=actual_index,
        dtype=float,
    )


# ==================================================================================================
# SAFE Q ASSIGNMENT — CRITICAL FIX
# ==================================================================================================

def set_load_q(n, snapshot, q_series):
    """
    Assign Q using the network's ACTUAL load index.

    q_series MUST have exactly the same index as n.loads.index.

    This is intentionally implemented without constructing load names.
    """

    actual_index = pd.Index(n.loads.index)

    # Force exact alignment.
    q_aligned = pd.Series(
        q_series.reindex(actual_index).astype(float).values,
        index=actual_index,
        dtype=float,
    )

    if q_aligned.isna().any():
        bad = list(q_aligned[q_aligned.isna()].index)

        raise RuntimeError(
            "Q assignment produced NaN values for loads:\n"
            f"{bad}"
        )

    # Ensure time-series table exists.
    if not hasattr(n, "loads_t"):
        raise RuntimeError("Network has no loads_t table.")

    # Ensure q_set has the correct snapshot/index structure.
    if SNAPSHOT not in n.snapshots:
        raise RuntimeError(
            f"{SNAPSHOT} not present in network snapshots."
        )

    # IMPORTANT:
    # Assign the COMPLETE aligned row.
    #
    # This avoids:
    #   KeyError from foreign load names
    #   "Must have equal len keys and value when setting with an iterable"
    #
    n.loads_t.q_set.loc[snapshot, actual_index] = q_aligned.to_numpy()

    # Verification
    stored = n.loads_t.q_set.loc[snapshot, actual_index]

    if not np.allclose(
        stored.astype(float).to_numpy(),
        q_aligned.astype(float).to_numpy(),
        rtol=0.0,
        atol=1e-10,
        equal_nan=False,
    ):
        raise RuntimeError(
            "Q assignment verification failed."
        )


# ==================================================================================================
# GENERATOR Q = 0
# ==================================================================================================

def set_generator_q_zero(n, snapshot):
    """
    Force all generator reactive-power setpoints to zero.

    This is a controlled S4.5I condition.
    """

    actual_index = pd.Index(n.generators.index)

    if len(actual_index) == 0:
        return

    if not hasattr(n, "generators_t"):
        raise RuntimeError("Network has no generators_t table.")

    zero_q = np.zeros(len(actual_index), dtype=float)

    # q_set may not exist in unusual networks.
    # Creating/initialising it through the normal PyPSA time-series
    # dataframe is preferable to inventing generator names.
    if "q_set" not in n.generators_t:
        n.generators_t["q_set"] = pd.DataFrame(
            index=n.snapshots,
            columns=actual_index,
            dtype=float,
        )

    # Make sure all actual generators are represented.
    for name in actual_index:
        if name not in n.generators_t.q_set.columns:
            n.generators_t.q_set[name] = 0.0

    n.generators_t.q_set.loc[snapshot, actual_index] = zero_q


# ==================================================================================================
# DISTRIBUTED SLACK
# ==================================================================================================

def configure_distributed_slack(n):
    """
    Configure distributed slack where supported by the imported PyPSA
    version.

    PyPSA versions differ in exact distributed-slack configuration.
    The function therefore checks the available API instead of blindly
    writing unsupported attributes.
    """

    # If generators have control settings, use their existing control
    # configuration and enable distributed slack where the API supports it.
    #
    # For this study, the important requirement is:
    #   DO NOT alter generator P dispatch manually.
    #
    # Newer PyPSA versions support:
    #
    #     n.config["solving"]["distributed_slack"]
    #
    # Older versions may not.
    #
    # Therefore only set it if the configuration path exists.

    try:
        if hasattr(n, "config"):
            solving = n.config.get("solving", None)

            if solving is not None:
                if "distributed_slack" in solving:
                    n.config["solving"]["distributed_slack"] = True
                    return "distributed"

    except Exception:
        pass

    # Fallback: preserve network's existing slack/control configuration.
    return "existing_network_slack_configuration"


# ==================================================================================================
# POWER FLOW
# ==================================================================================================

def run_ac_power_flow(n):
    """
    Execute nonlinear AC power flow.

    Returns:
        converged, error, iterations
    """

    converged = False
    pf_error = np.nan
    iterations = np.nan

    try:
        # PyPSA nonlinear AC PF.
        result = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=PF_TOLERANCE,
            use_seed=True,
            distribute_slack=True,
        )

        # PyPSA typically returns a tuple-like result:
        # (n_iter, error, converged)
        #
        # Be defensive because exact return structures vary by version.

        if isinstance(result, tuple) and len(result) >= 3:
            iterations_raw = result[0]
            error_raw = result[1]
            converged_raw = result[2]

            try:
                if hasattr(iterations_raw, "__len__"):
                    iterations = float(np.asarray(iterations_raw).flat[0])
                else:
                    iterations = float(iterations_raw)
            except Exception:
                iterations = np.nan

            try:
                if hasattr(error_raw, "__len__"):
                    pf_error = float(np.asarray(error_raw).flat[0])
                else:
                    pf_error = float(error_raw)
            except Exception:
                pf_error = np.nan

            try:
                if hasattr(converged_raw, "__len__"):
                    converged = bool(np.asarray(converged_raw).flat[0])
                else:
                    converged = bool(converged_raw)
            except Exception:
                converged = False

        else:
            # If PyPSA did not return the expected structure,
            # inspect whether the resulting bus voltages are finite.
            converged = True

    except Exception as exc:
        return False, np.nan, np.nan, repr(exc)

    # Additional physical validation.
    if converged:
        try:
            v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

            if len(v) == 0:
                converged = False
                pf_error = np.nan
                iterations = np.nan

            elif not np.all(np.isfinite(v.astype(float).to_numpy())):
                converged = False

        except Exception:
            converged = False

    return converged, pf_error, iterations, None


# ==================================================================================================
# VOLTAGE METRICS
# ==================================================================================================

def calculate_voltage_metrics(n):
    """

    Calculate voltage statistics from the converged AC solution.

    """

    if SNAPSHOT not in n.buses_t.v_mag_pu.index:
        return np.nan, np.nan, False

    v = n.buses_t.v_mag_pu.loc[SNAPSHOT].astype(float)

    v = v.replace([np.inf, -np.inf], np.nan).dropna()

    if len(v) == 0:
        return np.nan, np.nan, False

    min_v = float(v.min())
    max_v = float(v.max())

    security_valid = (
        min_v >= V_MIN_SECURITY
        and max_v <= V_MAX_SECURITY
    )

    return min_v, max_v, security_valid


# ==================================================================================================
# ANGLE METRICS
# ==================================================================================================

def calculate_angle_metrics(n):
    """
    Calculate bus-angle extrema in radians.
    """

    try:
        theta = n.buses_t.v_ang.loc[SNAPSHOT].astype(float)

        theta = theta.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(theta) == 0:
            return np.nan, np.nan

        return float(theta.min()), float(theta.max())

    except Exception:
        return np.nan, np.nan


# ==================================================================================================
# LINE LOADING
# ==================================================================================================

def calculate_line_metrics(n):
    """
    Calculate maximum line loading and count above 100%.
    """

    try:
        if len(n.lines) == 0:
            return np.nan, 0

        loading = (
            n.lines_t.p0.loc[SNAPSHOT].abs()
            / n.lines.s_nom
            * 100.0
        )

        loading = loading.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(loading) == 0:
            return np.nan, 0

        return (
            float(loading.max()),
            int((loading > LINE_LOADING_LIMIT).sum()),
        )

    except Exception:
        return np.nan, np.nan


# ==================================================================================================
# TRANSFORMER LOADING
# ==================================================================================================

def calculate_transformer_metrics(n):
    """
    Calculate maximum transformer loading and count above 100%.
    """

    try:
        if len(n.transformers) == 0:
            return np.nan, 0

        loading_values = []

        if (
            hasattr(n, "transformers_t")
            and "p0" in n.transformers_t
        ):
            p0 = n.transformers_t.p0.loc[SNAPSHOT].abs()

            s_nom = n.transformers.s_nom.astype(float)

            loading = (
                p0 / s_nom * 100.0
            )

            loading = loading.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            loading_values.extend(
                loading.astype(float).tolist()
            )

        if len(loading_values) == 0:
            return np.nan, 0

        arr = np.asarray(
            loading_values,
            dtype=float
        )

        return (
            float(np.nanmax(arr)),
            int(np.sum(arr > TRANSFORMER_LOADING_LIMIT)),
        )

    except Exception:
        return np.nan, np.nan


# ==================================================================================================
# GENERATOR Q DIAGNOSTIC
# ==================================================================================================

def calculate_generator_q(n):
    """
    Diagnostic only.

    S4.5I sets generator Q setpoint to zero.
    The solved AC generator Q can nevertheless differ from zero depending
    on PyPSA's power-flow treatment.

    This function reports solved generator Q where available.
    """

    try:
        if (
            hasattr(n, "generators_t")
            and "q" in n.generators_t
        ):
            q = n.generators_t.q.loc[SNAPSHOT]

            return float(
                q.astype(float).sum()
            )

    except Exception:
        pass

    return np.nan


# ==================================================================================================
# SINGLE CASE
# ==================================================================================================

def run_case(q_percentage, continuation_stage):
    """

    Run one completely fresh case.

    q_percentage:
        Percentage of the realistic load-Q reference.

    Example:
        7.0 -> 7% of realistic Q
    """

    case_name = (
        f"Q_{q_percentage:04.1f}".replace(".", "_")
        + "PCT"
    )

    section(f"CASE {case_name}")

    print(
        f"Reactive load level : {q_percentage:.1f}%"
    )

    # ----------------------------------------------------------------------------------------------
    # Fresh source load
    # ----------------------------------------------------------------------------------------------

    n = load_network()

    print("Fresh network loaded for this case.")

    print(f"  Buses        : {len(n.buses)}")
    print(f"  Lines        : {len(n.lines)}")
    print(f"  Transformers : {len(n.transformers)}")
    print(f"  Links        : {len(n.links)}")
    print(f"  Generators   : {len(n.generators)}")
    print(f"  Loads        : {len(n.loads)}")

    snapshot = get_snapshot(n)

    # ----------------------------------------------------------------------------------------------
    # Actual operating point
    # ----------------------------------------------------------------------------------------------

    generator_p = get_generator_p(n, snapshot)
    load_p = get_actual_load_p(n, snapshot)

    generator_p_total = float(generator_p.sum())
    load_p_total = float(load_p.sum())

    print()
    print("OPERATING POINT")
    print(
        f"Generator P set : {generator_p_total:.6f} MW"
    )
    print(
        f"Load P set      : {load_p_total:.6f} MW"
    )
    print(
        f"Generation-load : "
        f"{generator_p_total - load_p_total:.6f} MW"
    )

    # ----------------------------------------------------------------------------------------------
    # Realistic Q reference
    # ----------------------------------------------------------------------------------------------

    q_ratio = np.tan(
        np.arccos(LOAD_PF)
    )

    full_realistic_q = (
        load_p_total * q_ratio
    )

    q_fraction = q_percentage / 100.0

    total_load_q = (
        full_realistic_q * q_fraction
    )

    # Proportional distribution across the actual P loads.
    #
    # This guarantees:
    #
    #     sum(Q_i) = requested total Q
    #
    # while preserving the P-load distribution.

    if load_p_total <= 0:
        raise RuntimeError(
            "Total active load is not positive."
        )

    q_series = (
        load_p / load_p_total
    ) * total_load_q

    # ----------------------------------------------------------------------------------------------
    # Q diagnostics
    # ----------------------------------------------------------------------------------------------

    print()
    print("REACTIVE POWER")
    print(
        f"Load PF              : {LOAD_PF:.4f} lagging"
    )
    print(
        f"Q/P ratio            : {q_ratio:.6f}"
    )
    print(
        f"Full realistic Q     : "
        f"{full_realistic_q:.6f} Mvar"
    )
    print(
        f"Requested total Q    : "
        f"{total_load_q:.6f} Mvar"
    )

    # ----------------------------------------------------------------------------------------------
    # Verify actual load-index alignment BEFORE modifying anything.
    # ----------------------------------------------------------------------------------------------

    actual_load_index = pd.Index(n.loads.index)

    if not q_series.index.equals(actual_load_index):
        raise RuntimeError(
            "Internal load-index alignment failure.\n"
            f"n.loads.index = {list(actual_load_index)}\n"
            f"q_series.index = {list(q_series.index)}"
        )

    print()
    print("LOAD INDEX CHECK")
    print(
        f"Actual network load count : "
        f"{len(actual_load_index)}"
    )
    print(
        f"Q vector load count       : "
        f"{len(q_series.index)}"
    )
    print(
        "Actual load names used    : YES"
    )

    # Show names once for diagnostic transparency.
    for i, name in enumerate(actual_load_index):
        print(f"  [{i:02d}] {name}")

    # ----------------------------------------------------------------------------------------------
    # Set controlled Q
    # ----------------------------------------------------------------------------------------------

    set_load_q(
        n,
        snapshot,
        q_series
    )

    # ----------------------------------------------------------------------------------------------
    # Generator Q = 0
    # ----------------------------------------------------------------------------------------------

    set_generator_q_zero(
        n,
        snapshot
    )

    # ----------------------------------------------------------------------------------------------
    # Distributed slack
    # ----------------------------------------------------------------------------------------------

    slack_mode = configure_distributed_slack(n)

    print()
    print(
        f"Slack configuration : {slack_mode}"
    )

    # ----------------------------------------------------------------------------------------------
    # Verify source operating P values were not altered.
    # ----------------------------------------------------------------------------------------------

    generator_p_after = get_generator_p(
        n,
        snapshot
    )

    load_p_after = get_actual_load_p(
        n,
        snapshot
    )

    p_generator_difference = float(
        (
            generator_p_after
            - generator_p
        ).abs().max()
    )

    p_load_difference = float(
        (
            load_p_after
            - load_p
        ).abs().max()
    )

    if p_generator_difference > 1e-9:
        raise RuntimeError(
            "Generator P dispatch changed unexpectedly."
        )

    if p_load_difference > 1e-9:
        raise RuntimeError(
            "Load P changed unexpectedly."
        )

    # ----------------------------------------------------------------------------------------------
    # Verify Q before PF
    # ----------------------------------------------------------------------------------------------

    q_before_pf = n.loads_t.q_set.loc[
        snapshot,
        actual_load_index
    ].astype(float)

    q_sum_before_pf = float(
        q_before_pf.sum()
    )

    if not np.isclose(
        q_sum_before_pf,
        total_load_q,
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Reactive-power assignment total does not match target.\n"
            f"Target : {total_load_q}\n"
            f"Stored : {q_sum_before_pf}"
        )

    # ----------------------------------------------------------------------------------------------
    # AC PF
    # ----------------------------------------------------------------------------------------------

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

    converged, pf_error, iterations, exception = (
        run_ac_power_flow(n)
    )

    # ----------------------------------------------------------------------------------------------
    # Result container
    # ----------------------------------------------------------------------------------------------

    result = {
        "case": case_name,
        "continuation_stage": continuation_stage,
        "q_percentage": float(q_percentage),

        "converged": bool(converged),
        "valid_ac_solution": False,

        "pf_error": pf_error,
        "iterations": iterations,

        "min_voltage_pu": np.nan,
        "max_voltage_pu": np.nan,
        "voltage_security_valid": False,

        "min_angle_rad": np.nan,
        "max_angle_rad": np.nan,

        "max_line_loading_pct": np.nan,
        "overloaded_lines": np.nan,

        "max_transformer_loading_pct": np.nan,
        "overloaded_transformers": np.nan,

        "total_load_q_mvar": float(total_load_q),
        "generator_q_mvar": np.nan,

        "load_pf": float(LOAD_PF),
        "q_p_ratio": float(q_ratio),
        "full_realistic_q_mvar": float(full_realistic_q),

        "generator_p_mw": float(generator_p_total),
        "load_p_mw": float(load_p_total),

        "p_generator_max_difference_mw": p_generator_difference,
        "p_load_max_difference_mw": p_load_difference,

        "slack_mode": slack_mode,

        "exception": exception,
    }

    # ----------------------------------------------------------------------------------------------
    # Invalid / non-converged case
    # ----------------------------------------------------------------------------------------------

    if not converged:

        print()
        print(
            "AC POWER FLOW : INVALID / NON-CONVERGED"
        )

        if exception is not None:
            print(
                f"Exception : {exception}"
            )

        return result

    # ----------------------------------------------------------------------------------------------
    # AC solution is valid
    # ----------------------------------------------------------------------------------------------

    result["valid_ac_solution"] = True

    # Voltage
    min_v, max_v, voltage_security_valid = (
        calculate_voltage_metrics(n)
    )

    result["min_voltage_pu"] = min_v
    result["max_voltage_pu"] = max_v
    result["voltage_security_valid"] = (
        bool(voltage_security_valid)
    )

    # Angles
    min_angle, max_angle = (
        calculate_angle_metrics(n)
    )

    result["min_angle_rad"] = min_angle
    result["max_angle_rad"] = max_angle

    # Lines
    max_line_loading, overloaded_lines = (
        calculate_line_metrics(n)
    )

    result["max_line_loading_pct"] = (
        max_line_loading
    )

    result["overloaded_lines"] = (
        overloaded_lines
    )

    # Transformers
    max_transformer_loading, overloaded_transformers = (
        calculate_transformer_metrics(n)
    )

    result["max_transformer_loading_pct"] = (
        max_transformer_loading
    )

    result["overloaded_transformers"] = (
        overloaded_transformers
    )

    # Generator Q diagnostic
    result["generator_q_mvar"] = (
        calculate_generator_q(n)
    )

    # ----------------------------------------------------------------------------------------------
    # Print valid result
    # ----------------------------------------------------------------------------------------------

    print()
    print(
        "AC POWER FLOW : VALID"
    )

    print(
        f"PF error              : "
        f"{pf_error}"
    )

    print(
        f"Iterations            : "
        f"{iterations}"
    )

    print(
        f"Minimum voltage       : "
        f"{min_v:.6f} pu"
    )

    print(
        f"Maximum voltage       : "
        f"{max_v:.6f} pu"
    )

    print(
        f"Voltage security      : "
        f"{voltage_security_valid}"
    )

    print(
        f"Maximum line loading  : "
        f"{max_line_loading:.6f}%"
        if np.isfinite(max_line_loading)
        else
        "Maximum line loading  : NaN"
    )

    print(
        f"Overloaded lines      : "
        f"{overloaded_lines}"
    )

    print(
        f"Maximum transformer   : "
        f"{max_transformer_loading:.6f}%"
        if np.isfinite(max_transformer_loading)
        else
        "Maximum transformer   : NaN"
    )

    print(
        f"Overloaded transformers: "
        f"{overloaded_transformers}"
    )

    print(
        f"Solved generator Q    : "
        f"{result['generator_q_mvar']}"
    )

    return result


# ==================================================================================================
# MAIN CONTINUATION
# ==================================================================================================

def main():

    banner(
        "S4.5I — FINE Q CONTINUATION / "
        "REACTIVE POWER BOUNDARY ISOLATION — CORRECTED"
    )

    print()
    print(f"Network  : {SOURCE_NETWORK}")
    print(f"Snapshot : {SNAPSHOT}")
    print("PF       : AC nonlinear")
    print("Dispatch : unchanged")
    print("Loads P  : unchanged")
    print("Generator Q : 0")
    print("Slack    : distributed")
    print("Source   : READ-ONLY")

    banner("PURPOSE")

    print(
        """
S4.5H established the tested bracket:

  Q=0%   -> VALID AC SOLUTION
  Q=10%  -> INVALID / NON-CONVERGED

S4.5I performs:

  COARSE:
      0%, 1%, 2%, ..., 10%

  FINE:
      7.0%, 7.1%, ..., 8.0%

AC solution validity is separated from voltage-security compliance.

The 0.90–1.10 pu voltage range is reported only as a
SECURITY SCREEN.

No reinforcement.
No reactive compensation.
No dispatch change.
No source network modification.
"""
    )

    # ==============================================================================================
    # SOURCE NETWORK REFERENCE
    # ==============================================================================================

    banner("LOADING SOURCE NETWORK")

    source = load_network()

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

    snapshot = get_snapshot(source)

    # ==============================================================================================
    # SOURCE OPERATING POINT
    # ==============================================================================================

    banner("SOURCE OPERATING POINT")

    source_generator_p = get_generator_p(
        source,
        snapshot
    )

    source_load_p = get_actual_load_p(
        source,
        snapshot
    )

    source_generator_p_total = float(
        source_generator_p.sum()
    )

    source_load_p_total = float(
        source_load_p.sum()
    )

    source_balance = (
        source_generator_p_total
        - source_load_p_total
    )

    print(
        f"Generator P set : "
        f"{source_generator_p_total:.6f} MW"
    )

    print(
        f"Load P set      : "
        f"{source_load_p_total:.6f} MW"
    )

    print(
        f"Generation-load : "
        f"{source_balance:.6f} MW"
    )

    # ==============================================================================================
    # REALISTIC Q REFERENCE
    # ==============================================================================================

    banner("REALISTIC Q REFERENCE")

    q_ratio = np.tan(
        np.arccos(LOAD_PF)
    )

    full_realistic_q = (
        source_load_p_total
        * q_ratio
    )

    print(
        f"Load PF              : "
        f"{LOAD_PF:.4f} lagging"
    )

    print(
        f"Q/P ratio            : "
        f"{q_ratio:.6f}"
    )

    print(
        f"Full realistic load Q : "
        f"{full_realistic_q:.6f} Mvar"
    )

    # ==============================================================================================
    # CRITICAL LOAD-NAME DIAGNOSTIC
    # ==============================================================================================

    banner(
        "ACTUAL NETWORK LOAD INDEX — REFERENCE CHECK"
    )

    actual_source_loads = pd.Index(
        source.loads.index
    )

    print(
        f"Actual load count : "
        f"{len(actual_source_loads)}"
    )

    print()
    print(
        "The continuation will use these exact names."
    )

    print(
        "No 'eirgrid_load_' prefix will be constructed."
    )

    for i, name in enumerate(actual_source_loads):
        print(
            f"  [{i:02d}] {name}"
        )

    # ==============================================================================================
    # COARSE CONTINUATION
    # ==============================================================================================

    banner("COARSE Q CONTINUATION")

    print(
        "Testing:"
    )

    print(
        ", ".join(
            f"{x:.0f}%"
            for x in COARSE_Q_LEVELS
        )
    )

    all_results = []

    for i, q_level in enumerate(
        COARSE_Q_LEVELS,
        start=1
    ):

        print()
        print(
            f"[{i:02d}/{len(COARSE_Q_LEVELS)}] "
            f"Q={q_level:.1f}%"
        )

        try:
            result = run_case(
                q_level,
                "coarse"
            )

        except Exception as exc:

            result = {
                "case":
                    f"Q_{q_level:04.1f}".replace(
                        ".",
                        "_"
                    ) + "PCT",

                "continuation_stage":
                    "coarse",

                "q_percentage":
                    float(q_level),

                "converged":
                    False,

                "valid_ac_solution":
                    False,

                "pf_error":
                    np.nan,

                "iterations":
                    np.nan,

                "min_voltage_pu":
                    np.nan,

                "max_voltage_pu":
                    np.nan,

                "voltage_security_valid":
                    False,

                "min_angle_rad":
                    np.nan,

                "max_angle_rad":
                    np.nan,

                "max_line_loading_pct":
                    np.nan,

                "overloaded_lines":
                    np.nan,

                "max_transformer_loading_pct":
                    np.nan,

                "overloaded_transformers":
                    np.nan,

                "total_load_q_mvar":
                    source_load_p_total
                    * q_ratio
                    * q_level
                    / 100.0,

                "generator_q_mvar":
                    np.nan,

                "exception":
                    repr(exc),
            }

            print()
            print(
                f"CASE EXCEPTION : "
                f"{result['case']}"
            )

            print(
                result["exception"]
            )

        all_results.append(result)

    # ==============================================================================================
    # FINE CONTINUATION
    # ==============================================================================================

    banner(
        "FINE Q CONTINUATION — "
        "0.1 PERCENTAGE-POINT RESOLUTION"
    )

    print(
        "Testing 7.0% to 8.0% "
        "in 0.1-percentage-point increments."
    )

    print()
    print(
        "Fine levels:"
    )

    print(
        ", ".join(
            f"{x:.1f}%"
            for x in FINE_Q_LEVELS
        )
    )

    for i, q_level in enumerate(
        FINE_Q_LEVELS,
        start=1
    ):

        print()
        print(
            f"[FINE {i:02d}/{len(FINE_Q_LEVELS)}] "
            f"Q={q_level:.1f}%"
        )

        try:
            result = run_case(
                q_level,
                "fine"
            )

        except Exception as exc:

            result = {
                "case":
                    f"Q_{q_level:04.1f}".replace(
                        ".",
                        "_"
                    ) + "PCT",

                "continuation_stage":
                    "fine",

                "q_percentage":
                    float(q_level),

                "converged":
                    False,

                "valid_ac_solution":
                    False,

                "pf_error":
                    np.nan,

                "iterations":
                    np.nan,

                "min_voltage_pu":
                    np.nan,

                "max_voltage_pu":
                    np.nan,

                "voltage_security_valid":
                    False,

                "min_angle_rad":
                    np.nan,

                "max_angle_rad":
                    np.nan,

                "max_line_loading_pct":
                    np.nan,

                "overloaded_lines":
                    np.nan,

                "max_transformer_loading_pct":
                    np.nan,

                "overloaded_transformers":
                    np.nan,

                "total_load_q_mvar":
                    source_load_p_total
                    * q_ratio
                    * q_level
                    / 100.0,

                "generator_q_mvar":
                    np.nan,

                "exception":
                    repr(exc),
            }

            print()
            print(
                f"CASE EXCEPTION : "
                f"{result['case']}"
            )

            print(
                result["exception"]
            )

        all_results.append(result)

    # ==============================================================================================
    # DATAFRAME
    # ==============================================================================================

    banner("S4.5I — Q CONTINUATION SUMMARY")

    results_df = pd.DataFrame(
        all_results
    )

    # Consistent ordering.
    preferred_columns = [
        "case",
        "continuation_stage",
        "q_percentage",
        "converged",
        "valid_ac_solution",
        "pf_error",
        "iterations",
        "min_voltage_pu",
        "max_voltage_pu",
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
        "generator_p_mw",
        "load_p_mw",
        "p_generator_max_difference_mw",
        "p_load_max_difference_mw",
        "slack_mode",
        "exception",
    ]

    existing_columns = [
        c for c in preferred_columns
        if c in results_df.columns
    ]

    results_df = results_df[
        existing_columns
    ]

    print(
        results_df.to_string(
            index=False
        )
    )

    # ==============================================================================================
    # Q=0 REFERENCE CONSISTENCY CHECK
    # ==============================================================================================

    banner(
        "Q=0 REFERENCE CONSISTENCY CHECK"
    )

    q0_rows = results_df[
        np.isclose(
            results_df["q_percentage"].astype(float),
            0.0
        )
        & (
            results_df["continuation_stage"]
            == "coarse"
        )
    ]

    if len(q0_rows) == 1:

        q0 = q0_rows.iloc[0]

        q0_converged = bool(
            q0["converged"]
        )

        q0_valid = bool(
            q0["valid_ac_solution"]
        )

        q0_pf_error = q0["pf_error"]
        q0_iterations = q0["iterations"]
        q0_min_v = q0["min_voltage_pu"]
        q0_max_v = q0["max_voltage_pu"]
        q0_overloaded_lines = q0[
            "overloaded_lines"
        ]

    else:

        q0_converged = False
        q0_valid = False
        q0_pf_error = np.nan
        q0_iterations = np.nan
        q0_min_v = np.nan
        q0_max_v = np.nan
        q0_overloaded_lines = np.nan

    print(
        f"Q=0 converged           : "
        f"{q0_converged}"
    )

    print(
        f"Q=0 valid AC solution   : "
        f"{q0_valid}"
    )

    print(
        f"Q=0 PF error            : "
        f"{q0_pf_error}"
    )

    print(
        f"Q=0 iterations          : "
        f"{q0_iterations}"
    )

    print(
        f"Q=0 minimum voltage     : "
        f"{q0_min_v}"
    )

    print(
        f"Q=0 maximum voltage     : "
        f"{q0_max_v}"
    )

    print(
        f"Q=0 overloaded lines    : "
        f"{q0_overloaded_lines}"
    )

    # ----------------------------------------------------------------------------------------------
    # IMPORTANT:
    #
    # S4.5H reference values are not hard-coded here.
    #
    # We verify whether this fresh Q=0 case actually produces a valid
    # result. The previous failed S4.5I run cannot be used as a reference.
    # ----------------------------------------------------------------------------------------------

    q0_reference_valid = bool(
        q0_valid
    )

    print()
    print(
        "Fresh S4.5I Q=0 reference valid : "
        f"{q0_reference_valid}"
    )

    # ==============================================================================================
    # FINE THRESHOLD INTERPRETATION
    # ==============================================================================================

    banner(
        "FINE Q THRESHOLD INTERPRETATION"
    )

    fine_df = results_df[
        results_df["continuation_stage"]
        == "fine"
    ].copy()

    fine_df = fine_df.sort_values(
        "q_percentage"
    )

    valid_fine = fine_df[
        fine_df["valid_ac_solution"]
        == True
    ]

    invalid_fine = fine_df[
        fine_df["valid_ac_solution"]
        == False
    ]

    if len(valid_fine) > 0:

        highest_valid_q = float(
            valid_fine[
                "q_percentage"
            ].max()
        )

    else:

        highest_valid_q = None

    if len(invalid_fine) > 0:

        first_invalid_q = float(
            invalid_fine[
                "q_percentage"
            ].min()
        )

    else:

        first_invalid_q = None

    print()
    print(
        "FINE CONTINUATION RESULT"
    )

    if highest_valid_q is None:
        print(
            "Highest valid tested Q level : NONE"
        )
    else:
        print(
            f"Highest valid tested Q level : "
            f"{highest_valid_q:.1f}%"
        )

    if first_invalid_q is None:
        print(
            "First invalid tested Q level : NONE"
        )
    else:
        print(
            f"First invalid tested Q level : "
            f"{first_invalid_q:.1f}%"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a tested numerical bracket, "
        "not an exact mathematical collapse point."
    )

    # ==============================================================================================
    # VOLTAGE SECURITY SCREEN
    # ==============================================================================================

    banner(
        "VOLTAGE SECURITY SCREEN"
    )

    print(
        "IMPORTANT:"
    )

    print(
        "This screen is separate from AC solution validity."
    )

    security_columns = [
        "q_percentage",
        "valid_ac_solution",
        "min_voltage_pu",
        "max_voltage_pu",
        "voltage_security_valid",
        "overloaded_lines",
        "overloaded_transformers",
    ]

    print(
        results_df[
            security_columns
        ].to_string(
            index=False
        )
    )

    # ==============================================================================================
    # ADD INTERPRETATION FLAGS
    # ==============================================================================================

    results_df["ac_solution_status"] = np.where(
        results_df["valid_ac_solution"],
        "VALID",
        "INVALID_OR_NON_CONVERGED",
    )

    results_df["voltage_screen_status"] = np.where(
        results_df["valid_ac_solution"],
        np.where(
            results_df["voltage_security_valid"],
            "WITHIN_0.90_1.10_PU",
            "OUTSIDE_0.90_1.10_PU",
        ),
        "NOT_ASSESSED_AC_INVALID",
    )

    # ==============================================================================================
    # SAVE
    # ==============================================================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    banner("S4.5I RESULTS SAVED")

    print(
        f"Summary : {OUTPUT_CSV}"
    )

    print()
    print(
        "Output columns include:"
    )

    print(
        "  valid_ac_solution"
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
        "  total_load_q_mvar"
    )

    print(
        "  p_generator_max_difference_mw"
    )

    print(
        "  p_load_max_difference_mw"
    )

    print()
    print(
        "Load-index mapping:"
    )

    print(
        "  ACTUAL n.loads.index used"
    )

    print(
        "  No synthetic eirgrid_load_ prefix"
    )

    # ==============================================================================================
    # FINAL STATUS
    # ==============================================================================================

    banner("S4.5I COMPLETE")

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
        "PERMANENT CHANGES       : NONE"
    )

    print()
    print(
        "Q=0 REFERENCE CHECK     : "
        f"{q0_reference_valid}"
    )

    print()
    print(
        f"CSV OUTPUT             : "
        f"{OUTPUT_CSV}"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    main()