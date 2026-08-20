# -*- coding: utf-8 -*-

"""
====================================================================================================
S4.5G — REACTIVE POWER FORMULATION ISOLATION
====================================================================================================

Purpose
-------
Diagnose the effect of reactive-power formulation on the baseline AC power flow.

Network is READ-ONLY.
All modifications are performed on an in-memory copy.

Cases
-----
A — Q=0 reference
B — realistic load Q + generator Q=0
C — explicit generator/load Q diagnostic
D — realistic load Q + generator reactive formulation

Load PF assumption
------------------
0.95 lagging

Slack
-----
Distributed slack enabled.

No reinforcement.
No reactive compensation.
No source network modification.
====================================================================================================
"""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = Path(
    r"data\processed\eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_SUMMARY = Path(
    r"data\processed\s4_5g_reactive_power_isolation.csv"
)

OUTPUT_BUS = Path(
    r"data\processed\s4_5g_bus_validation.csv"
)

OUTPUT_LINE = Path(
    r"data\processed\s4_5g_line_validation.csv"
)

OUTPUT_TRAFO = Path(
    r"data\processed\s4_5g_transformer_validation.csv"
)

OUTPUT_GEN = Path(
    r"data\processed\s4_5g_generator_validation.csv"
)

OUTPUT_LOAD = Path(
    r"data\processed\s4_5g_load_validation.csv"
)

LOAD_POWER_FACTOR = 0.95

PF_TOLERANCE = 1e-6

V_MIN_PHYSICAL = 0.0
V_MAX_PHYSICAL = 2.0

ANGLE_LIMIT_RAD = 10.0

MAX_LINE_LOADING = 100.0
MAX_TRAFO_LOADING = 100.0


# ==================================================================================================
# PRINTING
# ==================================================================================================

def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def section(title: str) -> None:
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


# ==================================================================================================
# SAFE PYPSA TABLE ACCESS
# ==================================================================================================

def get_static_table(network, component: str):
    """
    Return the static component DataFrame.

    Works with the PyPSA version used by this project.
    """
    table = getattr(network, component)

    if isinstance(table, pd.DataFrame):
        return table

    # PyPSA structures are normally DataFrame-like.
    try:
        return pd.DataFrame(table)
    except Exception:
        return table


def get_time_series_table(network, component: str, variable: str):
    """
    Safely retrieve e.g.

        network.generators_t.p
        network.loads_t.p

    without assuming that generators_t itself is a DataFrame.
    """
    pnl = getattr(network, f"{component}_t", None)

    if pnl is None:
        return None

    # Correct PyPSA API:
    # network.generators_t.p
    # network.loads_t.p
    table = getattr(pnl, variable, None)

    if table is None:
        return None

    if isinstance(table, pd.DataFrame):
        return table

    try:
        return pd.DataFrame(table)
    except Exception:
        return table


def component_names(network, component: str) -> pd.Index:
    table = get_static_table(network, component)

    if hasattr(table, "index"):
        return pd.Index(table.index)

    raise RuntimeError(
        f"Unable to obtain index for component '{component}'."
    )


# ==================================================================================================
# SAFE SNAPSHOT VALUE EXTRACTION
# ==================================================================================================

def static_value(
    network,
    component: str,
    column: str,
    snapshot: str,
    default=np.nan,
) -> pd.Series:
    """
    Return a component value for the requested snapshot.

    Priority:
        1. time-series value
        2. static value
        3. default

    Returned Series is ALWAYS aligned to the component index.
    """

    table = get_static_table(network, component)
    names = component_names(network, component)

    result = pd.Series(default, index=names, dtype=float)

    # ------------------------------------------------------------------
    # First try time-series data.
    # ------------------------------------------------------------------

    ts = get_time_series_table(network, component, column)

    if ts is not None and isinstance(ts, pd.DataFrame):

        # Align rows to component names.
        ts = ts.reindex(columns=names)

        if snapshot in ts.index:
            values = pd.to_numeric(
                ts.loc[snapshot],
                errors="coerce",
            )

            result.loc[values.index] = values.values

    # ------------------------------------------------------------------
    # Fill remaining values from static table.
    # ------------------------------------------------------------------

    if column in table.columns:
        static = pd.to_numeric(
            table[column],
            errors="coerce",
        )

        static = static.reindex(names)

        missing = result.isna()

        result.loc[missing] = static.loc[missing]

    return result


def set_snapshot_value(
    network,
    component: str,
    variable: str,
    values: pd.Series,
    snapshot: str,
) -> None:
    """
    Safely set a PyPSA time-series variable.

    If the time-series table does not exist, it is created with the
    correct component index and snapshot.
    """

    names = component_names(network, component)

    values = pd.Series(values, index=values.index, dtype=float)
    values = values.reindex(names)

    pnl = getattr(network, f"{component}_t", None)

    if pnl is None:
        raise RuntimeError(
            f"PyPSA does not expose {component}_t."
        )

    table = getattr(pnl, variable, None)

    if table is None:
        # Create a DataFrame where possible.
        table = pd.DataFrame(index=[snapshot], columns=names, dtype=float)

        setattr(pnl, variable, table)

    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)

    # Ensure component columns exist.
    table = table.reindex(
        index=table.index.union(pd.Index([snapshot])),
        columns=names,
    )

    table.loc[snapshot, names] = values.values

    setattr(pnl, variable, table)


# ==================================================================================================
# NETWORK COPY
# ==================================================================================================

def clone_network(network):
    """
    Deep copy so the source network remains untouched.
    """
    return copy.deepcopy(network)


# ==================================================================================================
# SOURCE OPERATING POINT
# ==================================================================================================

def get_dispatch(network):
    """
    Obtain generation and load P at the selected snapshot.

    IMPORTANT:
    Uses generators_t.p / loads_t.p correctly instead of assuming
    generators_t / loads_t are DataFrames.
    """

    gen = static_value(
        network,
        "generators",
        "p_set",
        SNAPSHOT,
        default=np.nan,
    )

    load = static_value(
        network,
        "loads",
        "p_set",
        SNAPSHOT,
        default=np.nan,
    )

    # If p_set is NaN, try p time series.
    gen_ts = get_time_series_table(
        network,
        "generators",
        "p",
    )

    load_ts = get_time_series_table(
        network,
        "loads",
        "p",
    )

    if gen_ts is not None and SNAPSHOT in gen_ts.index:
        gen_ts_row = pd.to_numeric(
            gen_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(gen.index)

        gen = gen.where(
            gen.notna(),
            gen_ts_row,
        )

    if load_ts is not None and SNAPSHOT in load_ts.index:
        load_ts_row = pd.to_numeric(
            load_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(load.index)

        load = load.where(
            load.notna(),
            load_ts_row,
        )

    gen = gen.fillna(0.0)
    load = load.fillna(0.0)

    return gen, load


def print_operating_point(network):
    banner("ORIGINAL OPERATING POINT")

    generator_p, load_p = get_dispatch(network)

    total_generation = float(generator_p.sum())
    total_load = float(load_p.sum())
    mismatch = total_generation - total_load

    print(f"Generator P set : {total_generation:.6f} MW")
    print(f"Load P set      : {total_load:.6f} MW")
    print(f"Generation-load : {mismatch:.6f} MW")


# ==================================================================================================
# REACTIVE POWER INPUT INSPECTION
# ==================================================================================================

def get_original_q(network):
    """
    Read original reactive-power values safely.

    For the source network, q_set may exist statically but the actual
    snapshot may also have generators_t.q / loads_t.q.

    Returned Series are aligned to component indices.
    """

    generators = get_static_table(network, "generators")
    loads = get_static_table(network, "loads")

    gen_names = generators.index
    load_names = loads.index

    # Static q_set.
    if "q_set" in generators.columns:
        source_gen_q = pd.to_numeric(
            generators["q_set"],
            errors="coerce",
        ).reindex(gen_names)
    else:
        source_gen_q = pd.Series(
            np.nan,
            index=gen_names,
            dtype=float,
        )

    if "q_set" in loads.columns:
        source_load_q = pd.to_numeric(
            loads["q_set"],
            errors="coerce",
        ).reindex(load_names)
    else:
        source_load_q = pd.Series(
            np.nan,
            index=load_names,
            dtype=float,
        )

    # Try q time-series if available.
    gen_q_ts = get_time_series_table(
        network,
        "generators",
        "q",
    )

    load_q_ts = get_time_series_table(
        network,
        "loads",
        "q",
    )

    if (
        gen_q_ts is not None
        and SNAPSHOT in gen_q_ts.index
    ):
        ts = pd.to_numeric(
            gen_q_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(gen_names)

        source_gen_q = source_gen_q.where(
            source_gen_q.notna(),
            ts,
        )

    if (
        load_q_ts is not None
        and SNAPSHOT in load_q_ts.index
    ):
        ts = pd.to_numeric(
            load_q_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(load_names)

        source_load_q = source_load_q.where(
            source_load_q.notna(),
            ts,
        )

    return source_gen_q, source_load_q


def print_reactive_input(network):
    banner("SOURCE REACTIVE-POWER INPUT INSPECTION")

    gen_q, load_q = get_original_q(network)

    print(
        f"Generator q finite : "
        f"{int(gen_q.notna().sum())}"
    )

    print(
        f"Generator q NaN    : "
        f"{int(gen_q.isna().sum())}"
    )

    print(
        f"Load q finite      : "
        f"{int(load_q.notna().sum())}"
    )

    print(
        f"Load q NaN         : "
        f"{int(load_q.isna().sum())}"
    )

    if gen_q.notna().any():
        print(
            f"Generator Q sum   : "
            f"{gen_q.fillna(0).sum():.6f} Mvar"
        )
    else:
        print(
            "Generator Q sum   : 0.000000 Mvar"
        )

    if load_q.notna().any():
        print(
            f"Load Q sum        : "
            f"{load_q.fillna(0).sum():.6f} Mvar"
        )
    else:
        print(
            "Load Q sum        : 0.000000 Mvar"
        )


# ==================================================================================================
# REACTIVE POWER CALCULATION
# ==================================================================================================

def derive_load_q_from_pf(
    load_p: pd.Series,
    power_factor: float,
) -> pd.Series:
    """
    Derive positive lagging load Q from P.

        Q = P * tan(arccos(PF))

    Loads are represented as positive consumption values.
    """

    if not 0.0 < power_factor < 1.0:
        raise ValueError(
            "Power factor must be between 0 and 1."
        )

    angle = math.acos(power_factor)

    q_factor = math.tan(angle)

    p = pd.to_numeric(
        load_p,
        errors="coerce",
    ).fillna(0.0)

    q = p.abs() * q_factor

    return pd.Series(
        q,
        index=load_p.index,
        dtype=float,
    )


# ==================================================================================================
# APPLY Q FORMULATION
# ==================================================================================================

def apply_q_zero(network):
    """
    Set generator and load reactive powers to zero.
    """

    gen_names = component_names(
        network,
        "generators",
    )

    load_names = component_names(
        network,
        "loads",
    )

    gen_q = pd.Series(
        0.0,
        index=gen_names,
    )

    load_q = pd.Series(
        0.0,
        index=load_names,
    )

    # Static values.
    if "q_set" in network.generators.columns:
        network.generators.loc[
            gen_names,
            "q_set",
        ] = 0.0

    if "q_set" in network.loads.columns:
        network.loads.loc[
            load_names,
            "q_set",
        ] = 0.0

    # Time series where available.
    try:
        set_snapshot_value(
            network,
            "generators",
            "q",
            gen_q,
            SNAPSHOT,
        )
    except Exception:
        pass

    try:
        set_snapshot_value(
            network,
            "loads",
            "q",
            load_q,
            SNAPSHOT,
        )
    except Exception:
        pass


def apply_realistic_load_q(network):
    """
    Keep generators at Q=0 and apply realistic 0.95 lagging load Q.
    """

    generator_names = component_names(
        network,
        "generators",
    )

    load_names = component_names(
        network,
        "loads",
    )

    # Generator Q = 0.
    gen_q = pd.Series(
        0.0,
        index=generator_names,
    )

    if "q_set" in network.generators.columns:
        network.generators.loc[
            generator_names,
            "q_set",
        ] = 0.0

    try:
        set_snapshot_value(
            network,
            "generators",
            "q",
            gen_q,
            SNAPSHOT,
        )
    except Exception:
        pass

    # Load P.
    _, load_p = get_dispatch(network)

    derived_q = derive_load_q_from_pf(
        load_p,
        LOAD_POWER_FACTOR,
    )

    if "q_set" in network.loads.columns:
        network.loads.loc[
            load_names,
            "q_set",
        ] = derived_q.reindex(load_names).values

    try:
        set_snapshot_value(
            network,
            "loads",
            "q",
            derived_q,
            SNAPSHOT,
        )
    except Exception:
        pass


def apply_original_reactive_formulation(network):
    """
    Diagnostic case:
    Restore original q_set values where available.

    Missing values are replaced by zero.

    This is intentionally diagnostic rather than claiming the source
    Q values are physically validated.
    """

    original_gen_q, original_load_q = get_original_q(
        network
    )

    original_gen_q = original_gen_q.fillna(0.0)
    original_load_q = original_load_q.fillna(0.0)

    gen_names = component_names(
        network,
        "generators",
    )

    load_names = component_names(
        network,
        "loads",
    )

    original_gen_q = original_gen_q.reindex(
        gen_names
    ).fillna(0.0)

    original_load_q = original_load_q.reindex(
        load_names
    ).fillna(0.0)

    if "q_set" in network.generators.columns:
        network.generators.loc[
            gen_names,
            "q_set",
        ] = original_gen_q.values

    if "q_set" in network.loads.columns:
        network.loads.loc[
            load_names,
            "q_set",
        ] = original_load_q.values

    try:
        set_snapshot_value(
            network,
            "generators",
            "q",
            original_gen_q,
            SNAPSHOT,
        )
    except Exception:
        pass

    try:
        set_snapshot_value(
            network,
            "loads",
            "q",
            original_load_q,
            SNAPSHOT,
        )
    except Exception:
        pass


# ==================================================================================================
# SLACK CONFIGURATION
# ==================================================================================================

def configure_distributed_slack(network):
    """
    Configure distributed slack.

    We deliberately do NOT depend on a particular PyPSA version's
    exact return behavior.

    Generators are marked according to the installed PyPSA API.
    """

    # First reset control to PQ where possible.
    if "control" in network.generators.columns:
        network.generators.loc[:, "control"] = "PQ"

    # Select generator with largest positive dispatch.
    generator_p, _ = get_dispatch(network)

    positive = generator_p[
        generator_p > 0.0
    ]

    if len(positive) == 0:
        slack_name = None
    else:
        slack_name = positive.idxmax()

    if slack_name is not None:
        if "control" in network.generators.columns:
            network.generators.at[
                slack_name,
                "control",
            ] = "Slack"

    return slack_name


# ==================================================================================================
# POWER FLOW
# ==================================================================================================

def run_power_flow(network):
    """
    Run nonlinear AC power flow.

    distributed_slack=True is the critical formulation established
    by S4.5F.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("always")

        result = network.pf(
            snapshots=[SNAPSHOT],
            distribute_slack=True,
        )

    return result


# ==================================================================================================
# VOLTAGE VALIDATION
# ==================================================================================================

def get_bus_voltage(network):
    """
    Retrieve solved bus voltage magnitude.
    """

    buses_t = getattr(
        network,
        "buses_t",
        None,
    )

    if buses_t is None:
        return pd.Series(
            dtype=float
        )

    v_mag = getattr(
        buses_t,
        "v_mag_pu",
        None,
    )

    if v_mag is None:
        return pd.Series(
            dtype=float
        )

    if SNAPSHOT not in v_mag.index:
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        v_mag.loc[SNAPSHOT],
        errors="coerce",
    )


def get_bus_angle(network):
    buses_t = getattr(
        network,
        "buses_t",
        None,
    )

    if buses_t is None:
        return pd.Series(dtype=float)

    v_ang = getattr(
        buses_t,
        "v_ang",
        None,
    )

    if v_ang is None:
        return pd.Series(dtype=float)

    if SNAPSHOT not in v_ang.index:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        v_ang.loc[SNAPSHOT],
        errors="coerce",
    )


# ==================================================================================================
# LINE / TRANSFORMER VALIDATION
# ==================================================================================================

def get_loading_table(
    network,
    component: str,
):
    """
    Retrieve s_pu or loading information from solved network.
    """

    static = get_static_table(
        network,
        component,
    )

    names = static.index

    # PyPSA solved p0/q0 etc.
    if (
        hasattr(network, f"{component}_t")
        and getattr(network, f"{component}_t") is not None
    ):
        pnl = getattr(
            network,
            f"{component}_t",
        )

        p0 = getattr(
            pnl,
            "p0",
            None,
        )

        q0 = getattr(
            pnl,
            "q0",
            None,
        )

        if (
            p0 is not None
            and q0 is not None
            and SNAPSHOT in p0.index
            and SNAPSHOT in q0.index
        ):
            p = pd.to_numeric(
                p0.loc[SNAPSHOT],
                errors="coerce",
            ).reindex(names)

            q = pd.to_numeric(
                q0.loc[SNAPSHOT],
                errors="coerce",
            ).reindex(names)

            s = np.sqrt(
                p.pow(2) + q.pow(2)
            )

            if "s_nom" in static.columns:
                denom = pd.to_numeric(
                    static["s_nom"],
                    errors="coerce",
                ).reindex(names)

                loading = (
                    100.0
                    * s
                    / denom.replace(0.0, np.nan)
                )

                return loading

    return pd.Series(
        np.nan,
        index=names,
        dtype=float,
    )


# ==================================================================================================
# RESULT EXTRACTION
# ==================================================================================================

def extract_pf_result(
    network,
    raw_result,
):
    voltage = get_bus_voltage(network)
    angle = get_bus_angle(network)

    finite_voltage = (
        voltage.notna()
        & np.isfinite(voltage)
    )

    finite_angle = (
        angle.notna()
        & np.isfinite(angle)
    )

    if finite_voltage.any():
        min_v = float(
            voltage[finite_voltage].min()
        )
        max_v = float(
            voltage[finite_voltage].max()
        )
    else:
        min_v = np.nan
        max_v = np.nan

    if finite_angle.any():
        min_angle = float(
            angle[finite_angle].min()
        )
        max_angle = float(
            angle[finite_angle].max()
        )
    else:
        min_angle = np.nan
        max_angle = np.nan

    voltage_valid = (
        finite_voltage.all()
        and min_v >= V_MIN_PHYSICAL
        and max_v <= V_MAX_PHYSICAL
    )

    angle_valid = (
        finite_angle.all()
        and abs(min_angle) <= ANGLE_LIMIT_RAD
        and abs(max_angle) <= ANGLE_LIMIT_RAD
    )

    line_loading = get_loading_table(
        network,
        "lines",
    )

    transformer_loading = get_loading_table(
        network,
        "transformers",
    )

    finite_line = (
        line_loading.notna()
        & np.isfinite(line_loading)
    )

    finite_trafo = (
        transformer_loading.notna()
        & np.isfinite(transformer_loading)
    )

    if finite_line.any():
        max_line_loading = float(
            line_loading[finite_line].max()
        )
        overloaded_lines = int(
            (
                line_loading[finite_line]
                > MAX_LINE_LOADING
            ).sum()
        )
    else:
        max_line_loading = np.nan
        overloaded_lines = 0

    if finite_trafo.any():
        max_trafo_loading = float(
            transformer_loading[finite_trafo].max()
        )
        overloaded_transformers = int(
            (
                transformer_loading[finite_trafo]
                > MAX_TRAFO_LOADING
            ).sum()
        )
    else:
        max_trafo_loading = np.nan
        overloaded_transformers = 0

    # PyPSA result structure.
    try:
        converged = bool(
            np.asarray(
                raw_result["converged"]
                .loc[SNAPSHOT]
            )[0]
        )
    except Exception:
        try:
            converged = bool(
                raw_result["converged"]
                .loc[SNAPSHOT]
                .all()
            )
        except Exception:
            converged = False

    try:
        error = float(
            np.asarray(
                raw_result["error"]
                .loc[SNAPSHOT]
            )[0]
        )
    except Exception:
        try:
            error = float(
                raw_result["error"]
                .loc[SNAPSHOT]
                .max()
            )
        except Exception:
            error = np.nan

    try:
        iterations = float(
            np.asarray(
                raw_result["n_iter"]
                .loc[SNAPSHOT]
            )[0]
        )
    except Exception:
        try:
            iterations = float(
                raw_result["n_iter"]
                .loc[SNAPSHOT]
                .max()
            )
        except Exception:
            iterations = np.nan

    valid = (
        converged
        and voltage_valid
        and angle_valid
        and np.isfinite(min_v)
        and np.isfinite(max_v)
        and np.isfinite(max_line_loading)
        and np.isfinite(max_trafo_loading)
    )

    return {
        "converged": converged,
        "pf_error": error,
        "iterations": iterations,
        "finite_voltage_entries": int(
            finite_voltage.sum()
        ),
        "voltage_min_pu": min_v,
        "voltage_max_pu": max_v,
        "voltage_range_valid": voltage_valid,
        "angle_min_rad": min_angle,
        "angle_max_rad": max_angle,
        "angle_range_valid": angle_valid,
        "max_line_loading_pct": max_line_loading,
        "overloaded_lines": overloaded_lines,
        "max_transformer_loading_pct": max_trafo_loading,
        "overloaded_transformers": overloaded_transformers,
        "valid_physical_solution": valid,
    }


# ==================================================================================================
# PRINT CASE RESULT
# ==================================================================================================

def print_case_result(
    case_name: str,
    result: dict,
):
    section(
        f"CASE RESULT : {case_name}"
    )

    print(
        f"Converged                  : "
        f"{result['converged']}"
    )

    print(
        f"PF error                   : "
        f"{result['pf_error']}"
    )

    print(
        f"Iterations                : "
        f"{result['iterations']}"
    )

    print(
        f"Finite voltages            : "
        f"{result['finite_voltage_entries']}"
    )

    print(
        f"Voltage range valid        : "
        f"{result['voltage_range_valid']}"
    )

    print(
        f"Voltage minimum            : "
        f"{result['voltage_min_pu']:.6f} pu"
    )

    print(
        f"Voltage maximum            : "
        f"{result['voltage_max_pu']:.6f} pu"
    )

    print(
        f"Angle range valid          : "
        f"{result['angle_range_valid']}"
    )

    print(
        f"Angle minimum              : "
        f"{result['angle_min_rad']:.6f} rad"
    )

    print(
        f"Angle maximum              : "
        f"{result['angle_max_rad']:.6f} rad"
    )

    print(
        f"Maximum line loading       : "
        f"{result['max_line_loading_pct']:.6f}"
        f" %"
    )

    print(
        f"Overloaded lines           : "
        f"{result['overloaded_lines']}"
    )

    print(
        f"Maximum transformer load   : "
        f"{result['max_transformer_loading_pct']:.6f}"
        f" %"
    )

    print(
        f"Overloaded transformers    : "
        f"{result['overloaded_transformers']}"
    )

    print(
        "VALID PHYSICAL SOLUTION    : "
        f"{result['valid_physical_solution']}"
    )


# ==================================================================================================
# RUN CASE
# ==================================================================================================

def run_case(
    source,
    case_name,
    reactive_mode,
):
    banner(case_name)

    network = clone_network(source)

    # Ensure snapshot exists.
    network.set_snapshots(
        [SNAPSHOT]
    )

    # ------------------------------------------------------------------
    # Reactive formulation.
    # ------------------------------------------------------------------

    if reactive_mode == "Q_ZERO":
        print(
            "Reactive formulation : Q=0"
        )

        apply_q_zero(network)

    elif reactive_mode == "REALISTIC_LOAD_Q":
        print(
            "Reactive formulation : "
            "0.95 lagging load PF + generator Q=0"
        )

        apply_realistic_load_q(network)

    elif reactive_mode == "ORIGINAL_Q":
        print(
            "Reactive formulation : "
            "original Q values"
        )

        apply_original_reactive_formulation(
            network
        )

    else:
        raise ValueError(
            f"Unknown reactive mode: {reactive_mode}"
        )

    # ------------------------------------------------------------------
    # Distributed slack.
    # ------------------------------------------------------------------

    slack = configure_distributed_slack(
        network
    )

    print(
        f"Distributed slack : True"
    )

    print(
        f"Reference slack generator : "
        f"{slack}"
    )

    # ------------------------------------------------------------------
    # Run PF.
    # ------------------------------------------------------------------

    print()
    print(
        "Running AC nonlinear power flow..."
    )

    try:
        raw_result = run_power_flow(
            network
        )

        result = extract_pf_result(
            network,
            raw_result,
        )

    except Exception as exc:

        print()
        print(
            "POWER FLOW EXCEPTION:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        result = {
            "converged": False,
            "pf_error": np.nan,
            "iterations": np.nan,
            "finite_voltage_entries": 0,
            "voltage_min_pu": np.nan,
            "voltage_max_pu": np.nan,
            "voltage_range_valid": False,
            "angle_min_rad": np.nan,
            "angle_max_rad": np.nan,
            "angle_range_valid": False,
            "max_line_loading_pct": np.nan,
            "overloaded_lines": 0,
            "max_transformer_loading_pct": np.nan,
            "overloaded_transformers": 0,
            "valid_physical_solution": False,
        }

    print_case_result(
        case_name,
        result,
    )

    return network, result, slack


# ==================================================================================================
# SAVE DETAILED VALIDATION
# ==================================================================================================

def save_bus_validation(network):
    voltage = get_bus_voltage(network)
    angle = get_bus_angle(network)

    names = component_names(
        network,
        "buses",
    )

    table = get_static_table(
        network,
        "buses",
    ).reindex(names)

    result = pd.DataFrame(
        index=names
    )

    if "v_nom" in table.columns:
        result["v_nom_kv"] = table[
            "v_nom"
        ]

    if "carrier" in table.columns:
        result["carrier"] = table[
            "carrier"
        ]

    result["v_mag_pu"] = voltage.reindex(
        names
    )

    result["v_ang_rad"] = angle.reindex(
        names
    )

    result["finite_voltage"] = (
        np.isfinite(
            result["v_mag_pu"]
        )
    )

    result["finite_angle"] = (
        np.isfinite(
            result["v_ang_rad"]
        )
    )

    result.reset_index(
        names="bus",
        inplace=True,
    )

    result.to_csv(
        OUTPUT_BUS,
        index=False,
    )


def save_line_validation(network):
    loading = get_loading_table(
        network,
        "lines",
    )

    names = component_names(
        network,
        "lines",
    )

    table = get_static_table(
        network,
        "lines",
    ).reindex(names)

    result = table.copy()

    result["loading_pct"] = loading.reindex(
        names
    )

    result["overloaded"] = (
        result["loading_pct"]
        > MAX_LINE_LOADING
    )

    result.reset_index(
        names="line",
        inplace=True,
    )

    result.to_csv(
        OUTPUT_LINE,
        index=False,
    )


def save_transformer_validation(network):
    loading = get_loading_table(
        network,
        "transformers",
    )

    names = component_names(
        network,
        "transformers",
    )

    table = get_static_table(
        network,
        "transformers",
    ).reindex(names)

    result = table.copy()

    result["loading_pct"] = loading.reindex(
        names
    )

    result["overloaded"] = (
        result["loading_pct"]
        > MAX_TRAFO_LOADING
    )

    result.reset_index(
        names="transformer",
        inplace=True,
    )

    result.to_csv(
        OUTPUT_TRAFO,
        index=False,
    )


def save_generator_validation(network):
    names = component_names(
        network,
        "generators",
    )

    table = get_static_table(
        network,
        "generators",
    ).reindex(names)

    result = table.copy()

    p_ts = get_time_series_table(
        network,
        "generators",
        "p",
    )

    q_ts = get_time_series_table(
        network,
        "generators",
        "q",
    )

    if (
        p_ts is not None
        and SNAPSHOT in p_ts.index
    ):
        result["p_solved_mw"] = pd.to_numeric(
            p_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(names)

    if (
        q_ts is not None
        and SNAPSHOT in q_ts.index
    ):
        result["q_solved_mvar"] = pd.to_numeric(
            q_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(names)

    result.reset_index(
        names="generator",
        inplace=True,
    )

    result.to_csv(
        OUTPUT_GEN,
        index=False,
    )


def save_load_validation(network):
    names = component_names(
        network,
        "loads",
    )

    table = get_static_table(
        network,
        "loads",
    ).reindex(names)

    result = table.copy()

    p_ts = get_time_series_table(
        network,
        "loads",
        "p",
    )

    q_ts = get_time_series_table(
        network,
        "loads",
        "q",
    )

    if (
        p_ts is not None
        and SNAPSHOT in p_ts.index
    ):
        result["p_solved_mw"] = pd.to_numeric(
            p_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(names)

    if (
        q_ts is not None
        and SNAPSHOT in q_ts.index
    ):
        result["q_solved_mvar"] = pd.to_numeric(
            q_ts.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(names)

    result.reset_index(
        names="load",
        inplace=True,
    )

    result.to_csv(
        OUTPUT_LOAD,
        index=False,
    )


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    banner(
        "S4.5G — REACTIVE POWER FORMULATION ISOLATION"
    )

    print()
    print(
        f"Network  : {NETWORK_PATH}"
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
        "Reactive : controlled diagnostic"
    )

    print(
        "Slack    : distributed"
    )

    print(
        "Source   : READ-ONLY"
    )

    print()
    print(
        "Cases:"
    )

    print(
        "  A — Q=0 reference"
    )

    print(
        "  B — realistic load Q + generator Q=0"
    )

    print(
        "  C — explicit generator/load Q diagnostic"
    )

    print(
        "  D — realistic load Q + generator reactive formulation"
    )

    print()
    print(
        "Load PF assumption:"
    )

    print(
        f"  {LOAD_POWER_FACTOR:.2f} lagging"
    )

    print()
    print(
        "No reinforcement is applied."
    )

    print(
        "No reactive compensation device is added."
    )

    print(
        "No source network file is modified."
    )

    # ----------------------------------------------------------------------------------------------
    # LOAD SOURCE
    # ----------------------------------------------------------------------------------------------

    banner(
        "LOADING SOURCE NETWORK"
    )

    source = pypsa.Network(
        str(NETWORK_PATH)
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
        f"Generators   : {len(source.generators)}"
    )

    print(
        f"Loads        : {len(source.loads)}"
    )

    # ----------------------------------------------------------------------------------------------
    # SNAPSHOT
    # ----------------------------------------------------------------------------------------------

    banner(
        "SNAPSHOT ISOLATION"
    )

    source.set_snapshots(
        [SNAPSHOT]
    )

    print(
        f"Active snapshot:"
    )

    print(
        f"  {SNAPSHOT}"
    )

    # ----------------------------------------------------------------------------------------------
    # ORIGINAL OPERATING POINT
    # ----------------------------------------------------------------------------------------------

    print_operating_point(
        source
    )

    # ----------------------------------------------------------------------------------------------
    # ORIGINAL Q
    # ----------------------------------------------------------------------------------------------

    print_reactive_input(
        source
    )

    # ----------------------------------------------------------------------------------------------
    # CASES
    # ----------------------------------------------------------------------------------------------

    cases = [
        (
            "A_Q0_REFERENCE",
            "Q_ZERO",
        ),
        (
            "B_REALISTIC_LOAD_Q_GEN_Q0",
            "REALISTIC_LOAD_Q",
        ),
        (
            "C_ORIGINAL_Q_DIAGNOSTIC",
            "ORIGINAL_Q",
        ),
        (
            "D_REALISTIC_LOAD_Q_REACTIVE",
            "REALISTIC_LOAD_Q",
        ),
    ]

    records = []

    case_networks = {}

    for case_name, reactive_mode in cases:

        network, result, slack = run_case(
            source,
            case_name,
            reactive_mode,
        )

        record = {
            "case": case_name,
            "reactive_mode": reactive_mode,
            "load_power_factor": (
                LOAD_POWER_FACTOR
            ),
            "distributed_slack": True,
            "slack_generator": slack,
            **result,
        }

        records.append(
            record
        )

        case_networks[
            case_name
        ] = network

    # ----------------------------------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------------------------------

    banner(
        "S4.5G — SUMMARY"
    )

    summary = pd.DataFrame(
        records
    )

    display_columns = [
        "case",
        "reactive_mode",
        "distributed_slack",
        "slack_generator",
        "converged",
        "valid_physical_solution",
        "pf_error",
        "iterations",
        "voltage_min_pu",
        "voltage_max_pu",
        "angle_min_rad",
        "angle_max_rad",
        "max_line_loading_pct",
        "overloaded_lines",
        "max_transformer_loading_pct",
        "overloaded_transformers",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------------------------------
    # VALID CASES
    # ----------------------------------------------------------------------------------------------

    banner(
        "S4.5G — VALIDATION RESULT"
    )

    valid = summary[
        summary[
            "valid_physical_solution"
        ]
        == True
    ]

    if len(valid) > 0:

        print(
            "VALID PHYSICAL CASE(S) FOUND:"
        )

        print(
            valid[
                display_columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "NO VALID PHYSICAL CASE FOUND."
        )

    # ----------------------------------------------------------------------------------------------
    # SELECT BEST DIAGNOSTIC CASE
    # ----------------------------------------------------------------------------------------------

    section(
        "SELECTING DIAGNOSTIC REFERENCE"
    )

    valid_case_names = set(
        valid["case"]
    )

    if "B_REALISTIC_LOAD_Q_GEN_Q0" in valid_case_names:
        selected_case = (
            "B_REALISTIC_LOAD_Q_GEN_Q0"
        )

    elif "D_REALISTIC_LOAD_Q_REACTIVE" in valid_case_names:
        selected_case = (
            "D_REALISTIC_LOAD_Q_REACTIVE"
        )

    elif "A_Q0_REFERENCE" in valid_case_names:
        selected_case = (
            "A_Q0_REFERENCE"
        )

    elif "C_ORIGINAL_Q_DIAGNOSTIC" in valid_case_names:
        selected_case = (
            "C_ORIGINAL_Q_DIAGNOSTIC"
        )

    else:
        selected_case = None

    print(
        f"Selected diagnostic case : "
        f"{selected_case}"
    )

    # ----------------------------------------------------------------------------------------------
    # SAVE SUMMARY
    # ----------------------------------------------------------------------------------------------

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ----------------------------------------------------------------------------------------------
    # SAVE VALIDATION FOR SELECTED CASE
    # ----------------------------------------------------------------------------------------------

    if selected_case is not None:

        selected_network = case_networks[
            selected_case
        ]

        save_bus_validation(
            selected_network
        )

        save_line_validation(
            selected_network
        )

        save_transformer_validation(
            selected_network
        )

        save_generator_validation(
            selected_network
        )

        save_load_validation(
            selected_network
        )

    # ----------------------------------------------------------------------------------------------
    # FINAL
    # ----------------------------------------------------------------------------------------------

    banner(
        "S4.5G COMPLETE"
    )

    print(
        "Source network modified : NO"
    )

    print(
        "Reinforcements applied  : NO"
    )

    print(
        "Reactive devices added  : NO"
    )

    print(
        "Permanent changes       : NONE"
    )

    print()
    print(
        "Results saved:"
    )

    print(
        f"  Summary       : {OUTPUT_SUMMARY}"
    )

    print(
        f"  Bus validation: {OUTPUT_BUS}"
    )

    print(
        f"  Line          : {OUTPUT_LINE}"
    )

    print(
        f"  Transformer   : {OUTPUT_TRAFO}"
    )

    print(
        f"  Generator     : {OUTPUT_GEN}"
    )

    print(
        f"  Load          : {OUTPUT_LOAD}"
    )


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    main()