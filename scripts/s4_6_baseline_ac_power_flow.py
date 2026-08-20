# -*- coding: utf-8 -*-

"""
====================================================================================================
S4.6 — BASELINE AC POWER-FLOW ESTABLISHMENT
====================================================================================================

Purpose
-------
Establish a clean diagnostic AC nonlinear power-flow baseline using the READ-ONLY
reinforced network produced by the previous stages.

Important:
    - Source network is NEVER modified on disk.
    - No reinforcement is applied.
    - No reactive compensation is added.
    - No dispatch optimisation is performed.
    - Original source reactive power is preserved.
    - Active operating-point values are read from the selected snapshot.
    - Initial voltage conditions are read from static bus values.
    - Solved AC voltages/angles are read from buses_t.
    - Line/transformer solved flows are read from their *_t tables.

Network
-------
data/processed/eirgrid_second_reinforced_network.nc

Snapshot
--------
S2_PEAK_DEMAND

Outputs
-------
data/processed/s4_6_baseline_ac_power_flow.csv
data/processed/s4_6_bus_validation.csv
data/processed/s4_6_line_validation.csv
data/processed/s4_6_transformer_validation.csv
data/processed/s4_6_generator_validation.csv
data/processed/s4_6_load_validation.csv
====================================================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


# ================================================================================================
# CONFIGURATION
# ================================================================================================

NETWORK_PATH = Path(
    "data",
    "processed",
    "eirgrid_second_reinforced_network.nc",
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_DIR = Path("data", "processed")

SUMMARY_PATH = OUTPUT_DIR / "s4_6_baseline_ac_power_flow.csv"
BUS_PATH = OUTPUT_DIR / "s4_6_bus_validation.csv"
LINE_PATH = OUTPUT_DIR / "s4_6_line_validation.csv"
TRANSFORMER_PATH = OUTPUT_DIR / "s4_6_transformer_validation.csv"
GENERATOR_PATH = OUTPUT_DIR / "s4_6_generator_validation.csv"
LOAD_PATH = OUTPUT_DIR / "s4_6_load_validation.csv"

VOLTAGE_MIN_LIMIT = 0.90
VOLTAGE_MAX_LIMIT = 1.10

ANGLE_MIN_LIMIT = -np.pi
ANGLE_MAX_LIMIT = np.pi

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0

PF_ERROR_LIMIT = 1e-5


# ================================================================================================
# PRINTING
# ================================================================================================

def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def subbanner(title: str) -> None:
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)


# ================================================================================================
# PYPSA COMPATIBILITY HELPERS
# ================================================================================================

def get_static(network, component: str) -> pd.DataFrame:
    """
    Return the static dataframe for a PyPSA component.

    PyPSA >= 1.0:
        network.components[component].static

    Fallback:
        network.<component>
    """

    try:
        static = network.components[component].static
    except Exception:
        # Compatibility fallback for older PyPSA versions.
        mapping = {
            "Bus": "buses",
            "Generator": "generators",
            "Load": "loads",
            "Line": "lines",
            "Transformer": "transformers",
            "Link": "links",
        }

        attr = mapping[component]
        static = getattr(network, attr)

    if not isinstance(static, pd.DataFrame):
        static = pd.DataFrame(static)

    return static


def get_dynamic_table(network, component: str, attribute: str) -> pd.DataFrame:
    """
    Return a PyPSA dynamic time-series table.

    Example:
        Generator -> p
        Load      -> p
        Load      -> q
        Bus       -> v_mag_pu
        Bus       -> v_ang
        Line      -> p0 / p1 / q0 / q1
        Transformer -> p0 / p1 / q0 / q1
    """

    mapping = {
        "Generator": network.generators_t,
        "Load": network.loads_t,
        "Bus": network.buses_t,
        "Line": network.lines_t,
        "Transformer": network.transformers_t,
        "Link": network.links_t,
    }

    container = mapping[component]

    if not hasattr(container, attribute):
        return pd.DataFrame()

    table = getattr(container, attribute)

    if table is None:
        return pd.DataFrame()

    return table


def get_snapshot_series(
    network,
    component: str,
    attribute: str,
    snapshot: str,
) -> pd.Series:
    """
    Safely retrieve a component time-series at the requested snapshot.

    Returns a Series indexed by the component names.

    If the dynamic table does not exist, returns an empty Series.
    """

    table = get_dynamic_table(network, component, attribute)

    if table is None or table.empty:
        return pd.Series(dtype=float)

    if snapshot not in table.index:
        raise RuntimeError(
            f"Snapshot '{snapshot}' not found in {component} time series "
            f"for attribute '{attribute}'. "
            f"Available snapshots: {list(table.index)}"
        )

    values = table.loc[snapshot]

    if isinstance(values, pd.DataFrame):
        values = values.iloc[0]

    return pd.to_numeric(values, errors="coerce")


def get_active_component_value(
    network,
    component: str,
    attribute: str,
    snapshot: str,
) -> pd.Series:
    """
    Retrieve the actual active operating-point values.

    Priority:
        1. Time-series value at requested snapshot.
        2. Static value if no dynamic table exists.

    This is essential because the source network uses snapshot-dependent
    dispatch/load values.
    """

    static = get_static(network, component)

    dynamic = get_snapshot_series(
        network,
        component,
        attribute,
        snapshot,
    )

    if not dynamic.empty:
        result = dynamic.reindex(static.index)
        return pd.to_numeric(result, errors="coerce")

    if attribute in static.columns:
        return pd.to_numeric(
            static[attribute],
            errors="coerce",
        )

    return pd.Series(
        np.nan,
        index=static.index,
        dtype=float,
    )


def get_solved_component_value(
    network,
    component: str,
    attribute: str,
    snapshot: str,
) -> pd.Series:
    """
    Retrieve solved AC power-flow values from *_t tables.

    If unavailable, return NaN series aligned with the static component index.
    """

    static = get_static(network, component)

    dynamic = get_snapshot_series(
        network,
        component,
        attribute,
        snapshot,
    )

    if dynamic.empty:
        return pd.Series(
            np.nan,
            index=static.index,
            dtype=float,
        )

    return pd.to_numeric(
        dynamic.reindex(static.index),
        errors="coerce",
    )


# ================================================================================================
# BASIC NETWORK INFORMATION
# ================================================================================================

def print_network_information(network) -> None:

    print(f"Buses        : {len(get_static(network, 'Bus'))}")
    print(f"Lines        : {len(get_static(network, 'Line'))}")
    print(f"Transformers : {len(get_static(network, 'Transformer'))}")
    print(f"Generators   : {len(get_static(network, 'Generator'))}")
    print(f"Loads        : {len(get_static(network, 'Load'))}")


# ================================================================================================
# SNAPSHOT VALIDATION
# ================================================================================================

def validate_snapshot(network) -> None:

    snapshots = list(network.snapshots)

    if SNAPSHOT not in snapshots:
        raise RuntimeError(
            f"Required snapshot '{SNAPSHOT}' does not exist.\n"
            f"Available snapshots:\n{snapshots}"
        )

    network.set_snapshots([SNAPSHOT])

    print(f"Active snapshot:")
    print(f"  {SNAPSHOT}")


# ================================================================================================
# OPERATING POINT
# ================================================================================================

def get_operating_point(network):
    """
    Read P/Q operating-point values from the active snapshot.

    IMPORTANT:
    This deliberately does NOT use static generator/load p_set when a
    time-series value exists.
    """

    generator_p = get_active_component_value(
        network,
        "Generator",
        "p",
        SNAPSHOT,
    )

    load_p = get_active_component_value(
        network,
        "Load",
        "p",
        SNAPSHOT,
    )

    generator_q = get_active_component_value(
        network,
        "Generator",
        "q",
        SNAPSHOT,
    )

    load_q = get_active_component_value(
        network,
        "Load",
        "q",
        SNAPSHOT,
    )

    return generator_p, load_p, generator_q, load_q


def print_operating_point(network) -> None:

    banner("ORIGINAL OPERATING POINT")

    generator_p, load_p, generator_q, load_q = get_operating_point(network)

    generator_p_sum = float(generator_p.fillna(0.0).sum())
    load_p_sum = float(load_p.fillna(0.0).sum())

    generator_q_sum = float(generator_q.fillna(0.0).sum())
    load_q_sum = float(load_q.fillna(0.0).sum())

    print(f"Generator P set : {generator_p_sum:.6f} MW")
    print(f"Load P set      : {load_p_sum:.6f} MW")
    print(f"Generation-load : {generator_p_sum - load_p_sum:.6f} MW")

    print()

    print(f"Generator Q set : {generator_q_sum:.6f} Mvar")
    print(f"Load Q set      : {load_q_sum:.6f} Mvar")


# ================================================================================================
# REACTIVE POWER VALIDATION
# ================================================================================================

def validate_reactive_power(network) -> None:

    banner("ORIGINAL REACTIVE-POWER VALIDATION")

    _, _, generator_q, load_q = get_operating_point(network)

    generator_q_nan = int(generator_q.isna().sum())
    load_q_nan = int(load_q.isna().sum())

    generator_q_sum = float(generator_q.fillna(0.0).sum())
    load_q_sum = float(load_q.fillna(0.0).sum())

    print(f"Generator Q NaNs : {generator_q_nan}")
    print(f"Load Q NaNs      : {load_q_nan}")
    print(f"Generator Q sum  : {generator_q_sum:.6f} Mvar")
    print(f"Load Q sum       : {load_q_sum:.6f} Mvar")


# ================================================================================================
# AC TOPOLOGY
# ================================================================================================

def get_ac_connected_components(network):

    buses = get_static(network, "Bus")

    adjacency = {
        bus: set()
        for bus in buses.index
    }

    lines = get_static(network, "Line")

    for name, row in lines.iterrows():

        if "bus0" not in row or "bus1" not in row:
            continue

        bus0 = row["bus0"]
        bus1 = row["bus1"]

        if bus0 not in adjacency or bus1 not in adjacency:
            continue

        adjacency[bus0].add(bus1)
        adjacency[bus1].add(bus0)

    transformers = get_static(network, "Transformer")

    for name, row in transformers.iterrows():

        if "bus0" not in row or "bus1" not in row:
            continue

        bus0 = row["bus0"]
        bus1 = row["bus1"]

        if bus0 not in adjacency or bus1 not in adjacency:
            continue

        adjacency[bus0].add(bus1)
        adjacency[bus1].add(bus0)

    components = []

    visited = set()

    for start in adjacency:

        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:

            bus = stack.pop()

            if bus in visited:
                continue

            visited.add(bus)
            component.append(bus)

            stack.extend(
                neighbour
                for neighbour in adjacency[bus]
                if neighbour not in visited
            )

        components.append(sorted(component))

    components.sort(
        key=lambda x: (-len(x), x[0] if x else "")
    )

    return components


def print_topology(network) -> None:

    banner("TOPOLOGY CONFIRMATION")

    components = get_ac_connected_components(network)

    print(
        f"Total AC connected components : {len(components)}"
    )

    for i, component in enumerate(components, start=1):

        print(
            f"Component {i:02d} : {len(component)} buses"
        )

        if len(component) <= 5:
            print(
                f"  Buses: {component}"
            )


# ================================================================================================
# INITIAL VOLTAGE CONDITIONS
# ================================================================================================

def inspect_initial_voltage(network) -> None:

    banner("INITIAL VOLTAGE CONDITIONS")

    buses = get_static(network, "Bus")

    # Static initial voltage conditions.
    #
    # These are NOT buses_t values. buses_t is primarily the dynamic
    # power-flow result table.

    if "v_mag_pu" in buses.columns:
        v_mag = pd.to_numeric(
            buses["v_mag_pu"],
            errors="coerce",
        )
    else:
        v_mag = pd.Series(
            1.0,
            index=buses.index,
            dtype=float,
        )

    if "v_ang" in buses.columns:
        v_ang = pd.to_numeric(
            buses["v_ang"],
            errors="coerce",
        )
    else:
        v_ang = pd.Series(
            0.0,
            index=buses.index,
            dtype=float,
        )

    finite_mag = int(np.isfinite(v_mag).sum())
    finite_ang = int(np.isfinite(v_ang).sum())

    print(
        f"Initial voltage magnitude entries : {len(v_mag)}"
    )
    print(
        f"Finite voltage magnitude entries  : {finite_mag}"
    )
    print(
        f"NaN voltage magnitude entries     : {len(v_mag) - finite_mag}"
    )

    if finite_mag > 0:
        print(
            f"Initial voltage minimum            : "
            f"{v_mag.min():.6f} pu"
        )
        print(
            f"Initial voltage maximum            : "
            f"{v_mag.max():.6f} pu"
        )
    else:
        print(
            "Initial voltage minimum            : NaN pu"
        )
        print(
            "Initial voltage maximum            : NaN pu"
        )

    print()

    print(
        f"Initial angle entries              : {len(v_ang)}"
    )
    print(
        f"Finite angle entries               : {finite_ang}"
    )
    print(
        f"NaN angle entries                  : "
        f"{len(v_ang) - finite_ang}"
    )

    if finite_ang > 0:
        print(
            f"Initial angle minimum              : "
            f"{v_ang.min():.6f} rad"
        )
        print(
            f"Initial angle maximum              : "
            f"{v_ang.max():.6f} rad"
        )
    else:
        print(
            "Initial angle minimum              : NaN rad"
        )
        print(
            "Initial angle maximum              : NaN rad"
        )


# ================================================================================================
# SLACK CONFIGURATION
# ================================================================================================

def configure_baseline_slack(network) -> None:

    banner("CONFIGURING BASELINE SLACK")

    generators = get_static(network, "Generator")

    if "control" not in generators.columns:
        generators["control"] = "PQ"

    print("Generator controls before configuration:")

    print(
        generators["control"].to_string()
    )

    print()

    # Do NOT force a particular generator to be slack.
    #
    # The source operating-point topology already contains the generator
    # configuration used by the previous diagnostic stages.
    #
    # Setting all generators to PQ and allowing PyPSA to establish the
    # slack would destroy the intended source operating point.
    #
    # Therefore we preserve the existing source controls.

    slack_generators = generators.index[
        generators["control"].astype(str).str.lower() == "slack"
    ].tolist()

    if len(slack_generators) == 0:
        explicit_slack = "NONE"
    else:
        explicit_slack = ", ".join(
            str(x) for x in slack_generators
        )

    print(
        f"Explicit slack generator : {explicit_slack}"
    )
    print(
        "Distributed slack        : True"
    )


# ================================================================================================
# POWER FLOW
# ================================================================================================

def run_ac_power_flow(network):

    banner("RUNNING AC NONLINEAR POWER FLOW")

    print("Configuration:")
    print("  Reactive power : ORIGINAL SOURCE Q")
    print("  Explicit slack : NONE")
    print("  Distributed slack : ENABLED")

    print()

    # PyPSA uses the network's current source P/Q time series.
    #
    # No dispatch is changed here.
    # No Q is manufactured here.
    # No reinforcement is applied.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = network.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-6,
            distribute_slack=True,
        )

    print()
    print("Raw power-flow result:")
    print(result)

    return result


# ================================================================================================
# PF RESULT EXTRACTION
# ================================================================================================

def extract_pf_result(result):

    try:
        n_iter_obj = result["n_iter"]
        error_obj = result["error"]
        converged_obj = result["converged"]
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected PyPSA power-flow result structure: {exc}"
        )

    def extract_snapshot_value(obj):

        if isinstance(obj, pd.DataFrame):

            if SNAPSHOT not in obj.index:
                raise RuntimeError(
                    f"Snapshot '{SNAPSHOT}' missing from PF result."
                )

            row = obj.loc[SNAPSHOT]

            if isinstance(row, pd.Series):
                return row.iloc[0]

            return row

        if isinstance(obj, pd.Series):

            if SNAPSHOT in obj.index:
                return obj.loc[SNAPSHOT]

            return obj.iloc[0]

        if np.isscalar(obj):
            return obj

        return np.asarray(obj).reshape(-1)[0]

    n_iter = float(
        extract_snapshot_value(n_iter_obj)
    )

    error = float(
        extract_snapshot_value(error_obj)
    )

    converged_raw = extract_snapshot_value(
        converged_obj
    )

    converged = bool(converged_raw)

    return n_iter, error, converged


# ================================================================================================
# BUS VALIDATION
# ================================================================================================

def validate_bus_results(network):

    banner("BUS VOLTAGE VALIDATION")

    buses = get_static(network, "Bus")

    v_mag = get_solved_component_value(
        network,
        "Bus",
        "v_mag_pu",
        SNAPSHOT,
    )

    v_ang = get_solved_component_value(
        network,
        "Bus",
        "v_ang",
        SNAPSHOT,
    )

    finite_mag_mask = np.isfinite(v_mag.to_numpy(dtype=float))
    finite_ang_mask = np.isfinite(v_ang.to_numpy(dtype=float))

    finite_mag = int(finite_mag_mask.sum())
    finite_ang = int(finite_ang_mask.sum())

    print(
        f"Voltage entries       : {len(buses)}"
    )
    print(
        f"Finite voltage entries: {finite_mag}"
    )
    print(
        f"NaN voltage entries   : {len(buses) - finite_mag}"
    )

    if finite_mag > 0:

        print(
            f"Minimum voltage       : "
            f"{v_mag[finite_mag_mask].min():.6f} pu"
        )

        print(
            f"Maximum voltage       : "
            f"{v_mag[finite_mag_mask].max():.6f} pu"
        )

    else:

        print(
            "Minimum voltage       : nan pu"
        )

        print(
            "Maximum voltage       : nan pu"
        )

    print()

    print(
        f"Angle entries         : {len(buses)}"
    )

    print(
        f"Finite angle entries  : {finite_ang}"
    )

    print(
        f"NaN angle entries     : {len(buses) - finite_ang}"
    )

    if finite_ang > 0:

        print(
            f"Minimum angle         : "
            f"{v_ang[finite_ang_mask].min():.6f} rad"
        )

        print(
            f"Maximum angle         : "
            f"{v_ang[finite_ang_mask].max():.6f} rad"
        )

    else:

        print(
            "Minimum angle         : nan rad"
        )

        print(
            "Maximum angle         : nan rad"
        )

    bus_validation = pd.DataFrame(
        {
            "bus": buses.index,
            "v_mag_pu": v_mag.reindex(buses.index),
            "v_ang_rad": v_ang.reindex(buses.index),
        }
    )

    bus_validation["voltage_finite"] = np.isfinite(
        bus_validation["v_mag_pu"]
    )

    bus_validation["angle_finite"] = np.isfinite(
        bus_validation["v_ang_rad"]
    )

    bus_validation["voltage_range_valid"] = (
        bus_validation["v_mag_pu"].between(
            VOLTAGE_MIN_LIMIT,
            VOLTAGE_MAX_LIMIT,
            inclusive="both",
        )
    )

    bus_validation["angle_range_valid"] = (
        bus_validation["v_ang_rad"].between(
            ANGLE_MIN_LIMIT,
            ANGLE_MAX_LIMIT,
            inclusive="both",
        )
    )

    return bus_validation


# ================================================================================================
# LINE VALIDATION
# ================================================================================================

def validate_line_results(network):

    banner("LINE LOADING VALIDATION")

    lines = get_static(network, "Line")

    p0 = get_solved_component_value(
        network,
        "Line",
        "p0",
        SNAPSHOT,
    )

    p1 = get_solved_component_value(
        network,
        "Line",
        "p1",
        SNAPSHOT,
    )

    q0 = get_solved_component_value(
        network,
        "Line",
        "q0",
        SNAPSHOT,
    )

    q1 = get_solved_component_value(
        network,
        "Line",
        "q1",
        SNAPSHOT,
    )

    s0 = np.sqrt(
        p0.pow(2) +
        q0.pow(2)
    )

    s1 = np.sqrt(
        p1.pow(2) +
        q1.pow(2)
    )

    max_s = pd.concat(
        [s0, s1],
        axis=1,
    ).max(axis=1)

    if "s_nom" in lines.columns:
        s_nom = pd.to_numeric(
            lines["s_nom"],
            errors="coerce",
        )
    else:
        s_nom = pd.Series(
            np.nan,
            index=lines.index,
        )

    loading_pct = (
        max_s
        .div(s_nom.replace(0.0, np.nan))
        * 100.0
    )

    finite_mask = np.isfinite(
        loading_pct.to_numpy(dtype=float)
    )

    finite_count = int(finite_mask.sum())

    print(
        f"Finite line-loading entries : {finite_count}"
    )

    if finite_count > 0:

        print(
            f"Maximum line loading       : "
            f"{loading_pct[finite_mask].max():.6f} %"
        )

        overloaded = loading_pct[
            loading_pct > LINE_LOADING_LIMIT_PCT
        ]

        print(
            f"Overloaded lines           : "
            f"{len(overloaded)}"
        )

        if len(overloaded) > 0:

            overloaded_df = pd.DataFrame(
                {
                    "bus0": lines.loc[
                        overloaded.index,
                        "bus0",
                    ],
                    "bus1": lines.loc[
                        overloaded.index,
                        "bus1",
                    ],
                    "s_nom_mva": s_nom.loc[
                        overloaded.index
                    ],
                    "max_s_mva": max_s.loc[
                        overloaded.index
                    ],
                    "loading_pct": loading_pct.loc[
                        overloaded.index
                    ],
                }
            ).sort_values(
                "loading_pct",
                ascending=False,
            )

            print()
            print("OVERLOADED LINES:")
            print(
                overloaded_df.to_string()
            )

    else:

        print(
            "Maximum line loading       : NaN %"
        )

        print(
            "Overloaded lines           : 0"
        )

    line_validation = pd.DataFrame(
        {
            "line": lines.index,
            "bus0": lines["bus0"],
            "bus1": lines["bus1"],
            "s_nom_mva": s_nom,
            "p0_mw": p0.reindex(lines.index),
            "p1_mw": p1.reindex(lines.index),
            "q0_mvar": q0.reindex(lines.index),
            "q1_mvar": q1.reindex(lines.index),
            "s0_mva": s0.reindex(lines.index),
            "s1_mva": s1.reindex(lines.index),
            "max_s_mva": max_s.reindex(lines.index),
            "loading_pct": loading_pct.reindex(lines.index),
        }
    )

    line_validation["finite_loading"] = np.isfinite(
        line_validation["loading_pct"]
    )

    line_validation["overloaded"] = (
        line_validation["loading_pct"]
        > LINE_LOADING_LIMIT_PCT
    )

    return line_validation


# ================================================================================================
# TRANSFORMER VALIDATION
# ================================================================================================

def validate_transformer_results(network):

    banner("TRANSFORMER LOADING VALIDATION")

    transformers = get_static(
        network,
        "Transformer",
    )

    if len(transformers) == 0:

        print(
            "No transformers present."
        )

        return pd.DataFrame(
            columns=[
                "transformer",
                "bus0",
                "bus1",
                "s_nom_mva",
                "p0_mw",
                "p1_mw",
                "q0_mvar",
                "q1_mvar",
                "max_s_mva",
                "loading_pct",
            ]
        )

    p0 = get_solved_component_value(
        network,
        "Transformer",
        "p0",
        SNAPSHOT,
    )

    p1 = get_solved_component_value(
        network,
        "Transformer",
        "p1",
        SNAPSHOT,
    )

    q0 = get_solved_component_value(
        network,
        "Transformer",
        "q0",
        SNAPSHOT,
    )

    q1 = get_solved_component_value(
        network,
        "Transformer",
        "q1",
        SNAPSHOT,
    )

    s0 = np.sqrt(
        p0.pow(2) +
        q0.pow(2)
    )

    s1 = np.sqrt(
        p1.pow(2) +
        q1.pow(2)
    )

    max_s = pd.concat(
        [s0, s1],
        axis=1,
    ).max(axis=1)

    if "s_nom" in transformers.columns:

        s_nom = pd.to_numeric(
            transformers["s_nom"],
            errors="coerce",
        )

    else:

        s_nom = pd.Series(
            np.nan,
            index=transformers.index,
        )

    loading_pct = (
        max_s
        .div(s_nom.replace(0.0, np.nan))
        * 100.0
    )

    finite_mask = np.isfinite(
        loading_pct.to_numpy(dtype=float)
    )

    finite_count = int(
        finite_mask.sum()
    )

    print(
        f"Finite transformer-loading entries : "
        f"{finite_count}"
    )

    if finite_count > 0:

        print(
            f"Maximum transformer loading       : "
            f"{loading_pct[finite_mask].max():.6f} %"
        )

        overloaded = loading_pct[
            loading_pct > TRANSFORMER_LOADING_LIMIT_PCT
        ]

        print(
            f"Overloaded transformers           : "
            f"{len(overloaded)}"
        )

    else:

        print(
            "Maximum transformer loading       : NaN %"
        )

        print(
            "Overloaded transformers           : 0"
        )

    transformer_validation = pd.DataFrame(
        {
            "transformer": transformers.index,
            "bus0": transformers["bus0"],
            "bus1": transformers["bus1"],
            "s_nom_mva": s_nom,
            "p0_mw": p0.reindex(transformers.index),
            "p1_mw": p1.reindex(transformers.index),
            "q0_mvar": q0.reindex(transformers.index),
            "q1_mvar": q1.reindex(transformers.index),
            "s0_mva": s0.reindex(transformers.index),
            "s1_mva": s1.reindex(transformers.index),
            "max_s_mva": max_s.reindex(transformers.index),
            "loading_pct": loading_pct.reindex(
                transformers.index
            ),
        }
    )

    transformer_validation["finite_loading"] = np.isfinite(
        transformer_validation["loading_pct"]
    )

    transformer_validation["overloaded"] = (
        transformer_validation["loading_pct"]
        > TRANSFORMER_LOADING_LIMIT_PCT
    )

    return transformer_validation


# ================================================================================================
# GENERATOR VALIDATION
# ================================================================================================

def validate_generator_results(network):

    banner("GENERATOR SOLUTION VALIDATION")

    generators = get_static(
        network,
        "Generator",
    )

    p_set = get_active_component_value(
        network,
        "Generator",
        "p",
        SNAPSHOT,
    )

    q_set = get_active_component_value(
        network,
        "Generator",
        "q",
        SNAPSHOT,
    )

    p_solved = get_solved_component_value(
        network,
        "Generator",
        "p",
        SNAPSHOT,
    )

    q_solved = get_solved_component_value(
        network,
        "Generator",
        "q",
        SNAPSHOT,
    )

    if "bus" in generators.columns:
        bus = generators["bus"]
    else:
        bus = pd.Series(
            np.nan,
            index=generators.index,
        )

    if "control" in generators.columns:
        control = generators["control"]
    else:
        control = pd.Series(
            "PQ",
            index=generators.index,
        )

    generator_validation = pd.DataFrame(
        {
            "generator": generators.index,
            "bus": bus,
            "control": control,
            "p_set_mw": p_set.reindex(
                generators.index
            ),
            "p_solved_mw": p_solved.reindex(
                generators.index
            ),
            "q_set_mvar": q_set.reindex(
                generators.index
            ),
            "q_solved_mvar": q_solved.reindex(
                generators.index
            ),
        }
    )

    generator_validation = (
        generator_validation
        .set_index("generator")
    )

    print(
        generator_validation.to_string()
    )

    return generator_validation


# ================================================================================================
# LOAD VALIDATION
# ================================================================================================

def validate_load_results(network):

    banner("LOAD SOLUTION VALIDATION")

    loads = get_static(
        network,
        "Load",
    )

    p_set = get_active_component_value(
        network,
        "Load",
        "p",
        SNAPSHOT,
    )

    q_set = get_active_component_value(
        network,
        "Load",
        "q",
        SNAPSHOT,
    )

    p_solved = get_solved_component_value(
        network,
        "Load",
        "p",
        SNAPSHOT,
    )

    q_solved = get_solved_component_value(
        network,
        "Load",
        "q",
        SNAPSHOT,
    )

    if "bus" in loads.columns:
        bus = loads["bus"]
    else:
        bus = pd.Series(
            np.nan,
            index=loads.index,
        )

    load_validation = pd.DataFrame(
        {
            "load": loads.index,
            "bus": bus,
            "p_set_mw": p_set.reindex(
                loads.index
            ),
            "p_solved_mw": p_solved.reindex(
                loads.index
            ),
            "q_set_mvar": q_set.reindex(
                loads.index
            ),
            "q_solved_mvar": q_solved.reindex(
                loads.index
            ),
        }
    )

    load_validation = (
        load_validation
        .set_index("load")
    )

    print(
        load_validation.to_string()
    )

    return load_validation


# ================================================================================================
# POST-PF BALANCE
# ================================================================================================

def calculate_post_pf_balance(network):

    banner("POST-PF POWER BALANCE")

    generator_p = get_solved_component_value(
        network,
        "Generator",
        "p",
        SNAPSHOT,
    )

    load_p = get_solved_component_value(
        network,
        "Load",
        "p",
        SNAPSHOT,
    )

    solved_generation = float(
        generator_p.fillna(0.0).sum()
    )

    solved_load = float(
        load_p.fillna(0.0).sum()
    )

    balance = (
        solved_generation
        - solved_load
    )

    print(
        f"Solved generation : "
        f"{solved_generation:.6f} MW"
    )

    print(
        f"Solved load       : "
        f"{solved_load:.6f} MW"
    )

    print(
        f"Generation-load   : "
        f"{balance:.6f} MW"
    )

    return (
        solved_generation,
        solved_load,
        balance,
    )


# ================================================================================================
# FINAL VALIDITY
# ================================================================================================

def assess_physical_validity(
    pf_converged,
    pf_error,
    bus_validation,
    line_validation,
    transformer_validation,
):

    banner("FINAL PHYSICAL VALIDITY ASSESSMENT")

    finite_voltage = bool(
        bus_validation["voltage_finite"].all()
    )

    finite_angle = bool(
        bus_validation["angle_finite"].all()
    )

    voltage_range_valid = bool(
        bus_validation["voltage_range_valid"].all()
    )

    angle_range_valid = bool(
        bus_validation["angle_range_valid"].all()
    )

    if len(line_validation) > 0:

        line_loading_valid = bool(
            line_validation["finite_loading"].all()
        )

    else:

        line_loading_valid = True

    if len(transformer_validation) > 0:

        transformer_loading_valid = bool(
            transformer_validation["finite_loading"].all()
        )

    else:

        transformer_loading_valid = True

    valid_physical_solution = bool(
        pf_converged
        and np.isfinite(pf_error)
        and pf_error <= PF_ERROR_LIMIT
        and finite_voltage
        and voltage_range_valid
        and finite_angle
        and angle_range_valid
        and line_loading_valid
        and transformer_loading_valid
    )

    print(
        f"PF converged               : {pf_converged}"
    )

    print(
        f"PF error                   : "
        f"{pf_error:.6e}"
    )

    print(
        f"Voltage finite             : "
        f"{finite_voltage}"
    )

    print(
        f"Voltage range valid        : "
        f"{voltage_range_valid}"
    )

    if finite_voltage:

        print(
            f"Voltage minimum            : "
            f"{bus_validation['v_mag_pu'].min():.6f} pu"
        )

        print(
            f"Voltage maximum            : "
            f"{bus_validation['v_mag_pu'].max():.6f} pu"
        )

    print(
        f"Angle finite               : "
        f"{finite_angle}"
    )

    print(
        f"Angle range valid          : "
        f"{angle_range_valid}"
    )

    if finite_angle:

        print(
            f"Angle minimum              : "
            f"{bus_validation['v_ang_rad'].min():.6f} rad"
        )

        print(
            f"Angle maximum              : "
            f"{bus_validation['v_ang_rad'].max():.6f} rad"
        )

    print(
        f"Line loading data valid    : "
        f"{line_loading_valid}"
    )

    print(
        f"Transformer loading valid  : "
        f"{transformer_loading_valid}"
    )

    print()

    print(
        f"VALID PHYSICAL SOLUTION    : "
        f"{valid_physical_solution}"
    )

    return {
        "finite_voltage": finite_voltage,
        "voltage_range_valid": voltage_range_valid,
        "finite_angle": finite_angle,
        "angle_range_valid": angle_range_valid,
        "line_loading_valid": line_loading_valid,
        "transformer_loading_valid": transformer_loading_valid,
        "valid_physical_solution": valid_physical_solution,
    }


# ================================================================================================
# SUMMARY
# ================================================================================================

def create_summary(
    network,
    pf_converged,
    pf_error,
    iterations,
    bus_validation,
    line_validation,
    transformer_validation,
    solved_generation,
    solved_load,
    generation_minus_load,
    validity,
):

    if len(line_validation) > 0:

        finite_line = line_validation[
            "finite_loading"
        ]

        if finite_line.any():

            max_line_loading = float(
                line_validation.loc[
                    finite_line,
                    "loading_pct",
                ].max()
            )

            overloaded_lines = int(
                line_validation["overloaded"].sum()
            )

        else:

            max_line_loading = np.nan
            overloaded_lines = 0

    else:

        max_line_loading = np.nan
        overloaded_lines = 0

    if len(transformer_validation) > 0:

        finite_transformer = transformer_validation[
            "finite_loading"
        ]

        if finite_transformer.any():

            max_transformer_loading = float(
                transformer_validation.loc[
                    finite_transformer,
                    "loading_pct",
                ].max()
            )

            overloaded_transformers = int(
                transformer_validation[
                    "overloaded"
                ].sum()
            )

        else:

            max_transformer_loading = np.nan
            overloaded_transformers = 0

    else:

        max_transformer_loading = np.nan
        overloaded_transformers = 0

    finite_voltage = (
        bus_validation["v_mag_pu"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    finite_angle = (
        bus_validation["v_ang_rad"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(finite_voltage) > 0:

        voltage_min = float(
            finite_voltage.min()
        )

        voltage_max = float(
            finite_voltage.max()
        )

    else:

        voltage_min = np.nan
        voltage_max = np.nan

    if len(finite_angle) > 0:

        angle_min = float(
            finite_angle.min()
        )

        angle_max = float(
            finite_angle.max()
        )

    else:

        angle_min = np.nan
        angle_max = np.nan

    summary = pd.DataFrame(
        [
            {
                "stage": "S4.6",
                "network": str(NETWORK_PATH),
                "snapshot": SNAPSHOT,
                "pf": "AC nonlinear",
                "reactive_mode": "ORIGINAL_SOURCE_Q",
                "slack_mode": "DISTRIBUTED",
                "explicit_slack": False,
                "converged": pf_converged,
                "pf_error": pf_error,
                "iterations": iterations,
                "voltage_min_pu": voltage_min,
                "voltage_max_pu": voltage_max,
                "angle_min_rad": angle_min,
                "angle_max_rad": angle_max,
                "max_line_loading_pct": max_line_loading,
                "overloaded_lines": overloaded_lines,
                "max_transformer_loading_pct":
                    max_transformer_loading,
                "overloaded_transformers":
                    overloaded_transformers,
                "solved_generation_mw":
                    solved_generation,
                "solved_load_mw":
                    solved_load,
                "generation_minus_load_mw":
                    generation_minus_load,
                "valid_physical_solution":
                    validity["valid_physical_solution"],
            }
        ]
    )

    return summary


# ================================================================================================
# SAVE RESULTS
# ================================================================================================

def save_results(
    summary,
    bus_validation,
    line_validation,
    transformer_validation,
    generator_validation,
    load_validation,
):

    banner("SAVING BASELINE RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    bus_validation.to_csv(
        BUS_PATH,
        index=False,
    )

    line_validation.to_csv(
        LINE_PATH,
        index=False,
    )

    transformer_validation.to_csv(
        TRANSFORMER_PATH,
        index=False,
    )

    generator_validation.to_csv(
        GENERATOR_PATH,
    )

    load_validation.to_csv(
        LOAD_PATH,
    )

    print(
        f"Summary       : {SUMMARY_PATH}"
    )

    print(
        f"Bus validation: {BUS_PATH}"
    )

    print(
        f"Line          : {LINE_PATH}"
    )

    print(
        f"Transformer   : {TRANSFORMER_PATH}"
    )

    print(
        f"Generator     : {GENERATOR_PATH}"
    )

    print(
        f"Load          : {LOAD_PATH}"
    )


# ================================================================================================
# MAIN
# ================================================================================================

def main():

    banner(
        "S4.6 — BASELINE AC POWER-FLOW ESTABLISHMENT"
    )

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
        "Reactive : ORIGINAL SOURCE Q"
    )

    print(
        "Slack    : DISTRIBUTED"
    )

    print(
        "Source   : READ-ONLY"
    )

    print()

    print(
        "No reinforcement is applied."
    )

    print(
        "No reactive compensation is added."
    )

    print(
        "No source network file is modified."
    )

    # --------------------------------------------------------------------------------------------
    # LOAD SOURCE NETWORK
    # --------------------------------------------------------------------------------------------

    banner("LOADING SOURCE NETWORK")

    if not NETWORK_PATH.exists():

        raise FileNotFoundError(
            f"Network file not found:\n{NETWORK_PATH}"
        )

    # Import once.
    source = pypsa.Network(
        str(NETWORK_PATH)
    )

    print_network_information(
        source
    )

    # --------------------------------------------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------------------------------------------

    banner("SNAPSHOT ISOLATION")

    validate_snapshot(
        source
    )

    # --------------------------------------------------------------------------------------------
    # ORIGINAL OPERATING POINT
    # --------------------------------------------------------------------------------------------

    print_operating_point(
        source
    )

    # --------------------------------------------------------------------------------------------
    # REACTIVE POWER
    # --------------------------------------------------------------------------------------------

    validate_reactive_power(
        source
    )

    # --------------------------------------------------------------------------------------------
    # TOPOLOGY
    # --------------------------------------------------------------------------------------------

    print_topology(
        source
    )

    # --------------------------------------------------------------------------------------------
    # INITIAL VOLTAGE
    # --------------------------------------------------------------------------------------------

    inspect_initial_voltage(
        source
    )

    # --------------------------------------------------------------------------------------------
    # SLACK
    # --------------------------------------------------------------------------------------------

    configure_baseline_slack(
        source
    )

    # --------------------------------------------------------------------------------------------
    # POWER FLOW
    # --------------------------------------------------------------------------------------------

    result = run_ac_power_flow(
        source
    )

    (
        iterations,
        pf_error,
        pf_converged,
    ) = extract_pf_result(
        result
    )

    # --------------------------------------------------------------------------------------------
    # BUS
    # --------------------------------------------------------------------------------------------

    bus_validation = validate_bus_results(
        source
    )

    # --------------------------------------------------------------------------------------------
    # LINE
    # --------------------------------------------------------------------------------------------

    line_validation = validate_line_results(
        source
    )

    # --------------------------------------------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------------------------------------------

    transformer_validation = (
        validate_transformer_results(
            source
        )
    )

    # --------------------------------------------------------------------------------------------
    # GENERATORS
    # --------------------------------------------------------------------------------------------

    generator_validation = (
        validate_generator_results(
            source
        )
    )

    # --------------------------------------------------------------------------------------------
    # LOADS
    # --------------------------------------------------------------------------------------------

    load_validation = (
        validate_load_results(
            source
        )
    )

    # --------------------------------------------------------------------------------------------
    # POST-PF BALANCE
    # --------------------------------------------------------------------------------------------

    (
        solved_generation,
        solved_load,
        generation_minus_load,
    ) = calculate_post_pf_balance(
        source
    )

    # --------------------------------------------------------------------------------------------
    # FINAL VALIDITY
    # --------------------------------------------------------------------------------------------

    validity = assess_physical_validity(
        pf_converged=pf_converged,
        pf_error=pf_error,
        bus_validation=bus_validation,
        line_validation=line_validation,
        transformer_validation=transformer_validation,
    )

    # --------------------------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------------------------

    summary = create_summary(
        network=source,
        pf_converged=pf_converged,
        pf_error=pf_error,
        iterations=iterations,
        bus_validation=bus_validation,
        line_validation=line_validation,
        transformer_validation=transformer_validation,
        solved_generation=solved_generation,
        solved_load=solved_load,
        generation_minus_load=generation_minus_load,
        validity=validity,
    )

    banner("S4.6 — BASELINE SUMMARY")

    print(
        summary.to_string(index=False)
    )

    # --------------------------------------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------------------------------------

    save_results(
        summary=summary,
        bus_validation=bus_validation,
        line_validation=line_validation,
        transformer_validation=transformer_validation,
        generator_validation=generator_validation,
        load_validation=load_validation,
    )

    # --------------------------------------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------------------------------------

    banner("S4.6 COMPLETE")

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

    if validity["valid_physical_solution"]:

        print(
            "BASELINE AC POWER FLOW : VALID"
        )

    else:

        print(
            "BASELINE AC POWER FLOW : INVALID"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This stage is diagnostic only."
    )

    print(
        "No reinforcement has been applied."
    )

    print(
        "No source network file has been modified."
    )


# ================================================================================================
# ENTRY POINT
# ================================================================================================

if __name__ == "__main__":
    main()