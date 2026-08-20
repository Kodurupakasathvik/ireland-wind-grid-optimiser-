"""
====================================================================================================
S4.7 — BASELINE BOTTLENECK & CRITICAL-BUS IDENTIFICATION
====================================================================================================

Purpose
-------
Identify the critical voltage and thermal bottlenecks of the established
S4.6 baseline AC power-flow solution.

Network  : data\processed\eirgrid_second_reinforced_network.nc
Snapshot : S2_PEAK_DEMAND
PF       : AC nonlinear
Reactive : ORIGINAL SOURCE Q
Slack    : DISTRIBUTED
Source   : READ-ONLY

NO reinforcement is applied.
NO reactive compensation is added.
NO dispatch change is applied.
NO load change is applied.
NO source network file is modified.

Outputs
-------
data\processed\s4_7_bottleneck_summary.csv
data\processed\s4_7_bus_criticality.csv
data\processed\s4_7_line_criticality.csv
data\processed\s4_7_transformer_criticality.csv
data\processed\s4_7_generator_criticality.csv
data\processed\s4_7_load_criticality.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


# ================================================================================================
# CONFIGURATION
# ================================================================================================

NETWORK_PATH = Path(
    r"data\processed\eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_DIR = Path("data") / "processed"

SUMMARY_PATH = OUTPUT_DIR / "s4_7_bottleneck_summary.csv"
BUS_PATH = OUTPUT_DIR / "s4_7_bus_criticality.csv"
LINE_PATH = OUTPUT_DIR / "s4_7_line_criticality.csv"
TRANSFORMER_PATH = OUTPUT_DIR / "s4_7_transformer_criticality.csv"
GENERATOR_PATH = OUTPUT_DIR / "s4_7_generator_criticality.csv"
LOAD_PATH = OUTPUT_DIR / "s4_7_load_criticality.csv"

# Physical screening thresholds.
# These are diagnostic thresholds only.
VOLTAGE_LOW_PU = 0.95
VOLTAGE_HIGH_PU = 1.05
LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0

# Number of most critical elements shown prominently in console output.
TOP_N = 15

# PyPSA logging can be verbose.
warnings.filterwarnings("ignore", category=FutureWarning)


# ================================================================================================
# FORMATTING
# ================================================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def safe_float(value) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ================================================================================================
# PYPSA-COMPATIBLE COMPONENT ACCESS
# ================================================================================================

def static_df(network: pypsa.Network, component: str) -> pd.DataFrame:
    """
    Current PyPSA-compatible access to static component data.

    Avoids deprecated network.df(component).
    """
    try:
        return network.components[component].static
    except Exception:
        # Compatibility fallback for unusual PyPSA builds.
        attr = component.lower() + "s"
        obj = getattr(network, attr)
        return obj


def pnl_df(network: pypsa.Network, component: str, attribute: str) -> pd.DataFrame:
    """
    Return a component time-series dataframe.

    Current PyPSA versions expose pnl through the component object.
    """
    try:
        return network.components[component].dynamic[attribute]
    except Exception:
        pass

    try:
        return network.components[component].pnl[attribute]
    except Exception:
        pass

    # Last-resort compatibility for older PyPSA structures.
    try:
        return getattr(network, component.lower()).t[attr]
    except Exception:
        return pd.DataFrame(index=network.snapshots)


def get_series(
    network: pypsa.Network,
    component: str,
    attribute: str,
    snapshot: str,
) -> pd.Series:
    """
    Safely obtain a component time-series attribute at the requested snapshot.

    If the dynamic attribute does not exist, return NaNs indexed by the
    component static dataframe.
    """
    df = static_df(network, component)
    index = df.index

    try:
        pnl = pnl_df(network, component, attribute)

        if pnl is None or len(pnl.columns) == 0:
            return pd.Series(np.nan, index=index, dtype=float)

        if snapshot not in pnl.index:
            return pd.Series(np.nan, index=index, dtype=float)

        row = pnl.loc[snapshot]

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        row = pd.to_numeric(row, errors="coerce")
        return row.reindex(index)

    except Exception:
        return pd.Series(np.nan, index=index, dtype=float)


def get_static_or_dynamic(
    network: pypsa.Network,
    component: str,
    attribute: str,
    snapshot: str,
) -> pd.Series:
    """
    Prefer the actual snapshot value.

    If no dynamic value exists, use the static value.
    """
    df = static_df(network, component)

    dynamic = get_series(network, component, attribute, snapshot)

    if dynamic.notna().any():
        result = dynamic.copy()

        static_values = pd.to_numeric(
            df[attribute], errors="coerce"
        ) if attribute in df.columns else pd.Series(
            np.nan, index=df.index
        )

        result = result.fillna(static_values)
        return result

    if attribute in df.columns:
        return pd.to_numeric(df[attribute], errors="coerce")

    return pd.Series(np.nan, index=df.index, dtype=float)


# ================================================================================================
# NETWORK INFORMATION
# ================================================================================================

def print_network_information(network: pypsa.Network) -> None:
    print_header("LOADING SOURCE NETWORK")

    print(f"Buses        : {len(static_df(network, 'Bus'))}")
    print(f"Lines        : {len(static_df(network, 'Line'))}")
    print(f"Transformers : {len(static_df(network, 'Transformer'))}")
    print(f"Generators   : {len(static_df(network, 'Generator'))}")
    print(f"Loads        : {len(static_df(network, 'Load'))}")


def isolate_snapshot(network: pypsa.Network) -> None:
    print_header("SNAPSHOT ISOLATION")

    if SNAPSHOT not in network.snapshots:
        raise RuntimeError(
            f"Required snapshot '{SNAPSHOT}' not found in network."
        )

    network.set_snapshots([SNAPSHOT])

    print(f"Active snapshot:")
    print(f"  {SNAPSHOT}")


# ================================================================================================
# OPERATING POINT
# ================================================================================================

def get_operating_point(network: pypsa.Network):
    generators = static_df(network, "Generator")
    loads = static_df(network, "Load")

    gen_p = get_static_or_dynamic(
        network, "Generator", "p_set", SNAPSHOT
    )

    load_p = get_static_or_dynamic(
        network, "Load", "p_set", SNAPSHOT
    )

    gen_q = get_static_or_dynamic(
        network, "Generator", "q_set", SNAPSHOT
    )

    load_q = get_static_or_dynamic(
        network, "Load", "q_set", SNAPSHOT
    )

    return gen_p, load_p, gen_q, load_q


def print_operating_point(network: pypsa.Network) -> None:
    print_header("ORIGINAL OPERATING POINT")

    gen_p, load_p, gen_q, load_q = get_operating_point(network)

    gen_p_sum = float(gen_p.fillna(0).sum())
    load_p_sum = float(load_p.fillna(0).sum())

    gen_q_sum = float(gen_q.fillna(0).sum())
    load_q_sum = float(load_q.fillna(0).sum())

    print(f"Generator P set : {gen_p_sum:.6f} MW")
    print(f"Load P set      : {load_p_sum:.6f} MW")
    print(f"Generation-load : {gen_p_sum - load_p_sum:.6f} MW")

    print()

    print(f"Generator Q set : {gen_q_sum:.6f} Mvar")
    print(f"Load Q set      : {load_q_sum:.6f} Mvar")


# ================================================================================================
# TOPOLOGY
# ================================================================================================

def confirm_topology(network: pypsa.Network) -> None:
    print_header("TOPOLOGY CONFIRMATION")

    buses = static_df(network, "Bus")

    # AC network is based on Lines + Transformers.
    adjacency = {bus: set() for bus in buses.index}

    lines = static_df(network, "Line")

    for _, row in lines.iterrows():
        bus0 = row.get("bus0")
        bus1 = row.get("bus1")

        if bus0 in adjacency and bus1 in adjacency:
            adjacency[bus0].add(bus1)
            adjacency[bus1].add(bus0)

    transformers = static_df(network, "Transformer")

    for _, row in transformers.iterrows():
        bus0 = row.get("bus0")
        bus1 = row.get("bus1")

        if bus0 in adjacency and bus1 in adjacency:
            adjacency[bus0].add(bus1)
            adjacency[bus1].add(bus0)

    visited = set()
    components = []

    for bus in buses.index:
        if bus in visited:
            continue

        stack = [bus]
        component = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    stack.append(neighbour)

        components.append(sorted(component))

    components.sort(key=len, reverse=True)

    print(f"Total AC connected components : {len(components)}")

    for i, component in enumerate(components, start=1):
        print(f"Component {i:02d} : {len(component)} buses")

        if len(component) <= 5:
            print(f"  Buses: {component}")


# ================================================================================================
# POWER FLOW
# ================================================================================================

def configure_distributed_slack(network: pypsa.Network) -> None:
    print_header("CONFIGURING BASELINE SLACK")

    generators = static_df(network, "Generator")

    print("Generator controls before configuration:")

    if "control" in generators.columns:
        print(generators["control"].to_string())
    else:
        print("control column unavailable")

    # Preserve the S4.6 methodology:
    # no explicit slack generator; use distributed slack.
    if "control" in generators.columns:
        generators.loc[:, "control"] = "PQ"

    print()
    print("Explicit slack generator : NONE")
    print("Distributed slack        : True")


def run_power_flow(network: pypsa.Network):
    print_header("RUNNING AC NONLINEAR POWER FLOW")

    print("Configuration:")
    print("  Reactive power : ORIGINAL SOURCE Q")
    print("  Explicit slack : NONE")
    print("  Distributed slack : ENABLED")
    print()

    result = network.pf(
        snapshots=[SNAPSHOT],
        distribute_slack=True,
    )

    print("Raw power-flow result:")
    print(result)

    return result


# ================================================================================================
# VOLTAGE EXTRACTION
# ================================================================================================

def get_bus_voltage(network: pypsa.Network) -> pd.DataFrame:
    buses = static_df(network, "Bus")

    v_mag = get_series(
        network, "Bus", "v_mag_pu", SNAPSHOT
    )

    v_ang = get_series(
        network, "Bus", "v_ang", SNAPSHOT
    )

    result = pd.DataFrame(index=buses.index)

    result["v_mag_pu"] = pd.to_numeric(v_mag, errors="coerce")
    result["v_ang_rad"] = pd.to_numeric(v_ang, errors="coerce")

    result["finite_voltage"] = np.isfinite(
        result["v_mag_pu"]
    )

    result["finite_angle"] = np.isfinite(
        result["v_ang_rad"]
    )

    result["voltage_low"] = (
        result["v_mag_pu"] < VOLTAGE_LOW_PU
    )

    result["voltage_high"] = (
        result["v_mag_pu"] > VOLTAGE_HIGH_PU
    )

    result["voltage_violation"] = (
        result["voltage_low"] |
        result["voltage_high"]
    )

    result["voltage_deviation_pu"] = np.abs(
        result["v_mag_pu"] - 1.0
    )

    return result


# ================================================================================================
# LINE LOADING
# ================================================================================================

def get_line_loading(network: pypsa.Network) -> pd.DataFrame:
    lines = static_df(network, "Line")

    result = lines.copy()

    # PyPSA AC power-flow outputs.
    s0 = get_series(network, "Line", "s0", SNAPSHOT)
    s1 = get_series(network, "Line", "s1", SNAPSHOT)

    s0_abs = pd.to_numeric(s0, errors="coerce").abs()
    s1_abs = pd.to_numeric(s1, errors="coerce").abs()

    # If s0/s1 are unavailable, calculate apparent power from p/q.
    if not s0_abs.notna().any():
        p0 = get_series(network, "Line", "p0", SNAPSHOT)
        q0 = get_series(network, "Line", "q0", SNAPSHOT)

        s0_abs = np.sqrt(
            pd.to_numeric(p0, errors="coerce") ** 2 +
            pd.to_numeric(q0, errors="coerce") ** 2
        )

    if not s1_abs.notna().any():
        p1 = get_series(network, "Line", "p1", SNAPSHOT)
        q1 = get_series(network, "Line", "q1", SNAPSHOT)

        s1_abs = np.sqrt(
            pd.to_numeric(p1, errors="coerce") ** 2 +
            pd.to_numeric(q1, errors="coerce") ** 2
        )

    s_max = pd.concat([s0_abs, s1_abs], axis=1).max(axis=1)

    if "s_nom" in lines.columns:
        s_nom = pd.to_numeric(
            lines["s_nom"], errors="coerce"
        )
    else:
        s_nom = pd.Series(np.nan, index=lines.index)

    result = pd.DataFrame(index=lines.index)

    result["bus0"] = lines["bus0"]
    result["bus1"] = lines["bus1"]
    result["s_nom_mva"] = s_nom
    result["s_max_mva"] = s_max

    result["loading_pct"] = (
        100.0 * result["s_max_mva"] /
        result["s_nom_mva"].replace(0, np.nan)
    )

    result["finite_loading"] = np.isfinite(
        result["loading_pct"]
    )

    result["overloaded"] = (
        result["loading_pct"] > LINE_LOADING_LIMIT_PCT
    )

    result["overload_pct"] = (
        result["loading_pct"] -
        LINE_LOADING_LIMIT_PCT
    ).clip(lower=0)

    return result


# ================================================================================================
# TRANSFORMER LOADING
# ================================================================================================

def get_transformer_loading(network: pypsa.Network) -> pd.DataFrame:
    transformers = static_df(network, "Transformer")

    result = pd.DataFrame(index=transformers.index)

    result["bus0"] = transformers["bus0"]
    result["bus1"] = transformers["bus1"]

    s0 = get_series(
        network, "Transformer", "s0", SNAPSHOT
    )

    s1 = get_series(
        network, "Transformer", "s1", SNAPSHOT
    )

    s0_abs = pd.to_numeric(s0, errors="coerce").abs()
    s1_abs = pd.to_numeric(s1, errors="coerce").abs()

    if not s0_abs.notna().any():
        p0 = get_series(
            network, "Transformer", "p0", SNAPSHOT
        )
        q0 = get_series(
            network, "Transformer", "q0", SNAPSHOT
        )

        s0_abs = np.sqrt(
            pd.to_numeric(p0, errors="coerce") ** 2 +
            pd.to_numeric(q0, errors="coerce") ** 2
        )

    if not s1_abs.notna().any():
        p1 = get_series(
            network, "Transformer", "p1", SNAPSHOT
        )
        q1 = get_series(
            network, "Transformer", "q1", SNAPSHOT
        )

        s1_abs = np.sqrt(
            pd.to_numeric(p1, errors="coerce") ** 2 +
            pd.to_numeric(q1, errors="coerce") ** 2
        )

    s_max = pd.concat([s0_abs, s1_abs], axis=1).max(axis=1)

    if "s_nom" in transformers.columns:
        s_nom = pd.to_numeric(
            transformers["s_nom"], errors="coerce"
        )
    else:
        s_nom = pd.Series(np.nan, index=transformers.index)

    result["s_nom_mva"] = s_nom
    result["s_max_mva"] = s_max

    result["loading_pct"] = (
        100.0 * result["s_max_mva"] /
        result["s_nom_mva"].replace(0, np.nan)
    )

    result["finite_loading"] = np.isfinite(
        result["loading_pct"]
    )

    result["overloaded"] = (
        result["loading_pct"] >
        TRANSFORMER_LOADING_LIMIT_PCT
    )

    result["overload_pct"] = (
        result["loading_pct"] -
        TRANSFORMER_LOADING_LIMIT_PCT
    ).clip(lower=0)

    return result


# ================================================================================================
# BUS CRITICALITY
# ================================================================================================

def build_bus_criticality(
    network: pypsa.Network,
    voltage: pd.DataFrame,
    lines: pd.DataFrame,
) -> pd.DataFrame:

    buses = static_df(network, "Bus")

    result = voltage.copy()

    result["bus_v_nom_kv"] = pd.to_numeric(
        buses.get(
            "v_nom",
            pd.Series(np.nan, index=buses.index)
        ),
        errors="coerce",
    )

    # Number of incident lines.
    incident_line_count = pd.Series(
        0, index=buses.index, dtype=int
    )

    overloaded_incident_count = pd.Series(
        0, index=buses.index, dtype=int
    )

    max_incident_loading = pd.Series(
        np.nan, index=buses.index, dtype=float
    )

    for line_name, row in lines.iterrows():

        bus0 = row["bus0"]
        bus1 = row["bus1"]
        loading = safe_float(row["loading_pct"])
        overloaded = bool(row["overloaded"])

        for bus in [bus0, bus1]:

            if bus not in incident_line_count.index:
                continue

            incident_line_count.loc[bus] += 1

            if overloaded:
                overloaded_incident_count.loc[bus] += 1

            old = max_incident_loading.loc[bus]

            if np.isnan(old) or (
                np.isfinite(loading) and loading > old
            ):
                max_incident_loading.loc[bus] = loading

    result["incident_lines"] = incident_line_count
    result["incident_overloaded_lines"] = overloaded_incident_count
    result["max_incident_line_loading_pct"] = max_incident_loading

    # Connected generators.
    generators = static_df(network, "Generator")

    gen_count = pd.Series(
        0, index=buses.index, dtype=int
    )

    gen_p = get_static_or_dynamic(
        network, "Generator", "p_set", SNAPSHOT
    )

    for generator, row in generators.iterrows():

        bus = row.get("bus")

        if bus in gen_count.index:
            gen_count.loc[bus] += 1

    result["generator_count"] = gen_count

    # Connected load.
    loads = static_df(network, "Load")

    load_count = pd.Series(
        0, index=buses.index, dtype=int
    )

    load_p = get_static_or_dynamic(
        network, "Load", "p_set", SNAPSHOT
    )

    for load, row in loads.iterrows():

        bus = row.get("bus")

        if bus in load_count.index:
            load_count.loc[bus] += 1

    result["load_count"] = load_count

    # Aggregate active power by bus.
    generation_by_bus = pd.Series(
        0.0, index=buses.index
    )

    for generator, row in generators.iterrows():

        bus = row.get("bus")

        if bus in generation_by_bus.index:
            value = safe_float(gen_p.get(generator, np.nan))

            if np.isfinite(value):
                generation_by_bus.loc[bus] += value

    load_by_bus = pd.Series(
        0.0, index=buses.index
    )

    for load, row in loads.iterrows():

        bus = row.get("bus")

        if bus in load_by_bus.index:
            value = safe_float(load_p.get(load, np.nan))

            if np.isfinite(value):
                load_by_bus.loc[bus] += value

    result["generation_p_mw"] = generation_by_bus
    result["load_p_mw"] = load_by_bus
    result["net_injection_p_mw"] = (
        generation_by_bus - load_by_bus
    )

    # Severity scores.
    voltage_score = pd.Series(
        0.0, index=result.index
    )

    low_mask = result["v_mag_pu"] < VOLTAGE_LOW_PU

    high_mask = result["v_mag_pu"] > VOLTAGE_HIGH_PU

    voltage_score.loc[low_mask] = (
        (VOLTAGE_LOW_PU - result.loc[low_mask, "v_mag_pu"])
        / VOLTAGE_LOW_PU
        * 100.0
    )

    voltage_score.loc[high_mask] = (
        (result.loc[high_mask, "v_mag_pu"] - VOLTAGE_HIGH_PU)
        / VOLTAGE_HIGH_PU
        * 100.0
    )

    result["voltage_violation_severity_pct"] = voltage_score

    result["criticality_score"] = (
        result["voltage_violation_severity_pct"] * 10.0
        + result["incident_overloaded_lines"] * 5.0
        + result["max_incident_line_loading_pct"].fillna(0.0) / 100.0
    )

    result = result.sort_values(
        [
            "voltage_violation",
            "voltage_deviation_pu",
            "incident_overloaded_lines",
            "max_incident_line_loading_pct",
        ],
        ascending=[False, False, False, False],
    )

    result["criticality_rank"] = np.arange(
        1, len(result) + 1
    )

    return result


# ================================================================================================
# GENERATOR / LOAD TABLES
# ================================================================================================

def build_generator_table(network: pypsa.Network) -> pd.DataFrame:

    generators = static_df(network, "Generator").copy()

    p_set = get_static_or_dynamic(
        network, "Generator", "p_set", SNAPSHOT
    )

    q_set = get_static_or_dynamic(
        network, "Generator", "q_set", SNAPSHOT
    )

    p_solved = get_series(
        network, "Generator", "p", SNAPSHOT
    )

    q_solved = get_series(
        network, "Generator", "q", SNAPSHOT
    )

    result = pd.DataFrame(index=generators.index)

    result["bus"] = generators["bus"]
    result["control"] = generators.get(
        "control",
        pd.Series(index=generators.index, dtype=object)
    )

    result["p_set_mw"] = p_set
    result["p_solved_mw"] = p_solved
    result["q_set_mvar"] = q_set
    result["q_solved_mvar"] = q_solved

    result["p_solution_delta_mw"] = (
        result["p_solved_mw"] -
        result["p_set_mw"]
    )

    return result


def build_load_table(network: pypsa.Network) -> pd.DataFrame:

    loads = static_df(network, "Load").copy()

    p_set = get_static_or_dynamic(
        network, "Load", "p_set", SNAPSHOT
    )

    q_set = get_static_or_dynamic(
        network, "Load", "q_set", SNAPSHOT
    )

    p_solved = get_series(
        network, "Load", "p", SNAPSHOT
    )

    q_solved = get_series(
        network, "Load", "q", SNAPSHOT
    )

    result = pd.DataFrame(index=loads.index)

    result["bus"] = loads["bus"]
    result["p_set_mw"] = p_set
    result["p_solved_mw"] = p_solved
    result["q_set_mvar"] = q_set
    result["q_solved_mvar"] = q_solved

    return result


# ================================================================================================
# CONSOLE DIAGNOSTICS
# ================================================================================================

def print_voltage_diagnostics(bus_table: pd.DataFrame) -> None:

    print_header("CRITICAL VOLTAGE BUS IDENTIFICATION")

    finite = bus_table["finite_voltage"]

    print(f"Voltage entries        : {len(bus_table)}")
    print(f"Finite voltage entries : {int(finite.sum())}")

    if finite.any():

        values = bus_table.loc[finite, "v_mag_pu"]

        print(
            f"Minimum voltage       : {values.min():.6f} pu"
        )

        print(
            f"Maximum voltage       : {values.max():.6f} pu"
        )

    critical = bus_table[
        bus_table["voltage_violation"]
    ].copy()

    critical = critical.sort_values(
        "v_mag_pu",
        ascending=True
    )

    print()
    print(
        f"Voltage violations (< {VOLTAGE_LOW_PU:.2f} pu or "
        f"> {VOLTAGE_HIGH_PU:.2f} pu) : {len(critical)}"
    )

    if len(critical) > 0:

        display_columns = [
            "v_mag_pu",
            "v_ang_rad",
            "incident_lines",
            "incident_overloaded_lines",
            "max_incident_line_loading_pct",
            "generator_count",
            "load_count",
            "generation_p_mw",
            "load_p_mw",
        ]

        print()
        print(
            critical[display_columns]
            .head(TOP_N)
            .to_string()
        )


def print_line_diagnostics(line_table: pd.DataFrame) -> None:

    print_header("CRITICAL LINE IDENTIFICATION")

    finite = line_table["finite_loading"]

    print(
        f"Finite line-loading entries : {int(finite.sum())}"
    )

    if finite.any():

        loading = line_table.loc[
            finite, "loading_pct"
        ]

        print(
            f"Maximum line loading       : "
            f"{loading.max():.6f} %"
        )

    overloaded = line_table[
        line_table["overloaded"]
    ].copy()

    overloaded = overloaded.sort_values(
        "loading_pct",
        ascending=False
    )

    print(
        f"Overloaded lines           : {len(overloaded)}"
    )

    if len(overloaded) > 0:

        print()
        print("TOP OVERLOADED LINES:")

        print(
            overloaded[
                [
                    "bus0",
                    "bus1",
                    "s_nom_mva",
                    "s_max_mva",
                    "loading_pct",
                    "overload_pct",
                ]
            ]
            .head(TOP_N)
            .to_string()
        )


def print_transformer_diagnostics(
    transformer_table: pd.DataFrame,
) -> None:

    print_header("CRITICAL TRANSFORMER IDENTIFICATION")

    finite = transformer_table["finite_loading"]

    print(
        f"Finite transformer-loading entries : "
        f"{int(finite.sum())}"
    )

    if finite.any():

        loading = transformer_table.loc[
            finite, "loading_pct"
        ]

        print(
            f"Maximum transformer loading       : "
            f"{loading.max():.6f} %"
        )

    overloaded = transformer_table[
        transformer_table["overloaded"]
    ]

    print(
        f"Overloaded transformers           : "
        f"{len(overloaded)}"
    )


# ================================================================================================
# BOTTLENECK OVERLAP
# ================================================================================================

def identify_bottleneck_overlap(
    bus_table: pd.DataFrame,
    line_table: pd.DataFrame,
) -> pd.DataFrame:

    overloaded_lines = line_table[
        line_table["overloaded"]
    ]

    critical_buses = set(
        bus_table[
            bus_table["voltage_violation"]
        ].index
    )

    overlap_records = []

    for line_name, row in overloaded_lines.iterrows():

        bus0 = row["bus0"]
        bus1 = row["bus1"]

        for bus, endpoint in [
            (bus0, "bus0"),
            (bus1, "bus1"),
        ]:

            overlap_records.append(
                {
                    "line": line_name,
                    "endpoint": endpoint,
                    "bus": bus,
                    "line_loading_pct": row[
                        "loading_pct"
                    ],
                    "voltage_critical": (
                        bus in critical_buses
                    ),
                }
            )

    return pd.DataFrame(overlap_records)


def print_overlap_diagnostics(
    bus_table: pd.DataFrame,
    line_table: pd.DataFrame,
) -> None:

    print_header(
        "VOLTAGE–THERMAL BOTTLENECK OVERLAP"
    )

    overlap = identify_bottleneck_overlap(
        bus_table,
        line_table,
    )

    if overlap.empty:

        print(
            "No overloaded-line endpoints were identified."
        )
        return

    critical_overlap = overlap[
        overlap["voltage_critical"]
    ].copy()

    print(
        f"Overloaded-line endpoints checked : "
        f"{len(overlap)}"
    )

    print(
        f"Endpoints also voltage-critical   : "
        f"{len(critical_overlap)}"
    )

    if len(critical_overlap) > 0:

        print()
        print(
            critical_overlap
            .sort_values(
                "line_loading_pct",
                ascending=False,
            )
            .to_string(index=False)
        )

    else:

        print()
        print(
            "No direct overlap between the selected "
            "voltage-critical buses and overloaded-line "
            "endpoints."
        )


# ================================================================================================
# SUMMARY
# ================================================================================================

def build_summary(
    network: pypsa.Network,
    pf_result,
    bus_table: pd.DataFrame,
    line_table: pd.DataFrame,
    transformer_table: pd.DataFrame,
    generator_table: pd.DataFrame,
    load_table: pd.DataFrame,
) -> pd.DataFrame:

    try:
        converged_raw = pf_result["converged"]

        if isinstance(converged_raw, pd.DataFrame):
            converged = bool(
                converged_raw.loc[
                    SNAPSHOT
                ].all()
            )

        elif isinstance(converged_raw, pd.Series):
            converged = bool(
                converged_raw.loc[SNAPSHOT]
            )

        else:
            converged = bool(converged_raw)

    except Exception:
        converged = False

    try:
        error_raw = pf_result["error"]

        if isinstance(error_raw, pd.DataFrame):
            pf_error = float(
                error_raw.loc[SNAPSHOT].max()
            )

        elif isinstance(error_raw, pd.Series):
            pf_error = float(
                error_raw.loc[SNAPSHOT]
            )

        else:
            pf_error = float(error_raw)

    except Exception:
        pf_error = np.nan

    try:
        n_iter_raw = pf_result["n_iter"]

        if isinstance(n_iter_raw, pd.DataFrame):
            iterations = float(
                n_iter_raw.loc[SNAPSHOT].max()
            )

        elif isinstance(n_iter_raw, pd.Series):
            iterations = float(
                n_iter_raw.loc[SNAPSHOT]
            )

        else:
            iterations = float(n_iter_raw)

    except Exception:
        iterations = np.nan

    finite_voltage = bool(
        bus_table["finite_voltage"].all()
    )

    finite_angle = bool(
        bus_table["finite_angle"].all()
    )

    voltage_values = bus_table.loc[
        bus_table["finite_voltage"],
        "v_mag_pu",
    ]

    angle_values = bus_table.loc[
        bus_table["finite_angle"],
        "v_ang_rad",
    ]

    voltage_range_valid = False

    if len(voltage_values) > 0:

        voltage_range_valid = bool(
            (
                voltage_values >= VOLTAGE_LOW_PU
            ).all()
            and
            (
                voltage_values <= VOLTAGE_HIGH_PU
            ).all()
        )

    angle_range_valid = bool(
        len(angle_values) > 0
    )

    finite_line_loading = bool(
        line_table["finite_loading"].all()
    )

    finite_transformer_loading = bool(
        transformer_table["finite_loading"].all()
    )

    max_line_loading = (
        line_table["loading_pct"].max()
    )

    max_transformer_loading = (
        transformer_table["loading_pct"].max()
    )

    overloaded_lines = int(
        line_table["overloaded"].sum()
    )

    overloaded_transformers = int(
        transformer_table["overloaded"].sum()
    )

    gen_p = generator_table[
        "p_solved_mw"
    ].sum(min_count=1)

    load_p = load_table[
        "p_solved_mw"
    ].sum(min_count=1)

    valid_physical = bool(
        converged
        and finite_voltage
        and voltage_range_valid
        and finite_angle
        and angle_range_valid
        and finite_line_loading
        and finite_transformer_loading
    )

    return pd.DataFrame(
        [
            {
                "stage": "S4.7",
                "network": str(NETWORK_PATH),
                "snapshot": SNAPSHOT,
                "pf": "AC nonlinear",
                "reactive_mode": "ORIGINAL_SOURCE_Q",
                "slack_mode": "DISTRIBUTED",
                "explicit_slack": False,
                "converged": converged,
                "pf_error": pf_error,
                "iterations": iterations,
                "voltage_min_pu": (
                    voltage_values.min()
                    if len(voltage_values)
                    else np.nan
                ),
                "voltage_max_pu": (
                    voltage_values.max()
                    if len(voltage_values)
                    else np.nan
                ),
                "angle_min_rad": (
                    angle_values.min()
                    if len(angle_values)
                    else np.nan
                ),
                "angle_max_rad": (
                    angle_values.max()
                    if len(angle_values)
                    else np.nan
                ),
                "max_line_loading_pct": max_line_loading,
                "overloaded_lines": overloaded_lines,
                "max_transformer_loading_pct":
                    max_transformer_loading,
                "overloaded_transformers":
                    overloaded_transformers,
                "critical_voltage_buses":
                    int(
                        bus_table[
                            "voltage_violation"
                        ].sum()
                    ),
                "solved_generation_mw": gen_p,
                "solved_load_mw": load_p,
                "generation_minus_load_mw":
                    gen_p - load_p,
                "valid_physical_solution":
                    valid_physical,
            }
        ]
    )


# ================================================================================================
# SAVE RESULTS
# ================================================================================================

def save_results(
    summary: pd.DataFrame,
    bus_table: pd.DataFrame,
    line_table: pd.DataFrame,
    transformer_table: pd.DataFrame,
    generator_table: pd.DataFrame,
    load_table: pd.DataFrame,
) -> None:

    print_header("SAVING S4.7 RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    bus_table.to_csv(
        BUS_PATH,
        index=True,
    )

    line_table.to_csv(
        LINE_PATH,
        index=True,
    )

    transformer_table.to_csv(
        TRANSFORMER_PATH,
        index=True,
    )

    generator_table.to_csv(
        GENERATOR_PATH,
        index=True,
    )

    load_table.to_csv(
        LOAD_PATH,
        index=True,
    )

    print(f"Summary            : {SUMMARY_PATH}")
    print(f"Bus criticality    : {BUS_PATH}")
    print(f"Line criticality   : {LINE_PATH}")
    print(f"Transformer        : {TRANSFORMER_PATH}")
    print(f"Generator          : {GENERATOR_PATH}")
    print(f"Load               : {LOAD_PATH}")


# ================================================================================================
# MAIN
# ================================================================================================

def main() -> None:

    print_header(
        "S4.7 — BASELINE BOTTLENECK & CRITICAL-BUS IDENTIFICATION"
    )

    print(
        f"""
Network  : {NETWORK_PATH}
Snapshot : {SNAPSHOT}
PF       : AC nonlinear
Reactive : ORIGINAL SOURCE Q
Slack    : DISTRIBUTED
Source   : READ-ONLY

No reinforcement is applied.
No reactive compensation is added.
No dispatch change is applied.
No load change is applied.
No source network file is modified.
"""
    )

    if not NETWORK_PATH.exists():
        raise FileNotFoundError(
            f"Network file not found: {NETWORK_PATH}"
        )

    # --------------------------------------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------------------------------------

    network = pypsa.Network(
        str(NETWORK_PATH)
    )

    print_network_information(network)

    # --------------------------------------------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------------------------------------------

    isolate_snapshot(network)

    # --------------------------------------------------------------------------------------------
    # OPERATING POINT
    # --------------------------------------------------------------------------------------------

    print_operating_point(network)

    # --------------------------------------------------------------------------------------------
    # TOPOLOGY
    # --------------------------------------------------------------------------------------------

    confirm_topology(network)

    # --------------------------------------------------------------------------------------------
    # SLACK
    # --------------------------------------------------------------------------------------------

    configure_distributed_slack(network)

    # --------------------------------------------------------------------------------------------
    # POWER FLOW
    # --------------------------------------------------------------------------------------------

    pf_result = run_power_flow(network)

    # --------------------------------------------------------------------------------------------
    # EXTRACT SOLUTION
    # --------------------------------------------------------------------------------------------

    print_header("EXTRACTING BASELINE SOLUTION")

    voltage = get_bus_voltage(network)

    line_table = get_line_loading(network)

    transformer_table = get_transformer_loading(
        network
    )

    generator_table = build_generator_table(
        network
    )

    load_table = build_load_table(
        network
    )

    # --------------------------------------------------------------------------------------------
    # BUS CRITICALITY
    # --------------------------------------------------------------------------------------------

    bus_table = build_bus_criticality(
        network,
        voltage,
        line_table,
    )

    # --------------------------------------------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------------------------------------------

    print_voltage_diagnostics(
        bus_table
    )

    print_line_diagnostics(
        line_table
    )

    print_transformer_diagnostics(
        transformer_table
    )

    print_overlap_diagnostics(
        bus_table,
        line_table,
    )

    # --------------------------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------------------------

    summary = build_summary(
        network,
        pf_result,
        bus_table,
        line_table,
        transformer_table,
        generator_table,
        load_table,
    )

    print_header("S4.7 — BASELINE BOTTLENECK SUMMARY")

    print(
        summary.to_string(index=False)
    )

    # --------------------------------------------------------------------------------------------
    # TOP CRITICAL BUSES
    # --------------------------------------------------------------------------------------------

    print_header("TOP CRITICAL BUSES")

    print(
        bus_table[
            [
                "criticality_rank",
                "v_mag_pu",
                "v_ang_rad",
                "voltage_violation",
                "incident_lines",
                "incident_overloaded_lines",
                "max_incident_line_loading_pct",
                "generator_count",
                "load_count",
                "generation_p_mw",
                "load_p_mw",
                "criticality_score",
            ]
        ]
        .head(TOP_N)
        .to_string()
    )

    # --------------------------------------------------------------------------------------------
    # TOP CRITICAL LINES
    # --------------------------------------------------------------------------------------------

    print_header("TOP CRITICAL LINES")

    print(
        line_table[
            [
                "bus0",
                "bus1",
                "s_nom_mva",
                "s_max_mva",
                "loading_pct",
                "overload_pct",
                "overloaded",
            ]
        ]
        .sort_values(
            "loading_pct",
            ascending=False,
        )
        .head(TOP_N)
        .to_string()
    )

    # --------------------------------------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------------------------------------

    save_results(
        summary,
        bus_table,
        line_table,
        transformer_table,
        generator_table,
        load_table,
    )

    # --------------------------------------------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------------------------------------------

    print_header("S4.7 COMPLETE")

    valid = bool(
        summary.iloc[0][
            "valid_physical_solution"
        ]
    )

    critical_bus_count = int(
        summary.iloc[0][
            "critical_voltage_buses"
        ]
    )

    overloaded_line_count = int(
        summary.iloc[0][
            "overloaded_lines"
        ]
    )

    print(
        f"Critical voltage buses : {critical_bus_count}"
    )

    print(
        f"Overloaded lines       : {overloaded_line_count}"
    )

    print(
        "Source network modified : NO"
    )

    print(
        "Reinforcements applied  : NO"
    )

    print(
        "Reactive devices added : NO"
    )

    print(
        "Permanent changes       : NONE"
    )

    print()

    if valid:
        print(
            "BASELINE PHYSICAL STATUS : VALID"
        )
    else:
        print(
            "BASELINE PHYSICAL STATUS : INVALID"
        )

    print()
    print(
        "S4.7 is diagnostic only."
    )


if __name__ == "__main__":
    main()