"""
====================================================================================================
S4.8 — REINFORCEMENT CANDIDATE GENERATION & SCREENING
====================================================================================================

Purpose
-------
Generate and rank reinforcement candidates from the S4.7 baseline bottleneck results.

IMPORTANT
---------
This stage is READ-ONLY.

- Source network is loaded read-only.
- No reinforcement is applied.
- No reactive compensation is added.
- No dispatch change is applied.
- No load change is applied.
- No generator setpoint is changed.
- No source .nc file is overwritten.
- No candidate is physically tested in the network yet.

S4.8 identifies candidate interventions only.

Candidate classes
-----------------
1. THERMAL
   Reinforce overloaded / high-loading AC lines.

2. VOLTAGE
   Identify buses in the severe low-voltage region as locations
   for later controlled reactive-support testing.

3. COMBINED
   Pair thermal candidates with voltage-critical buses/endpoints.

Outputs
-------
data/processed/s4_8_candidate_summary.csv
data/processed/s4_8_thermal_candidates.csv
data/processed/s4_8_voltage_candidates.csv
data/processed/s4_8_combined_candidates.csv

====================================================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = Path(
    "data"
) / "processed" / "eirgrid_second_reinforced_network.nc"

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_DIR = Path("data") / "processed"

SUMMARY_PATH = OUTPUT_DIR / "s4_8_candidate_summary.csv"
THERMAL_PATH = OUTPUT_DIR / "s4_8_thermal_candidates.csv"
VOLTAGE_PATH = OUTPUT_DIR / "s4_8_voltage_candidates.csv"
COMBINED_PATH = OUTPUT_DIR / "s4_8_combined_candidates.csv"

# Physical thresholds
VOLTAGE_LOW_LIMIT = 0.95
VOLTAGE_HIGH_LIMIT = 1.05

LINE_OVERLOAD_LIMIT_PCT = 100.0

# Candidate screening thresholds
THERMAL_CANDIDATE_MIN_LOADING_PCT = 90.0
VOLTAGE_CANDIDATE_MAX_PU = 0.95

# Number of candidates to retain in the final ranked lists
MAX_THERMAL_CANDIDATES = 15
MAX_VOLTAGE_CANDIDATES = 20
MAX_COMBINED_CANDIDATES = 30

# Candidate reinforcement multipliers.
# These are CANDIDATE PARAMETERS ONLY.
# They are NOT applied to the network in S4.8.
THERMAL_MULTIPLIERS = (1.25, 1.50, 2.00)

# Ranking weights
THERMAL_WEIGHT_LOADING = 0.55
THERMAL_WEIGHT_OVERLOAD = 0.30
THERMAL_WEIGHT_VOLTAGE = 0.15

VOLTAGE_WEIGHT_SEVERITY = 0.60
VOLTAGE_WEIGHT_LOADING = 0.20
VOLTAGE_WEIGHT_DEGREE = 0.20


# ==================================================================================================
# DISPLAY HELPERS
# ==================================================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return default


def finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


# ==================================================================================================
# PYPSA ACCESS HELPERS
# ==================================================================================================

def get_static(network: pypsa.Network, component: str) -> pd.DataFrame:
    """
    Prefer the current PyPSA component API.

    Falls back to df() for compatibility with older PyPSA versions.
    """
    try:
        return network.components[component].static
    except Exception:
        return network.df(component)


def get_dynamic(network: pypsa.Network, component: str, attribute: str):
    """
    Prefer current PyPSA dynamic API.

    Falls back to pnl for compatibility.
    """
    try:
        return network.components[component].dynamic[attribute]
    except Exception:
        try:
            return network.components[component].pnl[attribute]
        except Exception:
            return None


def get_snapshot_series(
    network: pypsa.Network,
    component: str,
    attribute: str,
    snapshot: str,
) -> pd.Series:
    dynamic = get_dynamic(network, component, attribute)

    if dynamic is None:
        return pd.Series(dtype=float)

    if snapshot not in dynamic.index:
        return pd.Series(index=dynamic.columns, dtype=float)

    return pd.to_numeric(dynamic.loc[snapshot], errors="coerce")


# ==================================================================================================
# TOPOLOGY
# ==================================================================================================

def build_ac_adjacency(network: pypsa.Network) -> dict[str, set[str]]:
    buses = get_static(network, "Bus")

    adjacency = {
        str(bus): set()
        for bus in buses.index
    }

    for component in ("Line", "Transformer"):
        try:
            df = get_static(network, component)
        except Exception:
            continue

        if df.empty:
            continue

        for _, row in df.iterrows():
            bus0 = str(row.get("bus0", ""))
            bus1 = str(row.get("bus1", ""))

            if bus0 in adjacency and bus1 in adjacency:
                adjacency[bus0].add(bus1)
                adjacency[bus1].add(bus0)

    return adjacency


def connected_component_summary(network: pypsa.Network) -> list[list[str]]:
    adjacency = build_ac_adjacency(network)

    remaining = set(adjacency)
    components = []

    while remaining:
        root = next(iter(remaining))
        stack = [root]
        component = set()

        while stack:
            bus = stack.pop()

            if bus in component:
                continue

            component.add(bus)

            for neighbour in adjacency.get(bus, set()):
                if neighbour not in component:
                    stack.append(neighbour)

        remaining -= component
        components.append(sorted(component))

    components.sort(key=len, reverse=True)
    return components


# ==================================================================================================
# BASELINE SOLUTION EXTRACTION
# ==================================================================================================

def configure_distributed_slack(network: pypsa.Network) -> None:
    """
    Configure the same distributed-slack philosophy used in S4.6/S4.7.

    No explicit slack generator is selected.

    All generators are set to PQ before running the nonlinear PF.
    PyPSA distributed slack then balances the connected AC network.
    """
    generators = get_static(network, "Generator")

    if generators.empty:
        return

    generators["control"] = "PQ"


def run_baseline_pf(network: pypsa.Network):
    configure_distributed_slack(network)

    pf_result = network.pf(
        snapshots=[SNAPSHOT],
        distribute_slack=True,
        x_tol=1e-8,
        use_seed=True,
    )

    return pf_result


# ==================================================================================================
# VOLTAGE EXTRACTION
# ==================================================================================================

def extract_bus_solution(network: pypsa.Network) -> pd.DataFrame:
    buses = get_static(network, "Bus")

    v_mag = get_snapshot_series(
        network,
        "Bus",
        "v_mag_pu",
        SNAPSHOT,
    )

    v_ang = get_snapshot_series(
        network,
        "Bus",
        "v_ang",
        SNAPSHOT,
    )

    result = pd.DataFrame(index=buses.index)

    result.index.name = "name"

    result["v_mag_pu"] = v_mag.reindex(result.index)
    result["v_ang_rad"] = v_ang.reindex(result.index)

    return result


# ==================================================================================================
# LINE SOLUTION EXTRACTION
# ==================================================================================================

def extract_line_solution(network: pypsa.Network) -> pd.DataFrame:
    lines = get_static(network, "Line")

    result = lines.copy()

    result["s_nom_mva"] = pd.to_numeric(
        result.get("s_nom", np.nan),
        errors="coerce",
    )

    s_max = get_snapshot_series(
        network,
        "Line",
        "s",
        SNAPSHOT,
    )

    # PyPSA s is normally the apparent power at the line's first end.
    # Use absolute magnitude for loading screening.
    result["s_max_mva"] = s_max.abs().reindex(result.index)

    result["loading_pct"] = (
        result["s_max_mva"] / result["s_nom_mva"] * 100.0
    )

    result["overload_pct"] = (
        result["loading_pct"] - LINE_OVERLOAD_LIMIT_PCT
    ).clip(lower=0.0)

    result["overloaded"] = (
        result["loading_pct"] > LINE_OVERLOAD_LIMIT_PCT
    )

    return result


# ==================================================================================================
# TRANSFORMER SOLUTION EXTRACTION
# ==================================================================================================

def extract_transformer_solution(network: pypsa.Network) -> pd.DataFrame:
    transformers = get_static(network, "Transformer")

    result = transformers.copy()

    result["s_nom_mva"] = pd.to_numeric(
        result.get("s_nom", np.nan),
        errors="coerce",
    )

    s_max = get_snapshot_series(
        network,
        "Transformer",
        "s",
        SNAPSHOT,
    )

    result["s_max_mva"] = s_max.abs().reindex(result.index)

    result["loading_pct"] = (
        result["s_max_mva"] / result["s_nom_mva"] * 100.0
    )

    result["overloaded"] = (
        result["loading_pct"] > LINE_OVERLOAD_LIMIT_PCT
    )

    return result


# ==================================================================================================
# GENERATOR / LOAD INFORMATION
# ==================================================================================================

def extract_generator_information(network: pypsa.Network) -> pd.DataFrame:
    generators = get_static(network, "Generator")

    result = pd.DataFrame(index=generators.index)
    result.index.name = "name"

    result["bus"] = generators["bus"].astype(str)

    p = get_snapshot_series(
        network,
        "Generator",
        "p",
        SNAPSHOT,
    )

    q = get_snapshot_series(
        network,
        "Generator",
        "q",
        SNAPSHOT,
    )

    result["p_mw"] = p.reindex(result.index).fillna(0.0)
    result["q_mvar"] = q.reindex(result.index).fillna(0.0)

    return result


def extract_load_information(network: pypsa.Network) -> pd.DataFrame:
    loads = get_static(network, "Load")

    result = pd.DataFrame(index=loads.index)
    result.index.name = "name"

    result["bus"] = loads["bus"].astype(str)

    p = get_snapshot_series(
        network,
        "Load",
        "p",
        SNAPSHOT,
    )

    q = get_snapshot_series(
        network,
        "Load",
        "q",
        SNAPSHOT,
    )

    result["p_mw"] = p.reindex(result.index).fillna(0.0)
    result["q_mvar"] = q.reindex(result.index).fillna(0.0)

    return result


# ==================================================================================================
# BUS CRITICALITY
# ==================================================================================================

def build_bus_criticality(
    network: pypsa.Network,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    generators: pd.DataFrame,
    loads: pd.DataFrame,
) -> pd.DataFrame:

    adjacency = build_ac_adjacency(network)

    result = buses.copy()

    result["incident_lines"] = 0
    result["incident_overloaded_lines"] = 0
    result["max_incident_line_loading_pct"] = 0.0
    result["generator_count"] = 0
    result["load_count"] = 0
    result["generation_p_mw"] = 0.0
    result["load_p_mw"] = 0.0

    # Line incidence
    for line_name, row in lines.iterrows():

        bus0 = str(row.get("bus0", ""))
        bus1 = str(row.get("bus1", ""))

        loading = safe_float(row.get("loading_pct"))

        overloaded = bool(row.get("overloaded", False))

        for bus in (bus0, bus1):

            if bus not in result.index:
                continue

            result.loc[bus, "incident_lines"] += 1

            if overloaded:
                result.loc[bus, "incident_overloaded_lines"] += 1

            if np.isfinite(loading):
                result.loc[bus, "max_incident_line_loading_pct"] = max(
                    result.loc[bus, "max_incident_line_loading_pct"],
                    loading,
                )

    # Generator incidence
    for _, row in generators.iterrows():

        bus = str(row["bus"])

        if bus not in result.index:
            continue

        result.loc[bus, "generator_count"] += 1
        result.loc[bus, "generation_p_mw"] += safe_float(
            row["p_mw"],
            0.0,
        )

    # Load incidence
    for _, row in loads.iterrows():

        bus = str(row["bus"])

        if bus not in result.index:
            continue

        result.loc[bus, "load_count"] += 1
        result.loc[bus, "load_p_mw"] += safe_float(
            row["p_mw"],
            0.0,
        )

    # Voltage violation
    result["voltage_violation"] = (
        (result["v_mag_pu"] < VOLTAGE_LOW_LIMIT)
        | (result["v_mag_pu"] > VOLTAGE_HIGH_LIMIT)
    )

    # Voltage severity.
    #
    # A value of 0 means voltage is at or above the lower limit.
    # Increasing values represent increasingly severe undervoltage.
    result["voltage_severity"] = (
        (VOLTAGE_LOW_LIMIT - result["v_mag_pu"])
        .clip(lower=0.0)
        * 100.0
    )

    # Normalized thermal exposure.
    result["thermal_exposure_pct"] = (
        result["max_incident_line_loading_pct"]
        .clip(lower=0.0)
    )

    # Criticality score:
    #
    # 60% voltage severity
    # 20% thermal exposure
    # 20% network degree
    #
    # This is a screening score, NOT a physical sensitivity.
    result["criticality_score"] = (
        VOLTAGE_WEIGHT_SEVERITY
        * result["voltage_severity"]
        + VOLTAGE_WEIGHT_LOADING
        * result["thermal_exposure_pct"]
        + VOLTAGE_WEIGHT_DEGREE
        * result["incident_lines"]
    )

    return result


# ==================================================================================================
# THERMAL CANDIDATE GENERATION
# ==================================================================================================

def generate_thermal_candidates(lines: pd.DataFrame) -> pd.DataFrame:

    candidates = lines[
        lines["loading_pct"] >= THERMAL_CANDIDATE_MIN_LOADING_PCT
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Ranking components
    candidates["loading_score"] = (
        candidates["loading_pct"].clip(lower=0.0)
    )

    candidates["overload_score"] = (
        candidates["overload_pct"].clip(lower=0.0)
    )

    # Thermal candidates are later prioritized toward buses that
    # are also part of the low-voltage region.
    candidates["thermal_priority_score"] = (
        THERMAL_WEIGHT_LOADING
        * candidates["loading_score"]
        + THERMAL_WEIGHT_OVERLOAD
        * candidates["overload_score"]
    )

    candidates = candidates.sort_values(
        "thermal_priority_score",
        ascending=False,
    )

    candidates = candidates.head(
        MAX_THERMAL_CANDIDATES
    ).copy()

    rows = []

    for line_name, row in candidates.iterrows():

        bus0 = str(row["bus0"])
        bus1 = str(row["bus1"])

        current_loading = safe_float(
            row["loading_pct"]
        )

        current_smax = safe_float(
            row["s_max_mva"]
        )

        current_rating = safe_float(
            row["s_nom_mva"]
        )

        for multiplier in THERMAL_MULTIPLIERS:

            candidate_rating = (
                current_rating * multiplier
                if np.isfinite(current_rating)
                else np.nan
            )

            rows.append(
                {
                    "candidate_id": (
                        f"THERMAL_{len(rows) + 1:03d}"
                    ),
                    "candidate_type": "THERMAL",
                    "line": line_name,
                    "bus0": bus0,
                    "bus1": bus1,
                    "current_s_nom_mva": current_rating,
                    "current_s_mva": current_smax,
                    "current_loading_pct": current_loading,
                    "current_overload_pct": max(
                        current_loading - 100.0,
                        0.0,
                    ),
                    "reinforcement_multiplier": multiplier,
                    "candidate_s_nom_mva": candidate_rating,
                    "screening_priority_score": safe_float(
                        row["thermal_priority_score"],
                        0.0,
                    ),
                    "candidate_status": "SCREEN_ONLY",
                    "network_modified": False,
                    "physically_tested": False,
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "screening_priority_score",
            ascending=False,
        ).reset_index(drop=True)

    return result


# ==================================================================================================
# VOLTAGE CANDIDATE GENERATION
# ==================================================================================================

def generate_voltage_candidates(
    buses: pd.DataFrame,
) -> pd.DataFrame:

    candidates = buses[
        buses["v_mag_pu"] < VOLTAGE_CANDIDATE_MAX_PU
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(
        "criticality_score",
        ascending=False,
    ).head(
        MAX_VOLTAGE_CANDIDATES
    )

    rows = []

    for bus_name, row in candidates.iterrows():

        voltage = safe_float(row["v_mag_pu"])

        severity = max(
            VOLTAGE_LOW_LIMIT - voltage,
            0.0,
        )

        rows.append(
            {
                "candidate_id": (
                    f"VOLTAGE_{len(rows) + 1:03d}"
                ),
                "candidate_type": "VOLTAGE",
                "bus": bus_name,
                "v_mag_pu": voltage,
                "v_ang_rad": safe_float(
                    row["v_ang_rad"]
                ),
                "voltage_violation": bool(
                    row["voltage_violation"]
                ),
                "voltage_severity_pu": severity,
                "incident_lines": int(
                    row["incident_lines"]
                ),
                "incident_overloaded_lines": int(
                    row["incident_overloaded_lines"]
                ),
                "max_incident_line_loading_pct": safe_float(
                    row["max_incident_line_loading_pct"],
                    0.0,
                ),
                "generator_count": int(
                    row["generator_count"]
                ),
                "load_count": int(
                    row["load_count"]
                ),
                "generation_p_mw": safe_float(
                    row["generation_p_mw"],
                    0.0,
                ),
                "load_p_mw": safe_float(
                    row["load_p_mw"],
                    0.0,
                ),
                "screening_priority_score": safe_float(
                    row["criticality_score"],
                    0.0,
                ),
                "candidate_status": "SCREEN_ONLY",
                "network_modified": False,
                "physically_tested": False,
            }
        )

    return pd.DataFrame(rows)


# ==================================================================================================
# COMBINED CANDIDATES
# ==================================================================================================

def generate_combined_candidates(
    thermal_candidates: pd.DataFrame,
    voltage_candidates: pd.DataFrame,
    buses: pd.DataFrame,
) -> pd.DataFrame:

    if thermal_candidates.empty or voltage_candidates.empty:
        return pd.DataFrame()

    voltage_lookup = voltage_candidates.set_index(
        "bus"
    )

    rows = []

    for _, thermal in thermal_candidates.iterrows():

        bus0 = str(thermal["bus0"])
        bus1 = str(thermal["bus1"])

        endpoints = []

        if bus0 in voltage_lookup.index:
            endpoints.append(bus0)

        if bus1 in voltage_lookup.index and bus1 != bus0:
            endpoints.append(bus1)

        # If neither endpoint is voltage-critical, the line is still
        # retained as a thermal candidate but is not a direct
        # voltage/thermal overlap candidate.
        if not endpoints:
            continue

        for bus in endpoints:

            voltage = voltage_lookup.loc[bus]

            combined_score = (
                safe_float(
                    thermal["screening_priority_score"],
                    0.0,
                )
                + safe_float(
                    voltage["screening_priority_score"],
                    0.0,
                )
            )

            rows.append(
                {
                    "candidate_id": (
                        f"COMBINED_{len(rows) + 1:03d}"
                    ),
                    "candidate_type": "COMBINED",
                    "line": thermal["line"],
                    "bus0": bus0,
                    "bus1": bus1,
                    "voltage_bus": bus,
                    "current_loading_pct": safe_float(
                        thermal["current_loading_pct"]
                    ),
                    "current_overload_pct": safe_float(
                        thermal["current_overload_pct"]
                    ),
                    "reinforcement_multiplier": safe_float(
                        thermal["reinforcement_multiplier"]
                    ),
                    "candidate_s_nom_mva": safe_float(
                        thermal["candidate_s_nom_mva"]
                    ),
                    "voltage_pu": safe_float(
                        voltage["v_mag_pu"]
                    ),
                    "voltage_severity_pu": max(
                        VOLTAGE_LOW_LIMIT
                        - safe_float(
                            voltage["v_mag_pu"],
                            VOLTAGE_LOW_LIMIT,
                        ),
                        0.0,
                    ),
                    "incident_overloaded_lines": int(
                        voltage["incident_overloaded_lines"]
                    ),
                    "combined_priority_score": combined_score,
                    "candidate_status": "SCREEN_ONLY",
                    "network_modified": False,
                    "physically_tested": False,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        "combined_priority_score",
        ascending=False,
    ).head(
        MAX_COMBINED_CANDIDATES
    ).reset_index(drop=True)

    return result


# ==================================================================================================
# CANDIDATE SUMMARY
# ==================================================================================================

def build_summary(
    network: pypsa.Network,
    pf_result,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    transformers: pd.DataFrame,
    generators: pd.DataFrame,
    loads: pd.DataFrame,
    thermal_candidates: pd.DataFrame,
    voltage_candidates: pd.DataFrame,
    combined_candidates: pd.DataFrame,
) -> pd.DataFrame:

    # PF metadata
    converged = False
    pf_error = np.nan
    iterations = np.nan

    try:
        converged_value = pf_result["converged"].loc[
            SNAPSHOT
        ]

        if isinstance(converged_value, pd.Series):
            converged = bool(converged_value.all())
        else:
            converged = bool(converged_value)

    except Exception:
        converged = False

    try:
        error_value = pf_result["error"].loc[
            SNAPSHOT
        ]

        if isinstance(error_value, pd.Series):
            pf_error = float(error_value.max())
        else:
            pf_error = float(error_value)

    except Exception:
        pass

    try:
        iter_value = pf_result["n_iter"].loc[
            SNAPSHOT
        ]

        if isinstance(iter_value, pd.Series):
            iterations = float(iter_value.max())
        else:
            iterations = float(iter_value)

    except Exception:
        pass

    voltage = finite_series(
        buses["v_mag_pu"]
    )

    angles = finite_series(
        buses["v_ang_rad"]
    )

    loading = finite_series(
        lines["loading_pct"]
    )

    transformer_loading = finite_series(
        transformers["loading_pct"]
    )

    min_voltage = (
        float(voltage.min())
        if voltage.notna().any()
        else np.nan
    )

    max_voltage = (
        float(voltage.max())
        if voltage.notna().any()
        else np.nan
    )

    min_angle = (
        float(angles.min())
        if angles.notna().any()
        else np.nan
    )

    max_angle = (
        float(angles.max())
        if angles.notna().any()
        else np.nan
    )

    max_line_loading = (
        float(loading.max())
        if loading.notna().any()
        else np.nan
    )

    max_transformer_loading = (
        float(transformer_loading.max())
        if transformer_loading.notna().any()
        else np.nan
    )

    critical_voltage_buses = int(
        buses["voltage_violation"].sum()
    )

    overloaded_lines = int(
        lines["overloaded"].sum()
    )

    overloaded_transformers = int(
        transformers["overloaded"].sum()
    )

    finite_voltage = bool(
        voltage.notna().all()
    )

    finite_angles = bool(
        angles.notna().all()
    )

    voltage_valid = bool(
        finite_voltage
        and (voltage >= VOLTAGE_LOW_LIMIT).all()
        and (voltage <= VOLTAGE_HIGH_LIMIT).all()
    )

    angle_valid = bool(
        finite_angles
    )

    line_valid = bool(
        loading.notna().all()
    )

    transformer_valid = bool(
        transformer_loading.notna().all()
    )

    valid_physical_solution = bool(
        converged
        and finite_voltage
        and voltage_valid
        and finite_angles
        and angle_valid
        and line_valid
        and transformer_valid
    )

    solved_generation = float(
        generators["p_mw"].sum()
    )

    solved_load = float(
        loads["p_mw"].sum()
    )

    return pd.DataFrame(
        [
            {
                "stage": "S4.8",
                "network": str(NETWORK_PATH),
                "snapshot": SNAPSHOT,
                "pf": "AC nonlinear",
                "reactive_mode": "ORIGINAL_SOURCE_Q",
                "slack_mode": "DISTRIBUTED",
                "explicit_slack": False,
                "converged": converged,
                "pf_error": pf_error,
                "iterations": iterations,
                "voltage_min_pu": min_voltage,
                "voltage_max_pu": max_voltage,
                "angle_min_rad": min_angle,
                "angle_max_rad": max_angle,
                "max_line_loading_pct": max_line_loading,
                "overloaded_lines": overloaded_lines,
                "max_transformer_loading_pct": max_transformer_loading,
                "overloaded_transformers": overloaded_transformers,
                "critical_voltage_buses": critical_voltage_buses,
                "thermal_candidates": len(
                    thermal_candidates
                ),
                "voltage_candidates": len(
                    voltage_candidates
                ),
                "combined_candidates": len(
                    combined_candidates
                ),
                "solved_generation_mw": solved_generation,
                "solved_load_mw": solved_load,
                "generation_minus_load_mw": (
                    solved_generation
                    - solved_load
                ),
                "valid_physical_solution": valid_physical_solution,
                "network_modified": False,
                "reinforcements_applied": False,
                "reactive_devices_added": False,
                "dispatch_changed": False,
                "load_changed": False,
            }
        ]
    )


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
    )

    banner(
        "S4.8 — REINFORCEMENT CANDIDATE GENERATION & SCREENING"
    )

    print()
    print(f"Network  : {NETWORK_PATH}")
    print(f"Snapshot : {SNAPSHOT}")
    print("PF       : AC nonlinear")
    print("Reactive : ORIGINAL SOURCE Q")
    print("Slack    : DISTRIBUTED")
    print("Source   : READ-ONLY")
    print()
    print("No reinforcement is applied.")
    print("No reactive compensation is added.")
    print("No dispatch change is applied.")
    print("No load change is applied.")
    print("No source network file is modified.")
    print()

    # ----------------------------------------------------------------------------------------------
    # FILE CHECK
    # ----------------------------------------------------------------------------------------------

    if not NETWORK_PATH.exists():
        raise FileNotFoundError(
            f"Source network not found:\n{NETWORK_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------------------------------------

    section("LOADING SOURCE NETWORK")

    network = pypsa.Network(
        str(NETWORK_PATH)
    )

    print(
        f"Buses        : {len(get_static(network, 'Bus'))}"
    )
    print(
        f"Lines        : {len(get_static(network, 'Line'))}"
    )
    print(
        f"Transformers : {len(get_static(network, 'Transformer'))}"
    )
    print(
        f"Generators   : {len(get_static(network, 'Generator'))}"
    )
    print(
        f"Loads        : {len(get_static(network, 'Load'))}"
    )

    # ----------------------------------------------------------------------------------------------
    # SNAPSHOT
    # ----------------------------------------------------------------------------------------------

    section("SNAPSHOT ISOLATION")

    if SNAPSHOT not in network.snapshots:
        raise ValueError(
            f"Snapshot '{SNAPSHOT}' not found in source network."
        )

    print("Active snapshot:")
    print(f"  {SNAPSHOT}")

    # ----------------------------------------------------------------------------------------------
    # ORIGINAL OPERATING POINT
    # ----------------------------------------------------------------------------------------------

    section("ORIGINAL OPERATING POINT")

    generators_static = get_static(
        network,
        "Generator",
    )

    loads_static = get_static(
        network,
        "Load",
    )

    gen_p_set = get_snapshot_series(
        network,
        "Generator",
        "p_set",
        SNAPSHOT,
    )

    load_p_set = get_snapshot_series(
        network,
        "Load",
        "p_set",
        SNAPSHOT,
    )

    # Fallback to p where p_set is unavailable.
    if gen_p_set.empty or gen_p_set.notna().sum() == 0:
        gen_p_set = get_snapshot_series(
            network,
            "Generator",
            "p",
            SNAPSHOT,
        )

    if load_p_set.empty or load_p_set.notna().sum() == 0:
        load_p_set = get_snapshot_series(
            network,
            "Load",
            "p",
            SNAPSHOT,
        )

    gen_p_set = gen_p_set.reindex(
        generators_static.index
    )

    load_p_set = load_p_set.reindex(
        loads_static.index
    )

    gen_p_total = (
        pd.to_numeric(
            gen_p_set,
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    load_p_total = (
        pd.to_numeric(
            load_p_set,
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    gen_q_set = get_snapshot_series(
        network,
        "Generator",
        "q_set",
        SNAPSHOT,
    )

    load_q_set = get_snapshot_series(
        network,
        "Load",
        "q_set",
        SNAPSHOT,
    )

    gen_q_total = (
        pd.to_numeric(
            gen_q_set,
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    load_q_total = (
        pd.to_numeric(
            load_q_set,
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    print(
        f"Generator P set : {gen_p_total:.6f} MW"
    )
    print(
        f"Load P set      : {load_p_total:.6f} MW"
    )
    print(
        f"Generation-load : {gen_p_total - load_p_total:.6f} MW"
    )

    print()
    print(
        f"Generator Q set : {gen_q_total:.6f} Mvar"
    )
    print(
        f"Load Q set      : {load_q_total:.6f} Mvar"
    )

    # ----------------------------------------------------------------------------------------------
    # TOPOLOGY
    # ----------------------------------------------------------------------------------------------

    section("TOPOLOGY CONFIRMATION")

    components = connected_component_summary(
        network
    )

    print(
        f"Total AC connected components : {len(components)}"
    )

    for i, component in enumerate(
        components,
        start=1,
    ):

        print(
            f"Component {i:02d} : {len(component)} buses"
        )

        if len(component) <= 5:
            print(
                f"  Buses: {component}"
            )

    # ----------------------------------------------------------------------------------------------
    # RUN BASELINE PF
    # ----------------------------------------------------------------------------------------------

    section("CONFIGURING BASELINE SLACK")

    print(
        "Explicit slack generator : NONE"
    )
    print(
        "Distributed slack        : True"
    )

    section("RUNNING AC NONLINEAR POWER FLOW")

    print("Configuration:")
    print(
        "  Reactive power : ORIGINAL SOURCE Q"
    )
    print(
        "  Explicit slack : NONE"
    )
    print(
        "  Distributed slack : ENABLED"
    )

    pf_result = run_baseline_pf(
        network
    )

    print()
    print("Raw power-flow result:")
    print(pf_result)

    # ----------------------------------------------------------------------------------------------
    # EXTRACT SOLUTION
    # ----------------------------------------------------------------------------------------------

    section("EXTRACTING BASELINE SOLUTION")

    buses = extract_bus_solution(
        network
    )

    lines = extract_line_solution(
        network
    )

    transformers = extract_transformer_solution(
        network
    )

    generators = extract_generator_information(
        network
    )

    loads = extract_load_information(
        network
    )

    # ----------------------------------------------------------------------------------------------
    # BUS CRITICALITY
    # ----------------------------------------------------------------------------------------------

    section("CRITICAL VOLTAGE BUS IDENTIFICATION")

    buses = build_bus_criticality(
        network,
        buses,
        lines,
        generators,
        loads,
    )

    voltage = finite_series(
        buses["v_mag_pu"]
    )

    print(
        f"Voltage entries        : {len(voltage)}"
    )
    print(
        f"Finite voltage entries : {voltage.notna().sum()}"
    )

    if voltage.notna().any():

        print(
            f"Minimum voltage       : {voltage.min():.6f} pu"
        )
        print(
            f"Maximum voltage       : {voltage.max():.6f} pu"
        )

    critical_count = int(
        buses["voltage_violation"].sum()
    )

    print()
    print(
        f"Voltage violations (< {VOLTAGE_LOW_LIMIT:.2f} pu or > {VOLTAGE_HIGH_LIMIT:.2f} pu) : {critical_count}"
    )

    voltage_display_columns = [
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

    print(
        buses.sort_values(
            "criticality_score",
            ascending=False,
        )[voltage_display_columns].head(15)
    )

    # ----------------------------------------------------------------------------------------------
    # LINE CRITICALITY
    # ----------------------------------------------------------------------------------------------

    section("CRITICAL LINE IDENTIFICATION")

    finite_line_loading = finite_series(
        lines["loading_pct"]
    )

    print(
        f"Finite line-loading entries : {finite_line_loading.notna().sum()}"
    )

    if finite_line_loading.notna().any():

        print(
            f"Maximum line loading       : {finite_line_loading.max():.6f} %"
        )

    overloaded_line_count = int(
        lines["overloaded"].sum()
    )

    print(
        f"Overloaded lines           : {overloaded_line_count}"
    )

    print()
    print("TOP LOADED LINES:")

    line_display_columns = [
        "bus0",
        "bus1",
        "s_nom_mva",
        "s_max_mva",
        "loading_pct",
        "overload_pct",
        "overloaded",
    ]

    print(
        lines.sort_values(
            "loading_pct",
            ascending=False,
        )[line_display_columns].head(15)
    )

    # ----------------------------------------------------------------------------------------------
    # TRANSFORMERS
    # ----------------------------------------------------------------------------------------------

    section("CRITICAL TRANSFORMER IDENTIFICATION")

    finite_transformer_loading = finite_series(
        transformers["loading_pct"]
    )

    print(
        f"Finite transformer-loading entries : {finite_transformer_loading.notna().sum()}"
    )

    if finite_transformer_loading.notna().any():

        print(
            f"Maximum transformer loading       : {finite_transformer_loading.max():.6f} %"
        )

    print(
        f"Overloaded transformers           : {int(transformers['overloaded'].sum())}"
    )

    # ----------------------------------------------------------------------------------------------
    # GENERATE THERMAL CANDIDATES
    # ----------------------------------------------------------------------------------------------

    section("THERMAL REINFORCEMENT CANDIDATE GENERATION")

    thermal_candidates = generate_thermal_candidates(
        lines
    )

    if thermal_candidates.empty:

        print(
            "No thermal candidates met the screening threshold."
        )

    else:

        print(
            f"Thermal candidate rows generated : {len(thermal_candidates)}"
        )

        print()
        print(
            thermal_candidates[
                [
                    "candidate_id",
                    "line",
                    "bus0",
                    "bus1",
                    "current_loading_pct",
                    "current_overload_pct",
                    "reinforcement_multiplier",
                    "candidate_s_nom_mva",
                    "screening_priority_score",
                ]
            ].head(20)
        )

    # ----------------------------------------------------------------------------------------------
    # GENERATE VOLTAGE CANDIDATES
    # ----------------------------------------------------------------------------------------------

    section("VOLTAGE SUPPORT CANDIDATE GENERATION")

    voltage_candidates = generate_voltage_candidates(
        buses
    )

    if voltage_candidates.empty:

        print(
            "No voltage candidates met the screening threshold."
        )

    else:

        print(
            f"Voltage candidate rows generated : {len(voltage_candidates)}"
        )

        print()
        print(
            voltage_candidates[
                [
                    "candidate_id",
                    "bus",
                    "v_mag_pu",
                    "voltage_severity_pu",
                    "incident_lines",
                    "incident_overloaded_lines",
                    "max_incident_line_loading_pct",
                    "screening_priority_score",
                ]
            ].head(20)
        )

    # ----------------------------------------------------------------------------------------------
    # GENERATE COMBINED CANDIDATES
    # ----------------------------------------------------------------------------------------------

    section("COMBINED VOLTAGE–THERMAL CANDIDATE GENERATION")

    combined_candidates = generate_combined_candidates(
        thermal_candidates,
        voltage_candidates,
        buses,
    )

    if combined_candidates.empty:

        print(
            "No direct voltage–thermal overlap candidates generated."
        )

    else:

        print(
            f"Combined candidate rows generated : {len(combined_candidates)}"
        )

        print()
        print(
            combined_candidates[
                [
                    "candidate_id",
                    "line",
                    "voltage_bus",
                    "current_loading_pct",
                    "current_overload_pct",
                    "reinforcement_multiplier",
                    "voltage_pu",
                    "voltage_severity_pu",
                    "combined_priority_score",
                ]
            ].head(30)
        )

    # ----------------------------------------------------------------------------------------------
    # BUILD SUMMARY
    # ----------------------------------------------------------------------------------------------

    section("S4.8 — CANDIDATE SCREENING SUMMARY")

    summary = build_summary(
        network,
        pf_result,
        buses,
        lines,
        transformers,
        generators,
        loads,
        thermal_candidates,
        voltage_candidates,
        combined_candidates,
    )

    print(summary.to_string(index=False))

    # ----------------------------------------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------------------------------------

    section("SAVING S4.8 RESULTS")

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    thermal_candidates.to_csv(
        THERMAL_PATH,
        index=False,
    )

    voltage_candidates.to_csv(
        VOLTAGE_PATH,
        index=False,
    )

    combined_candidates.to_csv(
        COMBINED_PATH,
        index=False,
    )

    print(
        f"Summary             : {SUMMARY_PATH}"
    )
    print(
        f"Thermal candidates  : {THERMAL_PATH}"
    )
    print(
        f"Voltage candidates  : {VOLTAGE_PATH}"
    )
    print(
        f"Combined candidates : {COMBINED_PATH}"
    )

    # ----------------------------------------------------------------------------------------------
    # FINAL STATUS
    # ----------------------------------------------------------------------------------------------

    section("S4.8 COMPLETE")

    print(
        f"Thermal candidates  : {len(thermal_candidates)}"
    )
    print(
        f"Voltage candidates  : {len(voltage_candidates)}"
    )
    print(
        f"Combined candidates : {len(combined_candidates)}"
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
        "Dispatch changed        : NO"
    )
    print(
        "Load changed            : NO"
    )
    print(
        "Permanent changes       : NONE"
    )

    print()
    print(
        "S4.8 is SCREENING ONLY."
    )
    print(
        "No candidate has been physically tested."
    )
    print(
        "Controlled AC evaluation belongs to the next stage."
    )


if __name__ == "__main__":
    main()