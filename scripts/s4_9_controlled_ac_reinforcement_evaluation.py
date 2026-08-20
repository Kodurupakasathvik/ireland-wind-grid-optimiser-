"""
====================================================================================================
S4.9 — CONTROLLED AC REINFORCEMENT EVALUATION
====================================================================================================

Purpose
-------
Controlled diagnostic evaluation of the S4.8 voltage-support candidates.

This stage:
    1. Loads the READ-ONLY S4.8 source network.
    2. Reproduces the S2_PEAK_DEMAND baseline AC nonlinear PF.
    3. Loads the S4.8 voltage candidates.
    4. Tests each candidate independently.
    5. Adds temporary reactive support at one candidate bus.
    6. Runs AC nonlinear PF for each support level.
    7. Measures voltage and thermal response.
    8. Ranks candidates by physical improvement.
    9. Saves diagnostic results only.

IMPORTANT
---------
NO source network file is modified.

NO reinforcement is permanently applied.

NO dispatch change is intentionally applied.

NO load change is applied.

Reactive support is temporary and exists only inside an in-memory
copy of the network for each individual test.

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
    "data"
) / "processed" / "eirgrid_second_reinforced_network.nc"

SNAPSHOT = "S2_PEAK_DEMAND"

VOLTAGE_CANDIDATE_PATH = Path(
    "data"
) / "processed" / "s4_8_voltage_candidates.csv"

OUTPUT_DIR = Path("data") / "processed"

SUMMARY_PATH = OUTPUT_DIR / "s4_9_summary.csv"
CANDIDATE_RESULTS_PATH = OUTPUT_DIR / "s4_9_candidate_results.csv"
VOLTAGE_RESULTS_PATH = OUTPUT_DIR / "s4_9_voltage_results.csv"
THERMAL_RESULTS_PATH = OUTPUT_DIR / "s4_9_thermal_results.csv"
SCREENING_PATH = OUTPUT_DIR / "s4_9_reactive_support_screening.csv"


# Reactive-support sensitivity levels.
# Positive Q = capacitive reactive injection into the AC network.
REACTIVE_SUPPORT_LEVELS_MVAR = [
    25.0,
    50.0,
    100.0,
    150.0,
    200.0,
    300.0,
]


# Physical limits used for evaluation.
VOLTAGE_MIN_LIMIT = 0.95
VOLTAGE_MAX_LIMIT = 1.05

LINE_LOADING_LIMIT_PCT = 100.0
TRANSFORMER_LOADING_LIMIT_PCT = 100.0

PF_TOLERANCE = 1e-6


# We intentionally suppress only pandas/PyPSA deprecation warnings in this
# diagnostic stage. Numerical/runtime errors are NOT suppressed.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ================================================================================================
# FORMATTING HELPERS
# ================================================================================================

def banner(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def subsection(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except (TypeError, ValueError):
        return default


def finite_series(values):
    """Return numeric finite values from an iterable."""
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return s[np.isfinite(s)]


# ================================================================================================
# PYPSA COMPONENT ACCESS
# ================================================================================================

def component_static(network, component):
    """
    Use the current PyPSA component API where possible.
    Falls back to df() for compatibility with older versions.
    """
    try:
        return network.components[component].static
    except Exception:
        return network.df(component)


def component_dynamic(network, component, attribute):
    """
    Return dynamic time-series data using the current PyPSA API.
    """
    try:
        return network.components[component].dynamic[attribute]
    except Exception:
        try:
            return network.pnl(component)[attribute]
        except Exception:
            return pd.DataFrame()


# ================================================================================================
# SNAPSHOT VALIDATION
# ================================================================================================

def validate_snapshot(network):
    if SNAPSHOT not in network.snapshots:
        raise ValueError(
            f"Required snapshot '{SNAPSHOT}' not found in network.\n"
            f"Available snapshots: {list(network.snapshots)}"
        )

    network.set_snapshots([SNAPSHOT])


# ================================================================================================
# ORIGINAL OPERATING POINT
# ================================================================================================

def get_operating_point(network):
    generators = component_static(network, "Generator")
    loads = component_static(network, "Load")

    generator_p = component_dynamic(network, "Generator", "p_set")
    generator_q = component_dynamic(network, "Generator", "q_set")

    load_p = component_dynamic(network, "Load", "p_set")
    load_q = component_dynamic(network, "Load", "q_set")

    if generator_p.empty:
        gen_p_total = 0.0
    else:
        gen_p_total = pd.to_numeric(
            generator_p.loc[SNAPSHOT],
            errors="coerce",
        ).fillna(0.0).sum()

    if load_p.empty:
        load_p_total = 0.0
    else:
        load_p_total = pd.to_numeric(
            load_p.loc[SNAPSHOT],
            errors="coerce",
        ).fillna(0.0).sum()

    if generator_q.empty:
        gen_q_total = 0.0
        gen_q_nans = len(generators)
    else:
        gen_q_values = pd.to_numeric(
            generator_q.loc[SNAPSHOT],
            errors="coerce",
        )
        gen_q_nans = int(gen_q_values.isna().sum())
        gen_q_total = gen_q_values.fillna(0.0).sum()

    if load_q.empty:
        load_q_total = 0.0
        load_q_nans = len(loads)
    else:
        load_q_values = pd.to_numeric(
            load_q.loc[SNAPSHOT],
            errors="coerce",
        )
        load_q_nans = int(load_q_values.isna().sum())
        load_q_total = load_q_values.fillna(0.0).sum()

    return {
        "generator_p_mw": float(gen_p_total),
        "load_p_mw": float(load_p_total),
        "generation_minus_load_mw": float(gen_p_total - load_p_total),
        "generator_q_mvar": float(gen_q_total),
        "load_q_mvar": float(load_q_total),
        "generator_q_nans": gen_q_nans,
        "load_q_nans": load_q_nans,
    }


# ================================================================================================
# TOPOLOGY
# ================================================================================================

def get_ac_components(network):
    """
    Determine AC connected components using PyPSA's topology information.

    Returns a list of sets of bus names.
    """
    try:
        network.determine_network_topology()

        components = []

        for subnetwork in network.sub_networks.index:
            try:
                buses = list(
                    network.sub_networks_t.buses_i().loc[subnetwork]
                )
                components.append(set(buses))
            except Exception:
                pass

        if components:
            return components

    except Exception:
        pass

    # Fallback graph construction.
    buses = list(component_static(network, "Bus").index)
    adjacency = {bus: set() for bus in buses}

    lines = component_static(network, "Line")

    for _, row in lines.iterrows():
        if bool(row.get("active", True)):
            b0 = row["bus0"]
            b1 = row["bus1"]

            if b0 in adjacency and b1 in adjacency:
                adjacency[b0].add(b1)
                adjacency[b1].add(b0)

    transformers = component_static(network, "Transformer")

    for _, row in transformers.iterrows():
        if bool(row.get("active", True)):
            b0 = row["bus0"]
            b1 = row["bus1"]

            if b0 in adjacency and b1 in adjacency:
                adjacency[b0].add(b1)
                adjacency[b1].add(b0)

    components = []
    unseen = set(buses)

    while unseen:
        start = next(iter(unseen))
        stack = [start]
        group = set()

        while stack:
            current = stack.pop()

            if current in group:
                continue

            group.add(current)
            unseen.discard(current)

            stack.extend(adjacency[current] & unseen)

        components.append(group)

    return components


# ================================================================================================
# BASELINE SLACK
# ================================================================================================

def configure_distributed_slack(network):
    """
    Configure the baseline consistently with S4.6-S4.8.

    All generators are initially PQ.

    The nonlinear PF is then run with distributed_slack=True.
    PyPSA internally uses the appropriate slack treatment.
    """
    generators = component_static(network, "Generator").copy()

    if "control" in generators.columns:
        generators.loc[:, "control"] = "PQ"

        # Write through the current component API where possible.
        try:
            network.components["Generator"].static.loc[:, "control"] = "PQ"
        except Exception:
            pass

    return network


# ================================================================================================
# AC POWER FLOW
# ================================================================================================

def run_ac_power_flow(network):
    """
    Run nonlinear AC PF with distributed slack.

    Returns:
        pf_result
        converged
        pf_error
        iterations
    """
    try:
        result = network.pf(
            snapshots=[SNAPSHOT],
            use_seed=True,
            x_tol=PF_TOLERANCE,
            distribute_slack=True,
        )
    except TypeError:
        # Compatibility fallback for PyPSA versions whose keyword names differ.
        result = network.pf(
            snapshots=[SNAPSHOT],
            use_seed=True,
            distribute_slack=True,
        )

    converged = False
    pf_error = np.nan
    iterations = np.nan

    try:
        converged_raw = result["converged"].loc[SNAPSHOT]
        converged_values = pd.Series(converged_raw)

        converged = bool(converged_values.fillna(False).all())
    except Exception:
        try:
            converged = bool(result["converged"].loc[SNAPSHOT])
        except Exception:
            converged = False

    try:
        error_raw = result["error"].loc[SNAPSHOT]
        error_values = pd.to_numeric(
            pd.Series(error_raw),
            errors="coerce",
        )

        if len(error_values):
            pf_error = float(error_values.max())
    except Exception:
        try:
            pf_error = safe_float(result["error"].loc[SNAPSHOT])
        except Exception:
            pass

    try:
        iter_raw = result["n_iter"].loc[SNAPSHOT]
        iter_values = pd.to_numeric(
            pd.Series(iter_raw),
            errors="coerce",
        )

        if len(iter_values):
            iterations = float(iter_values.max())
    except Exception:
        try:
            iterations = safe_float(result["n_iter"].loc[SNAPSHOT])
        except Exception:
            pass

    return result, converged, pf_error, iterations


# ================================================================================================
# BUS VOLTAGE EXTRACTION
# ================================================================================================

def extract_bus_results(network):
    buses = component_static(network, "Bus")

    v_mag = component_dynamic(network, "Bus", "v_mag_pu")
    v_ang = component_dynamic(network, "Bus", "v_ang")

    if v_mag.empty:
        v_mag_snapshot = pd.Series(index=buses.index, dtype=float)
    else:
        v_mag_snapshot = pd.to_numeric(
            v_mag.loc[SNAPSHOT],
            errors="coerce",
        )

    if v_ang.empty:
        v_ang_snapshot = pd.Series(index=buses.index, dtype=float)
    else:
        v_ang_snapshot = pd.to_numeric(
            v_ang.loc[SNAPSHOT],
            errors="coerce",
        )

    v_mag_snapshot = v_mag_snapshot.reindex(buses.index)
    v_ang_snapshot = v_ang_snapshot.reindex(buses.index)

    return pd.DataFrame(
        {
            "bus": buses.index,
            "v_mag_pu": v_mag_snapshot.values,
            "v_ang_rad": v_ang_snapshot.values,
        }
    ).set_index("bus")


# ================================================================================================
# LINE THERMAL EXTRACTION
# ================================================================================================

def extract_line_results(network):
    """
    Extract actual solved AC apparent power from line s0/s1.

    This intentionally does NOT depend on the S4.8 candidate-screening
    loading extraction.

    Loading is:
        max(|s0|, |s1|) / s_nom * 100
    """
    lines = component_static(network, "Line")

    s0 = component_dynamic(network, "Line", "s0")
    s1 = component_dynamic(network, "Line", "s1")

    if s0.empty:
        s0_snapshot = pd.Series(index=lines.index, dtype=float)
    else:
        s0_snapshot = pd.to_numeric(
            s0.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(lines.index)

    if s1.empty:
        s1_snapshot = pd.Series(index=lines.index, dtype=float)
    else:
        s1_snapshot = pd.to_numeric(
            s1.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(lines.index)

    s0_abs = s0_snapshot.abs()
    s1_abs = s1_snapshot.abs()

    max_s = pd.concat(
        [s0_abs.rename("s0"), s1_abs.rename("s1")],
        axis=1,
    ).max(axis=1, skipna=True)

    s_nom = pd.to_numeric(
        lines["s_nom"],
        errors="coerce",
    )

    loading = np.where(
        s_nom > 0,
        max_s / s_nom * 100.0,
        np.nan,
    )

    result = pd.DataFrame(
        {
            "bus0": lines["bus0"],
            "bus1": lines["bus1"],
            "s_nom_mva": s_nom,
            "s0_mva": s0_snapshot,
            "s1_mva": s1_snapshot,
            "max_s_mva": max_s,
            "loading_pct": loading,
        },
        index=lines.index,
    )

    result["overloaded"] = (
        result["loading_pct"] > LINE_LOADING_LIMIT_PCT
    )

    result["overload_pct"] = (
        result["loading_pct"] - LINE_LOADING_LIMIT_PCT
    ).clip(lower=0.0)

    return result


# ================================================================================================
# TRANSFORMER THERMAL EXTRACTION
# ================================================================================================

def extract_transformer_results(network):
    """
    Extract actual solved AC apparent power from transformer s0/s1.
    """
    transformers = component_static(network, "Transformer")

    if transformers.empty:
        return pd.DataFrame(
            columns=[
                "bus0",
                "bus1",
                "s_nom_mva",
                "s0_mva",
                "s1_mva",
                "max_s_mva",
                "loading_pct",
                "overload_pct",
                "overloaded",
            ]
        )

    s0 = component_dynamic(network, "Transformer", "s0")
    s1 = component_dynamic(network, "Transformer", "s1")

    if s0.empty:
        s0_snapshot = pd.Series(
            index=transformers.index,
            dtype=float,
        )
    else:
        s0_snapshot = pd.to_numeric(
            s0.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(transformers.index)

    if s1.empty:
        s1_snapshot = pd.Series(
            index=transformers.index,
            dtype=float,
        )
    else:
        s1_snapshot = pd.to_numeric(
            s1.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(transformers.index)

    s0_abs = s0_snapshot.abs()
    s1_abs = s1_snapshot.abs()

    max_s = pd.concat(
        [s0_abs.rename("s0"), s1_abs.rename("s1")],
        axis=1,
    ).max(axis=1, skipna=True)

    s_nom = pd.to_numeric(
        transformers["s_nom"],
        errors="coerce",
    )

    loading = np.where(
        s_nom > 0,
        max_s / s_nom * 100.0,
        np.nan,
    )

    result = pd.DataFrame(
        {
            "bus0": transformers["bus0"],
            "bus1": transformers["bus1"],
            "s_nom_mva": s_nom,
            "s0_mva": s0_snapshot,
            "s1_mva": s1_snapshot,
            "max_s_mva": max_s,
            "loading_pct": loading,
        },
        index=transformers.index,
    )

    result["overloaded"] = (
        result["loading_pct"] > TRANSFORMER_LOADING_LIMIT_PCT
    )

    result["overload_pct"] = (
        result["loading_pct"] - TRANSFORMER_LOADING_LIMIT_PCT
    ).clip(lower=0.0)

    return result


# ================================================================================================
# GENERATOR / LOAD SOLUTION
# ================================================================================================

def extract_generator_results(network):
    generators = component_static(network, "Generator")

    p_set = component_dynamic(network, "Generator", "p_set")
    q_set = component_dynamic(network, "Generator", "q_set")

    p_solved = component_dynamic(network, "Generator", "p")
    q_solved = component_dynamic(network, "Generator", "q")

    def get_dynamic(df, component_index):
        if df.empty:
            return pd.Series(
                np.nan,
                index=component_index,
                dtype=float,
            )

        return pd.to_numeric(
            df.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(component_index)

    return pd.DataFrame(
        {
            "bus": generators["bus"],
            "control": generators["control"],
            "p_set_mw": get_dynamic(p_set, generators.index),
            "p_solved_mw": get_dynamic(p_solved, generators.index),
            "q_set_mvar": get_dynamic(q_set, generators.index),
            "q_solved_mvar": get_dynamic(q_solved, generators.index),
        },
        index=generators.index,
    )


def extract_load_results(network):
    loads = component_static(network, "Load")

    p_set = component_dynamic(network, "Load", "p_set")
    q_set = component_dynamic(network, "Load", "q_set")

    p_solved = component_dynamic(network, "Load", "p")
    q_solved = component_dynamic(network, "Load", "q")

    def get_dynamic(df, component_index):
        if df.empty:
            return pd.Series(
                np.nan,
                index=component_index,
                dtype=float,
            )

        return pd.to_numeric(
            df.loc[SNAPSHOT],
            errors="coerce",
        ).reindex(component_index)

    return pd.DataFrame(
        {
            "bus": loads["bus"],
            "p_set_mw": get_dynamic(p_set, loads.index),
            "p_solved_mw": get_dynamic(p_solved, loads.index),
            "q_set_mvar": get_dynamic(q_set, loads.index),
            "q_solved_mvar": get_dynamic(q_solved, loads.index),
        },
        index=loads.index,
    )


# ================================================================================================
# SOLUTION METRICS
# ================================================================================================

def calculate_solution_metrics(
    network,
    converged,
    pf_error,
    iterations,
):
    buses = extract_bus_results(network)
    lines = extract_line_results(network)
    transformers = extract_transformer_results(network)
    generators = extract_generator_results(network)
    loads = extract_load_results(network)

    finite_v = finite_series(buses["v_mag_pu"])
    finite_ang = finite_series(buses["v_ang_rad"])

    voltage_finite = (
        len(finite_v) == len(buses)
        and len(buses) > 0
    )

    angle_finite = (
        len(finite_ang) == len(buses)
        and len(buses) > 0
    )

    if len(finite_v):
        voltage_min = float(finite_v.min())
        voltage_max = float(finite_v.max())
    else:
        voltage_min = np.nan
        voltage_max = np.nan

    if len(finite_ang):
        angle_min = float(finite_ang.min())
        angle_max = float(finite_ang.max())
    else:
        angle_min = np.nan
        angle_max = np.nan

    voltage_violations = (
        (buses["v_mag_pu"] < VOLTAGE_MIN_LIMIT)
        | (buses["v_mag_pu"] > VOLTAGE_MAX_LIMIT)
    )

    voltage_violations = voltage_violations.fillna(True)

    critical_voltage_buses = int(voltage_violations.sum())

    finite_line_loading = finite_series(lines["loading_pct"])

    if len(finite_line_loading):
        max_line_loading = float(finite_line_loading.max())
        overloaded_lines = int(
            (finite_line_loading > LINE_LOADING_LIMIT_PCT).sum()
        )
        line_data_valid = True
    else:
        max_line_loading = np.nan
        overloaded_lines = 0
        line_data_valid = False

    finite_transformer_loading = finite_series(
        transformers["loading_pct"]
    )

    if len(finite_transformer_loading):
        max_transformer_loading = float(
            finite_transformer_loading.max()
        )
        overloaded_transformers = int(
            (
                finite_transformer_loading
                > TRANSFORMER_LOADING_LIMIT_PCT
            ).sum()
        )
        transformer_data_valid = True
    else:
        max_transformer_loading = np.nan
        overloaded_transformers = 0
        transformer_data_valid = (
            len(transformers) == 0
        )

    solved_generation = pd.to_numeric(
        generators["p_solved_mw"],
        errors="coerce",
    ).fillna(0.0).sum()

    solved_load = pd.to_numeric(
        loads["p_solved_mw"],
        errors="coerce",
    ).fillna(0.0).sum()

    generation_minus_load = (
        solved_generation - solved_load
    )

    voltage_range_valid = (
        voltage_finite
        and voltage_min >= VOLTAGE_MIN_LIMIT
        and voltage_max <= VOLTAGE_MAX_LIMIT
    )

    angle_range_valid = (
        angle_finite
        and angle_min >= -np.pi
        and angle_max <= np.pi
    )

    valid_physical_solution = (
        bool(converged)
        and voltage_finite
        and voltage_range_valid
        and angle_finite
        and angle_range_valid
        and line_data_valid
        and transformer_data_valid
        and overloaded_lines == 0
        and overloaded_transformers == 0
    )

    return {
        "converged": bool(converged),
        "pf_error": safe_float(pf_error),
        "iterations": safe_float(iterations),

        "voltage_finite": bool(voltage_finite),
        "voltage_range_valid": bool(voltage_range_valid),
        "voltage_min_pu": voltage_min,
        "voltage_max_pu": voltage_max,
        "critical_voltage_buses": critical_voltage_buses,

        "angle_finite": bool(angle_finite),
        "angle_range_valid": bool(angle_range_valid),
        "angle_min_rad": angle_min,
        "angle_max_rad": angle_max,

        "finite_line_loading_entries": int(
            len(finite_line_loading)
        ),
        "max_line_loading_pct": max_line_loading,
        "overloaded_lines": overloaded_lines,
        "line_data_valid": bool(line_data_valid),

        "finite_transformer_loading_entries": int(
            len(finite_transformer_loading)
        ),
        "max_transformer_loading_pct": max_transformer_loading,
        "overloaded_transformers": overloaded_transformers,
        "transformer_data_valid": bool(transformer_data_valid),

        "solved_generation_mw": float(solved_generation),
        "solved_load_mw": float(solved_load),
        "generation_minus_load_mw": float(
            generation_minus_load
        ),

        "valid_physical_solution": bool(
            valid_physical_solution
        ),

        "bus_results": buses,
        "line_results": lines,
        "transformer_results": transformers,
        "generator_results": generators,
        "load_results": loads,
    }


# ================================================================================================
# VOLTAGE CANDIDATE LOADING
# ================================================================================================

def load_voltage_candidates():
    if not VOLTAGE_CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"S4.8 voltage candidate file not found:\n"
            f"{VOLTAGE_CANDIDATE_PATH}"
        )

    candidates = pd.read_csv(VOLTAGE_CANDIDATE_PATH)

    required = {
        "candidate_id",
        "bus",
        "v_mag_pu",
    }

    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(
            "S4.8 candidate file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    candidates = candidates.copy()

    candidates["v_mag_pu"] = pd.to_numeric(
        candidates["v_mag_pu"],
        errors="coerce",
    )

    candidates = candidates[
        candidates["bus"].notna()
    ].copy()

    candidates = candidates.sort_values(
        by=["v_mag_pu", "candidate_id"],
        ascending=[True, True],
    )

    candidates = candidates.reset_index(drop=True)

    return candidates


# ================================================================================================
# TEMPORARY REACTIVE SUPPORT
# ================================================================================================

def add_temporary_reactive_support(
    network,
    bus,
    q_mvar,
):
    """
    Add a temporary PQ generator representing a reactive-support device.

    P = 0 MW
    Q = +q_mvar Mvar

    Positive Q represents reactive injection.

    The device is added only to the in-memory candidate copy.
    """
    buses = component_static(network, "Bus")

    if bus not in buses.index:
        raise ValueError(
            f"Candidate bus '{bus}' does not exist in the network."
        )

    device_name = (
        f"S4_9_TEMP_REACTIVE_SUPPORT__"
        f"{str(bus).replace('/', '_').replace(':', '_')}"
    )

    # Guarantee a unique temporary name.
    existing = set(
        component_static(network, "Generator").index
    )

    base_name = device_name
    counter = 1

    while device_name in existing:
        device_name = f"{base_name}_{counter}"
        counter += 1

    network.add(
        "Generator",
        device_name,
        bus=bus,
        control="PQ",
        p_nom=max(abs(q_mvar), 1.0),
        carrier="S4_9_REACTIVE_SUPPORT",
    )

    # Set P and Q explicitly for the active snapshot.
    try:
        network.components["Generator"].dynamic[
            "p_set"
        ].loc[SNAPSHOT, device_name] = 0.0

        network.components["Generator"].dynamic[
            "q_set"
        ].loc[SNAPSHOT, device_name] = q_mvar

    except Exception:
        # Compatibility fallback.
        try:
            network.generators_t.p_set.loc[
                SNAPSHOT,
                device_name,
            ] = 0.0

            network.generators_t.q_set.loc[
                SNAPSHOT,
                device_name,
            ] = q_mvar
        except Exception as exc:
            raise RuntimeError(
                "Unable to assign temporary reactive-support "
                "P/Q setpoints."
            ) from exc

    return device_name


# ================================================================================================
# CANDIDATE TEST
# ================================================================================================

def test_candidate(
    source_network,
    candidate_id,
    bus,
    q_mvar,
):
    """
    Test one candidate independently on a fresh network copy.
    """
    network = source_network.copy()

    validate_snapshot(network)

    configure_distributed_slack(network)

    device_name = add_temporary_reactive_support(
        network,
        bus,
        q_mvar,
    )

    pf_result, converged, pf_error, iterations = (
        run_ac_power_flow(network)
    )

    metrics = calculate_solution_metrics(
        network,
        converged,
        pf_error,
        iterations,
    )

    return {
        "candidate_id": candidate_id,
        "bus": bus,
        "q_support_mvar": float(q_mvar),
        "temporary_device": device_name,
        "converged": metrics["converged"],
        "pf_error": metrics["pf_error"],
        "iterations": metrics["iterations"],
        "voltage_min_pu": metrics["voltage_min_pu"],
        "voltage_max_pu": metrics["voltage_max_pu"],
        "critical_voltage_buses": metrics[
            "critical_voltage_buses"
        ],
        "max_line_loading_pct": metrics[
            "max_line_loading_pct"
        ],
        "overloaded_lines": metrics[
            "overloaded_lines"
        ],
        "max_transformer_loading_pct": metrics[
            "max_transformer_loading_pct"
        ],
        "overloaded_transformers": metrics[
            "overloaded_transformers"
        ],
        "solved_generation_mw": metrics[
            "solved_generation_mw"
        ],
        "solved_load_mw": metrics[
            "solved_load_mw"
        ],
        "generation_minus_load_mw": metrics[
            "generation_minus_load_mw"
        ],
        "valid_physical_solution": metrics[
            "valid_physical_solution"
        ],
        "line_data_valid": metrics[
            "line_data_valid"
        ],
        "transformer_data_valid": metrics[
            "transformer_data_valid"
        ],
        "bus_results": metrics["bus_results"],
        "line_results": metrics["line_results"],
        "transformer_results": metrics[
            "transformer_results"
        ],
    }


# ================================================================================================
# RANKING
# ================================================================================================

def rank_candidate_results(results, baseline):
    df = pd.DataFrame(results)

    if df.empty:
        return df

    df["voltage_min_improvement_pu"] = (
        df["voltage_min_pu"]
        - baseline["voltage_min_pu"]
    )

    df["voltage_violations_reduced"] = (
        baseline["critical_voltage_buses"]
        - df["critical_voltage_buses"]
    )

    if np.isfinite(
        baseline["max_line_loading_pct"]
    ):
        df["max_line_loading_change_pct"] = (
            df["max_line_loading_pct"]
            - baseline["max_line_loading_pct"]
        )
    else:
        df["max_line_loading_change_pct"] = np.nan

    if np.isfinite(
        baseline["max_transformer_loading_pct"]
    ):
        df["max_transformer_loading_change_pct"] = (
            df["max_transformer_loading_pct"]
            - baseline["max_transformer_loading_pct"]
        )
    else:
        df[
            "max_transformer_loading_change_pct"
        ] = np.nan

    df["thermal_overload_change"] = (
        df["overloaded_lines"]
        - baseline["overloaded_lines"]
    )

    # Primary objective:
    #     eliminate voltage violations.
    #
    # Secondary:
    #     maximize minimum voltage.
    #
    # Third:
    #     avoid creating thermal overloads.
    #
    # Fourth:
    #     minimize Q support.
    df["screening_score"] = (
        df["voltage_violations_reduced"] * 100.0
        + df["voltage_min_improvement_pu"] * 100.0
        - df["overloaded_lines"] * 10.0
        - df["overloaded_transformers"] * 10.0
        - df["q_support_mvar"] * 0.01
    )

    df = df.sort_values(
        by=[
            "critical_voltage_buses",
            "overloaded_lines",
            "overloaded_transformers",
            "q_support_mvar",
            "voltage_min_pu",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    return df


# ================================================================================================
# MAIN
# ================================================================================================

def main():

    banner(
        "S4.9 — CONTROLLED AC REINFORCEMENT EVALUATION"
    )

    print(
        f"""
Network  : {NETWORK_PATH}
Snapshot : {SNAPSHOT}
PF       : AC nonlinear
Reactive : ORIGINAL SOURCE Q + TEMPORARY TEST SUPPORT
Slack    : DISTRIBUTED
Source   : READ-ONLY

S4.8 voltage candidates will be tested independently.

No dispatch change is applied.
No load change is applied.
No permanent reinforcement is applied.
No source network file is modified.
"""
    )

    # ============================================================================================
    # LOAD SOURCE NETWORK
    # ============================================================================================

    subsection("LOADING SOURCE NETWORK")

    if not NETWORK_PATH.exists():
        raise FileNotFoundError(
            f"Source network not found:\n{NETWORK_PATH}"
        )

    source_network = pypsa.Network(NETWORK_PATH)

    validate_snapshot(source_network)

    buses = component_static(source_network, "Bus")
    lines = component_static(source_network, "Line")
    transformers = component_static(
        source_network,
        "Transformer",
    )
    generators = component_static(
        source_network,
        "Generator",
    )
    loads = component_static(
        source_network,
        "Load",
    )

    print(f"Buses        : {len(buses)}")
    print(f"Lines        : {len(lines)}")
    print(f"Transformers : {len(transformers)}")
    print(f"Generators   : {len(generators)}")
    print(f"Loads        : {len(loads)}")

    # ============================================================================================
    # OPERATING POINT
    # ============================================================================================

    subsection("ORIGINAL OPERATING POINT")

    operating_point = get_operating_point(
        source_network
    )

    print(
        f"Generator P set : "
        f"{operating_point['generator_p_mw']:.6f} MW"
    )

    print(
        f"Load P set      : "
        f"{operating_point['load_p_mw']:.6f} MW"
    )

    print(
        f"Generation-load : "
        f"{operating_point['generation_minus_load_mw']:.6f} MW"
    )

    print()

    print(
        f"Generator Q set : "
        f"{operating_point['generator_q_mvar']:.6f} Mvar"
    )

    print(
        f"Load Q set      : "
        f"{operating_point['load_q_mvar']:.6f} Mvar"
    )

    print(
        f"Generator Q NaNs: "
        f"{operating_point['generator_q_nans']}"
    )

    print(
        f"Load Q NaNs     : "
        f"{operating_point['load_q_nans']}"
    )

    # ============================================================================================
    # TOPOLOGY
    # ============================================================================================

    subsection("TOPOLOGY CONFIRMATION")

    components = get_ac_components(source_network)

    print(
        f"Total AC connected components : "
        f"{len(components)}"
    )

    components_sorted = sorted(
        components,
        key=lambda x: (-len(x), sorted(map(str, x))),
    )

    for i, component in enumerate(
        components_sorted,
        start=1,
    ):
        print(
            f"Component {i:02d} : "
            f"{len(component)} buses"
        )

        if len(component) <= 3:
            print(
                f"  Buses: "
                f"{sorted(component)}"
            )

    # ============================================================================================
    # BASELINE PF
    # ============================================================================================

    subsection(
        "RUNNING BASELINE AC NONLINEAR POWER FLOW"
    )

    baseline_network = source_network.copy()

    validate_snapshot(baseline_network)

    configure_distributed_slack(
        baseline_network
    )

    print(
        """
Configuration:
  Reactive power : ORIGINAL SOURCE Q
  Explicit slack : NONE
  Distributed slack : ENABLED
"""
    )

    (
        baseline_pf,
        baseline_converged,
        baseline_error,
        baseline_iterations,
    ) = run_ac_power_flow(
        baseline_network
    )

    print("Raw power-flow result:")
    print(baseline_pf)

    # ============================================================================================
    # BASELINE EXTRACTION
    # ============================================================================================

    subsection("EXTRACTING BASELINE SOLUTION")

    baseline = calculate_solution_metrics(
        baseline_network,
        baseline_converged,
        baseline_error,
        baseline_iterations,
    )

    print(
        f"Minimum voltage       : "
        f"{baseline['voltage_min_pu']:.6f} pu"
    )

    print(
        f"Maximum voltage       : "
        f"{baseline['voltage_max_pu']:.6f} pu"
    )

    print(
        f"Voltage-critical buses: "
        f"{baseline['critical_voltage_buses']}"
    )

    print(
        f"Finite line loadings  : "
        f"{baseline['finite_line_loading_entries']}"
    )

    print(
        f"Maximum line loading  : "
        f"{baseline['max_line_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded lines      : "
        f"{baseline['overloaded_lines']}"
    )

    print(
        f"Finite transformer loadings: "
        f"{baseline['finite_transformer_loading_entries']}"
    )

    print(
        f"Maximum transformer loading: "
        f"{baseline['max_transformer_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded transformers: "
        f"{baseline['overloaded_transformers']}"
    )

    # ============================================================================================
    # BASELINE THERMAL VALIDATION
    # ============================================================================================

    subsection(
        "BASELINE THERMAL VALIDATION"
    )

    baseline_lines = baseline["line_results"]

    finite_baseline_lines = baseline_lines[
        np.isfinite(
            baseline_lines["loading_pct"]
        )
    ]

    if not finite_baseline_lines.empty:

        print(
            f"Finite line-loading entries : "
            f"{len(finite_baseline_lines)}"
        )

        print(
            f"Maximum line loading       : "
            f"{finite_baseline_lines['loading_pct'].max():.6f} %"
        )

        print(
            f"Overloaded lines           : "
            f"{int(finite_baseline_lines['overloaded'].sum())}"
        )

        overloaded = finite_baseline_lines[
            finite_baseline_lines["overloaded"]
        ].sort_values(
            "loading_pct",
            ascending=False,
        )

        if not overloaded.empty:
            print()
            print(
                "TOP BASELINE OVERLOADED LINES:"
            )
            print(
                overloaded[
                    [
                        "bus0",
                        "bus1",
                        "s_nom_mva",
                        "max_s_mva",
                        "loading_pct",
                        "overload_pct",
                    ]
                ].head(15).to_string()
            )

    else:
        print(
            "WARNING: No finite baseline line-loading "
            "values were extracted."
        )

    # ============================================================================================
    # LOAD CANDIDATES
    # ============================================================================================

    subsection(
        "LOADING S4.8 VOLTAGE CANDIDATES"
    )

    candidates = load_voltage_candidates()

    print(
        f"Voltage candidates loaded : "
        f"{len(candidates)}"
    )

    valid_candidate_rows = []

    network_buses = set(
        component_static(
            source_network,
            "Bus",
        ).index
    )

    for _, row in candidates.iterrows():

        bus = str(row["bus"])

        if bus not in network_buses:
            print(
                f"WARNING: Candidate bus not found "
                f"in source network: {bus}"
            )
            continue

        valid_candidate_rows.append(row)

    candidates = pd.DataFrame(
        valid_candidate_rows
    ).reset_index(drop=True)

    print(
        f"Valid candidate buses : "
        f"{len(candidates)}"
    )

    if candidates.empty:
        raise RuntimeError(
            "No valid S4.8 voltage candidates remain."
        )

    print()

    print(
        candidates[
            [
                "candidate_id",
                "bus",
                "v_mag_pu",
            ]
        ].to_string(index=False)
    )

    # ============================================================================================
    # CONTROLLED REACTIVE SUPPORT SCREENING
    # ============================================================================================

    subsection(
        "CONTROLLED REACTIVE SUPPORT SCREENING"
    )

    print(
        "Each candidate is tested independently."
    )

    print(
        "Reactive support levels:"
    )

    print(
        ", ".join(
            f"{q:.0f} Mvar"
            for q in REACTIVE_SUPPORT_LEVELS_MVAR
        )
    )

    print()

    candidate_results = []
    voltage_result_rows = []
    thermal_result_rows = []

    total_tests = (
        len(candidates)
        * len(REACTIVE_SUPPORT_LEVELS_MVAR)
    )

    test_number = 0

    for _, candidate in candidates.iterrows():

        candidate_id = str(
            candidate["candidate_id"]
        )

        bus = str(candidate["bus"])

        print()
        print(
            "-" * 100
        )

        print(
            f"Candidate {candidate_id}"
        )

        print(
            f"Bus      : {bus}"
        )

        print(
            f"Baseline : "
            f"{safe_float(candidate['v_mag_pu']):.6f} pu"
        )

        print(
            "-" * 100
        )

        for q_mvar in REACTIVE_SUPPORT_LEVELS_MVAR:

            test_number += 1

            print(
                f"[{test_number:03d}/{total_tests:03d}] "
                f"{candidate_id} | "
                f"Q = {q_mvar:.0f} Mvar",
                end=" ... ",
                flush=True,
            )

            try:

                result = test_candidate(
                    source_network=source_network,
                    candidate_id=candidate_id,
                    bus=bus,
                    q_mvar=q_mvar,
                )

                candidate_results.append(
                    {
                        key: value
                        for key, value in result.items()
                        if key
                        not in {
                            "bus_results",
                            "line_results",
                            "transformer_results",
                        }
                    }
                )

                # ------------------------------------------------------------------
                # Bus-level result
                # ------------------------------------------------------------------

                bus_results = result[
                    "bus_results"
                ].copy()

                bus_results["candidate_id"] = (
                    candidate_id
                )

                bus_results["candidate_bus"] = bus

                bus_results["q_support_mvar"] = (
                    q_mvar
                )

                bus_results = bus_results.reset_index()

                voltage_result_rows.append(
                    bus_results
                )

                # ------------------------------------------------------------------
                # Line-level result
                # ------------------------------------------------------------------

                line_results = result[
                    "line_results"
                ].copy()

                line_results["candidate_id"] = (
                    candidate_id
                )

                line_results["candidate_bus"] = bus

                line_results["q_support_mvar"] = (
                    q_mvar
                )

                line_results = line_results.reset_index()

                thermal_result_rows.append(
                    line_results
                )

                # ------------------------------------------------------------------
                # Transformer-level result
                # ------------------------------------------------------------------

                transformer_results = result[
                    "transformer_results"
                ].copy()

                if not transformer_results.empty:

                    transformer_results[
                        "candidate_id"
                    ] = candidate_id

                    transformer_results[
                        "candidate_bus"
                    ] = bus

                    transformer_results[
                        "q_support_mvar"
                    ] = q_mvar

                    transformer_results = (
                        transformer_results.reset_index()
                    )

                    # Keep transformers in the thermal output.
                    thermal_result_rows.append(
                        transformer_results.assign(
                            component_type="Transformer"
                        )
                    )

                print(
                    f"Vmin="
                    f"{result['voltage_min_pu']:.4f} pu | "
                    f"Viol="
                    f"{result['critical_voltage_buses']} | "
                    f"LineMax="
                    f"{result['max_line_loading_pct']:.2f}% | "
                    f"OL="
                    f"{result['overloaded_lines']} | "
                    f"Conv="
                    f"{result['converged']}"
                )

            except Exception as exc:

                print(
                    "FAILED"
                )

                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        "bus": bus,
                        "q_support_mvar": q_mvar,
                        "temporary_device": "",
                        "converged": False,
                        "pf_error": np.nan,
                        "iterations": np.nan,
                        "voltage_min_pu": np.nan,
                        "voltage_max_pu": np.nan,
                        "critical_voltage_buses": np.nan,
                        "max_line_loading_pct": np.nan,
                        "overloaded_lines": np.nan,
                        "max_transformer_loading_pct": np.nan,
                        "overloaded_transformers": np.nan,
                        "solved_generation_mw": np.nan,
                        "solved_load_mw": np.nan,
                        "generation_minus_load_mw": np.nan,
                        "valid_physical_solution": False,
                        "line_data_valid": False,
                        "transformer_data_valid": False,
                        "test_error": str(exc),
                    }
                )

    # ============================================================================================
    # RESULT DATAFRAMES
    # ============================================================================================

    candidate_df = pd.DataFrame(
        candidate_results
    )

    if candidate_df.empty:
        raise RuntimeError(
            "No S4.9 candidate results were produced."
        )

    # ============================================================================================
    # RANKING
    # ============================================================================================

    subsection(
        "RANKING CONTROLLED REINFORCEMENT RESULTS"
    )

    candidate_df = rank_candidate_results(
        candidate_df,
        baseline,
    )

    print(
        candidate_df[
            [
                "rank",
                "candidate_id",
                "bus",
                "q_support_mvar",
                "voltage_min_pu",
                "critical_voltage_buses",
                "max_line_loading_pct",
                "overloaded_lines",
                "max_transformer_loading_pct",
                "overloaded_transformers",
                "converged",
                "valid_physical_solution",
                "voltage_min_improvement_pu",
                "voltage_violations_reduced",
            ]
        ].head(30).to_string(index=False)
    )

    # ============================================================================================
    # BEST RESULT
    # ============================================================================================

    subsection(
        "BEST CONTROLLED AC RESULT"
    )

    successful = candidate_df[
        candidate_df["converged"] == True
    ].copy()

    if successful.empty:

        print(
            "No candidate/support combination converged."
        )

        best = None

    else:

        # Choose the candidate with:
        # 1. Fewest voltage violations
        # 2. Fewest overloaded lines
        # 3. Fewest overloaded transformers
        # 4. Highest minimum voltage
        # 5. Lowest Q support
        successful = successful.sort_values(
            by=[
                "critical_voltage_buses",
                "overloaded_lines",
                "overloaded_transformers",
                "voltage_min_pu",
                "q_support_mvar",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                True,
            ],
        )

        best = successful.iloc[0]

        print(
            f"Candidate ID          : "
            f"{best['candidate_id']}"
        )

        print(
            f"Bus                   : "
            f"{best['bus']}"
        )

        print(
            f"Reactive support      : "
            f"{best['q_support_mvar']:.2f} Mvar"
        )

        print(
            f"Minimum voltage       : "
            f"{best['voltage_min_pu']:.6f} pu"
        )

        print(
            f"Maximum voltage       : "
            f"{best['voltage_max_pu']:.6f} pu"
        )

        print(
            f"Critical voltage buses: "
            f"{int(best['critical_voltage_buses'])}"
        )

        print(
            f"Maximum line loading  : "
            f"{best['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded lines      : "
            f"{int(best['overloaded_lines'])}"
        )

        print(
            f"Maximum transformer loading: "
            f"{best['max_transformer_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded transformers: "
            f"{int(best['overloaded_transformers'])}"
        )

        print(
            f"PF converged          : "
            f"{bool(best['converged'])}"
        )

        print(
            f"Physical solution     : "
            f"{bool(best['valid_physical_solution'])}"
        )

    # ============================================================================================
    # VOLTAGE RESULTS
    # ============================================================================================

    if voltage_result_rows:

        voltage_df = pd.concat(
            voltage_result_rows,
            ignore_index=True,
        )

    else:

        voltage_df = pd.DataFrame()

    # ============================================================================================
    # THERMAL RESULTS
    # ============================================================================================

    if thermal_result_rows:

        thermal_df = pd.concat(
            thermal_result_rows,
            ignore_index=True,
        )

    else:

        thermal_df = pd.DataFrame()

    # ============================================================================================
    # SCREENING SUMMARY
    # ============================================================================================

    subsection(
        "S4.9 — CONTROLLED SCREENING SUMMARY"
    )

    best_candidate_id = (
        best["candidate_id"]
        if best is not None
        else ""
    )

    best_bus = (
        best["bus"]
        if best is not None
        else ""
    )

    best_q = (
        safe_float(best["q_support_mvar"])
        if best is not None
        else np.nan
    )

    best_vmin = (
        safe_float(best["voltage_min_pu"])
        if best is not None
        else np.nan
    )

    best_vmax = (
        safe_float(best["voltage_max_pu"])
        if best is not None
        else np.nan
    )

    best_violations = (
        safe_float(
            best["critical_voltage_buses"]
        )
        if best is not None
        else np.nan
    )

    best_line_loading = (
        safe_float(
            best["max_line_loading_pct"]
        )
        if best is not None
        else np.nan
    )

    best_overloaded_lines = (
        safe_float(
            best["overloaded_lines"]
        )
        if best is not None
        else np.nan
    )

    best_transformer_loading = (
        safe_float(
            best["max_transformer_loading_pct"]
        )
        if best is not None
        else np.nan
    )

    best_overloaded_transformers = (
        safe_float(
            best["overloaded_transformers"]
        )
        if best is not None
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "stage": "S4.9",
                "network": str(NETWORK_PATH),
                "snapshot": SNAPSHOT,
                "pf": "AC nonlinear",
                "reactive_mode": (
                    "ORIGINAL_SOURCE_Q"
                    "+TEMPORARY_SUPPORT"
                ),
                "slack_mode": "DISTRIBUTED",
                "explicit_slack": False,
                "baseline_converged": baseline[
                    "converged"
                ],
                "baseline_pf_error": baseline[
                    "pf_error"
                ],
                "baseline_iterations": baseline[
                    "iterations"
                ],
                "baseline_voltage_min_pu": baseline[
                    "voltage_min_pu"
                ],
                "baseline_voltage_max_pu": baseline[
                    "voltage_max_pu"
                ],
                "baseline_critical_voltage_buses": (
                    baseline[
                        "critical_voltage_buses"
                    ]
                ),
                "baseline_max_line_loading_pct": (
                    baseline[
                        "max_line_loading_pct"
                    ]
                ),
                "baseline_overloaded_lines": (
                    baseline[
                        "overloaded_lines"
                    ]
                ),
                "baseline_max_transformer_loading_pct": (
                    baseline[
                        "max_transformer_loading_pct"
                    ]
                ),
                "baseline_overloaded_transformers": (
                    baseline[
                        "overloaded_transformers"
                    ]
                ),
                "candidate_count": len(candidates),
                "support_levels_tested": len(
                    REACTIVE_SUPPORT_LEVELS_MVAR
                ),
                "total_ac_tests": total_tests,
                "best_candidate_id": best_candidate_id,
                "best_candidate_bus": best_bus,
                "best_q_support_mvar": best_q,
                "best_voltage_min_pu": best_vmin,
                "best_voltage_max_pu": best_vmax,
                "best_critical_voltage_buses": (
                    best_violations
                ),
                "best_max_line_loading_pct": (
                    best_line_loading
                ),
                "best_overloaded_lines": (
                    best_overloaded_lines
                ),
                "best_max_transformer_loading_pct": (
                    best_transformer_loading
                ),
                "best_overloaded_transformers": (
                    best_overloaded_transformers
                ),
                "best_physical_solution": (
                    bool(best["valid_physical_solution"])
                    if best is not None
                    else False
                ),
                "source_network_modified": False,
                "reinforcements_permanently_applied": False,
                "dispatch_changed": False,
                "load_changed": False,
                "permanent_changes": False,
            }
        ]
    )

    print(
        summary.to_string(index=False)
    )

    # ============================================================================================
    # SAVE RESULTS
    # ============================================================================================

    subsection(
        "SAVING S4.9 RESULTS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    candidate_df.to_csv(
        CANDIDATE_RESULTS_PATH,
        index=False,
    )

    if not voltage_df.empty:
        voltage_df.to_csv(
            VOLTAGE_RESULTS_PATH,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            VOLTAGE_RESULTS_PATH,
            index=False,
        )

    if not thermal_df.empty:
        thermal_df.to_csv(
            THERMAL_RESULTS_PATH,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            THERMAL_RESULTS_PATH,
            index=False,
        )

    # A compact support-screening file.
    screening_columns = [
        "rank",
        "candidate_id",
        "bus",
        "q_support_mvar",
        "voltage_min_pu",
        "voltage_max_pu",
        "critical_voltage_buses",
        "max_line_loading_pct",
        "overloaded_lines",
        "max_transformer_loading_pct",
        "overloaded_transformers",
        "converged",
        "pf_error",
        "iterations",
        "valid_physical_solution",
        "voltage_min_improvement_pu",
        "voltage_violations_reduced",
        "max_line_loading_change_pct",
        "max_transformer_loading_change_pct",
        "screening_score",
    ]

    screening_columns = [
        col
        for col in screening_columns
        if col in candidate_df.columns
    ]

    candidate_df[
        screening_columns
    ].to_csv(
        SCREENING_PATH,
        index=False,
    )

    print(
        f"Summary             : {SUMMARY_PATH}"
    )

    print(
        f"Candidate results   : {CANDIDATE_RESULTS_PATH}"
    )

    print(
        f"Voltage results     : {VOLTAGE_RESULTS_PATH}"
    )

    print(
        f"Thermal results     : {THERMAL_RESULTS_PATH}"
    )

    print(
        f"Screening           : {SCREENING_PATH}"
    )

    # ============================================================================================
    # FINAL STATUS
    # ============================================================================================

    subsection(
        "S4.9 COMPLETE"
    )

    print(
        f"Candidates tested           : {len(candidates)}"
    )

    print(
        f"Reactive levels per candidate : "
        f"{len(REACTIVE_SUPPORT_LEVELS_MVAR)}"
    )

    print(
        f"Total controlled AC tests   : "
        f"{total_tests}"
    )

    print(
        f"Baseline minimum voltage   : "
        f"{baseline['voltage_min_pu']:.6f} pu"
    )

    print(
        f"Baseline critical buses    : "
        f"{baseline['critical_voltage_buses']}"
    )

    print(
        f"Baseline overloaded lines  : "
        f"{baseline['overloaded_lines']}"
    )

    if best is not None:

        print()
        print(
            "BEST CONTROLLED CANDIDATE"
        )

        print(
            f"Candidate : "
            f"{best['candidate_id']}"
        )

        print(
            f"Bus       : "
            f"{best['bus']}"
        )

        print(
            f"Q support : "
            f"{best['q_support_mvar']:.2f} Mvar"
        )

        print(
            f"Vmin      : "
            f"{best['voltage_min_pu']:.6f} pu"
        )

        print(
            f"V critical: "
            f"{int(best['critical_voltage_buses'])}"
        )

        print(
            f"Line max  : "
            f"{best['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Line OL   : "
            f"{int(best['overloaded_lines'])}"
        )

    print()
    print(
        "Source network modified       : NO"
    )

    print(
        "Reinforcements permanently applied : NO"
    )

    print(
        "Reactive devices permanently added : NO"
    )

    print(
        "Dispatch changed              : NO"
    )

    print(
        "Load changed                  : NO"
    )

    print(
        "Permanent changes             : NONE"
    )

    print()
    print(
        "S4.9 IS CONTROLLED SCREENING ONLY."
    )

    print(
        "The best candidate is NOT permanently applied."
    )

    print(
        "Final reinforcement selection belongs to the next stage."
    )


# ================================================================================================
# ENTRY POINT
# ================================================================================================

if __name__ == "__main__":
    main()