"""
======================================================================
IRELAND GRID - S2 SCENARIO DATA DIAGNOSTIC
======================================================================

Purpose
-------
Diagnose why S2_PEAK_DEMAND does not converge in the AC power flow.

This script:
  - DOES NOT modify any network
  - DOES NOT reinforce anything
  - DOES NOT change network parameters
  - DOES NOT declare bottlenecks resolved
  - DOES NOT trust failed AC loading values
  - Inspects the actual scenario data stored in PyPSA
  - Checks generator/load time series
  - Checks controllability and dispatch settings
  - Checks bus connectivity
  - Checks slack/reference-bus availability
  - Checks transformer and line topology
  - Performs a controlled S2 diagnostic AC solve
  - Saves diagnostic CSV files

======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

OPTIMIZED_PATH = DATA / "eirgrid_optimized_network.nc"
ITERATION2_PATH = DATA / "eirgrid_second_reinforced_network.nc"

OUTPUT_SUMMARY = DATA / "s2_scenario_data_diagnostic.csv"
OUTPUT_GENERATORS = DATA / "s2_generator_scenario_data.csv"
OUTPUT_LOADS = DATA / "s2_load_scenario_data.csv"
OUTPUT_BUSES = DATA / "s2_bus_scenario_diagnostic.csv"
OUTPUT_TOPOLOGY = DATA / "s2_topology_diagnostic.csv"


SCENARIO = "S2_PEAK_DEMAND"


# ---------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------

WIDTH = 70


def header(title):
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def subheader(title):
    print()
    print("-" * WIDTH)
    print(title)
    print("-" * WIDTH)


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return np.nan
    except Exception:
        return np.nan


def finite_values(series):
    if series is None:
        return np.array([], dtype=float)

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    return values[np.isfinite(values)]


def print_series_stats(label, series):
    values = finite_values(series)

    if len(values) == 0:
        print(f"{label:<35}: NO FINITE VALUES")
        return

    print(f"{label:<35}:")
    print(f"  count       : {len(values)}")
    print(f"  min         : {np.min(values):.6f}")
    print(f"  max         : {np.max(values):.6f}")
    print(f"  mean        : {np.mean(values):.6f}")
    print(f"  nonzero     : {np.count_nonzero(np.abs(values) > 1e-12)}")


# ---------------------------------------------------------------------
# LOAD NETWORK
# ---------------------------------------------------------------------

def load_network(path, label):
    header(f"LOADING {label.upper()}")

    print(path)

    if not path.exists():
        raise FileNotFoundError(f"Network not found: {path}")

    n = pypsa.Network(path)

    print("OK: Network loaded.")

    print()
    print("NETWORK")
    print(f"  Buses        : {len(n.buses)}")
    print(f"  Lines        : {len(n.lines)}")
    print(f"  Transformers : {len(n.transformers)}")
    print(f"  Generators   : {len(n.generators)}")
    print(f"  Loads        : {len(n.loads)}")

    print()
    print("SNAPSHOTS")
    print(list(n.snapshots))

    return n


# ---------------------------------------------------------------------
# SCENARIO EXISTENCE
# ---------------------------------------------------------------------

def check_scenario(n, label):
    subheader(f"{label.upper()} SCENARIO CHECK")

    snapshots = list(n.snapshots)

    exists = SCENARIO in snapshots

    print(f"Scenario requested : {SCENARIO}")
    print(f"Scenario present   : {exists}")

    if not exists:
        return False

    return True


# ---------------------------------------------------------------------
# GENERATOR DATA
# ---------------------------------------------------------------------

def diagnose_generators(n, label):
    subheader(f"{label.upper()} GENERATOR SCENARIO DATA")

    rows = []

    for name, row in n.generators.iterrows():

        p_nom = safe_float(row.get("p_nom", np.nan))
        p_min_pu = safe_float(row.get("p_min_pu", np.nan))
        p_max_pu = safe_float(row.get("p_max_pu", np.nan))
        control = row.get("control", "")
        bus = row.get("bus", "")
        carrier = row.get("carrier", "")

        p_set = np.nan
        p_min = np.nan
        p_max = np.nan

        if hasattr(n.generators_t, "p_set"):
            if name in n.generators_t.p_set.index:
                if SCENARIO in n.generators_t.p_set.columns:
                    p_set = safe_float(
                        n.generators_t.p_set.loc[name, SCENARIO]
                    )

        if hasattr(n.generators_t, "p_min_pu"):
            if name in n.generators_t.p_min_pu.index:
                if SCENARIO in n.generators_t.p_min_pu.columns:
                    p_min = safe_float(
                        n.generators_t.p_min_pu.loc[name, SCENARIO]
                    )

        if hasattr(n.generators_t, "p_max_pu"):
            if name in n.generators_t.p_max_pu.index:
                if SCENARIO in n.generators_t.p_max_pu.columns:
                    p_max = safe_float(
                        n.generators_t.p_max_pu.loc[name, SCENARIO]
                    )

        rows.append(
            {
                "network": label,
                "scenario": SCENARIO,
                "generator": name,
                "bus": bus,
                "carrier": carrier,
                "p_nom": p_nom,
                "p_min_pu_static": p_min_pu,
                "p_max_pu_static": p_max_pu,
                "control": control,
                "p_set": p_set,
                "p_min_pu_scenario": p_min,
                "p_max_pu_scenario": p_max,
                "p_set_finite": np.isfinite(p_set),
                "p_set_nonzero": (
                    np.isfinite(p_set) and abs(p_set) > 1e-12
                ),
            }
        )

        print(
            f"{name} | "
            f"bus={bus} | "
            f"carrier={carrier} | "
            f"p_nom={p_nom} | "
            f"control={control} | "
            f"p_set={p_set} | "
            f"p_max_pu={p_max}"
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:

        print()
        print("GENERATOR SUMMARY")

        print(
            f"Total p_nom                    : "
            f"{df['p_nom'].fillna(0).sum():.6f}"
        )

        print(
            f"Finite p_set count             : "
            f"{df['p_set_finite'].sum()}"
        )

        print(
            f"Non-zero p_set count           : "
            f"{df['p_set_nonzero'].sum()}"
        )

        finite_p_set = df.loc[df["p_set_finite"], "p_set"]

        if len(finite_p_set) > 0:
            print(
                f"Finite p_set total             : "
                f"{finite_p_set.sum():.6f}"
            )
        else:
            print("Finite p_set total             : 0.000000")

    return df


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def diagnose_loads(n, label):
    subheader(f"{label.upper()} LOAD SCENARIO DATA")

    rows = []

    for name, row in n.loads.iterrows():

        bus = row.get("bus", "")
        carrier = row.get("carrier", "")

        p_set = np.nan
        q_set = np.nan

        if hasattr(n.loads_t, "p_set"):
            if name in n.loads_t.p_set.index:
                if SCENARIO in n.loads_t.p_set.columns:
                    p_set = safe_float(
                        n.loads_t.p_set.loc[name, SCENARIO]
                    )

        if hasattr(n.loads_t, "q_set"):
            if name in n.loads_t.q_set.index:
                if SCENARIO in n.loads_t.q_set.columns:
                    q_set = safe_float(
                        n.loads_t.q_set.loc[name, SCENARIO]
                    )

        static_p = safe_float(row.get("p_set", np.nan))
        static_q = safe_float(row.get("q_set", np.nan))

        rows.append(
            {
                "network": label,
                "scenario": SCENARIO,
                "load": name,
                "bus": bus,
                "carrier": carrier,
                "static_p_set": static_p,
                "static_q_set": static_q,
                "scenario_p_set": p_set,
                "scenario_q_set": q_set,
                "p_set_finite": np.isfinite(p_set),
                "p_set_nonzero": (
                    np.isfinite(p_set) and abs(p_set) > 1e-12
                ),
                "q_set_finite": np.isfinite(q_set),
            }
        )

        print(
            f"{name} | "
            f"bus={bus} | "
            f"carrier={carrier} | "
            f"static_p={static_p} | "
            f"scenario_p={p_set} | "
            f"scenario_q={q_set}"
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:

        print()
        print("LOAD SUMMARY")

        print(
            f"Finite p_set count             : "
            f"{df['p_set_finite'].sum()}"
        )

        print(
            f"Non-zero p_set count           : "
            f"{df['p_set_nonzero'].sum()}"
        )

        finite_p = df.loc[df["p_set_finite"], "scenario_p_set"]

        if len(finite_p) > 0:
            print(
                f"Finite scenario p_set total   : "
                f"{finite_p.sum():.6f}"
            )
        else:
            print("Finite scenario p_set total   : 0.000000")

        finite_q = df.loc[df["q_set_finite"], "scenario_q_set"]

        if len(finite_q) > 0:
            print(
                f"Finite scenario q_set total   : "
                f"{finite_q.sum():.6f}"
            )
        else:
            print("Finite scenario q_set total   : 0.000000")

    return df


# ---------------------------------------------------------------------
# BUS DIAGNOSTIC
# ---------------------------------------------------------------------

def diagnose_buses(n, label):
    subheader(f"{label.upper()} BUS DIAGNOSTIC")

    rows = []

    for name, row in n.buses.iterrows():

        v_nom = safe_float(row.get("v_nom", np.nan))
        carrier = row.get("carrier", "")
        x = safe_float(row.get("x", np.nan))
        y = safe_float(row.get("y", np.nan))

        rows.append(
            {
                "network": label,
                "bus": name,
                "v_nom": v_nom,
                "carrier": carrier,
                "x": x,
                "y": y,
            }
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:

        print(f"Minimum v_nom : {df['v_nom'].min():.6f}")
        print(f"Maximum v_nom : {df['v_nom'].max():.6f}")

        invalid = df[
            ~np.isfinite(df["v_nom"]) |
            (df["v_nom"] <= 0)
        ]

        print(f"Invalid v_nom : {len(invalid)}")

        print()
        print("BUS CARRIERS")

        print(
            df["carrier"]
            .fillna("<NA>")
            .value_counts(dropna=False)
            .to_string()
        )

    return df


# ---------------------------------------------------------------------
# CONNECTIVITY
# ---------------------------------------------------------------------

def diagnose_connectivity(n, label):
    subheader(f"{label.upper()} CONNECTIVITY")

    # Build undirected graph from lines and transformers.
    edges = []

    for name, row in n.lines.iterrows():
        edges.append(
            (
                row["bus0"],
                row["bus1"],
                "line",
                name,
            )
        )

    for name, row in n.transformers.iterrows():
        edges.append(
            (
                row["bus0"],
                row["bus1"],
                "transformer",
                name,
            )
        )

    adjacency = {bus: set() for bus in n.buses.index}

    for bus0, bus1, component_type, name in edges:

        if bus0 in adjacency:
            adjacency[bus0].add(bus1)

        if bus1 in adjacency:
            adjacency[bus1].add(bus0)

    # Connected components.
    components = []

    unvisited = set(adjacency.keys())

    while unvisited:

        start = next(iter(unvisited))
        stack = [start]
        component = set()

        while stack:

            bus = stack.pop()

            if bus in component:
                continue

            component.add(bus)
            unvisited.discard(bus)

            for neighbor in adjacency.get(bus, []):
                if neighbor not in component:
                    stack.append(neighbor)

        components.append(component)

    print(f"Connected components : {len(components)}")

    for i, component in enumerate(
        sorted(components, key=len, reverse=True),
        start=1,
    ):
        print(
            f"  Component {i}: "
            f"{len(component)} buses"
        )

    rows = []

    for i, component in enumerate(
        sorted(components, key=len, reverse=True),
        start=1,
    ):

        for bus in component:

            rows.append(
                {
                    "network": label,
                    "component": i,
                    "component_size": len(component),
                    "bus": bus,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# SLACK / CONTROL DIAGNOSTIC
# ---------------------------------------------------------------------

def diagnose_controls(n, label):
    subheader(f"{label.upper()} GENERATOR CONTROL")

    rows = []

    for name, row in n.generators.iterrows():

        rows.append(
            {
                "network": label,
                "generator": name,
                "bus": row.get("bus", ""),
                "control": row.get("control", ""),
                "p_nom": safe_float(row.get("p_nom", np.nan)),
                "carrier": row.get("carrier", ""),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:

        print(
            df[
                [
                    "generator",
                    "bus",
                    "control",
                    "p_nom",
                    "carrier",
                ]
            ].to_string(index=False)
        )

        print()

        controls = (
            df["control"]
            .fillna("<NA>")
            .astype(str)
            .value_counts()
        )

        print("CONTROL COUNTS")
        print(controls.to_string())

    return df


# ---------------------------------------------------------------------
# LINE / TRANSFORMER TOPOLOGY
# ---------------------------------------------------------------------

def diagnose_topology(n, label):
    subheader(f"{label.upper()} TOPOLOGY")

    rows = []

    for name, row in n.lines.iterrows():

        rows.append(
            {
                "network": label,
                "component": "line",
                "name": name,
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "r": safe_float(row.get("r", np.nan)),
                "x": safe_float(row.get("x", np.nan)),
                "s_nom": safe_float(row.get("s_nom", np.nan)),
            }
        )

    for name, row in n.transformers.iterrows():

        rows.append(
            {
                "network": label,
                "component": "transformer",
                "name": name,
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "r": safe_float(row.get("r", np.nan)),
                "x": safe_float(row.get("x", np.nan)),
                "s_nom": safe_float(row.get("s_nom", np.nan)),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:

        invalid = df[
            ~np.isfinite(df["r"]) |
            ~np.isfinite(df["x"]) |
            ~np.isfinite(df["s_nom"]) |
            (df["s_nom"] <= 0)
        ]

        print(f"Topology components : {len(df)}")
        print(f"Invalid components  : {len(invalid)}")

    return df


# ---------------------------------------------------------------------
# AC TEST
# ---------------------------------------------------------------------

def run_ac_test(n, label):
    subheader(f"{label.upper()} S2 AC TEST")

    result = {
        "network": label,
        "scenario": SCENARIO,
        "status": "NOT_RUN",
        "max_loading_pct": np.nan,
        "overloaded_lines": np.nan,
        "finite_loading": False,
    }

    if SCENARIO not in n.snapshots:
        result["status"] = "SCENARIO_MISSING"
        print("Scenario missing.")
        return result

    test = n.copy()

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pf_result = test.pf(
                snapshots=[SCENARIO],
                x_tol=1e-6,
            )

        # PyPSA returns converged status.
        converged = False

        try:
            converged = bool(
                np.asarray(
                    pf_result["converged"]
                ).reshape(-1)[0]
            )
        except Exception:
            pass

        result["converged"] = converged

        if not converged:
            result["status"] = "NOT_CONVERGED"
            print("AC power flow : NOT CONVERGED")
            print("Loading data  : REJECTED")
            return result

        # Calculate loading only after convergence.
        loading_candidates = []

        if hasattr(test.lines_t, "p0"):
            p0 = test.lines_t.p0.loc[SCENARIO]

            s_nom = test.lines["s_nom"]

            loading = (
                np.abs(p0) /
                s_nom.replace(0, np.nan)
                * 100.0
            )

            loading_candidates.extend(
                pd.to_numeric(
                    loading,
                    errors="coerce"
                ).to_numpy()
            )

        if hasattr(test.lines_t, "q0"):
            q0 = test.lines_t.q0.loc[SCENARIO]

            s_nom = test.lines["s_nom"]

            q_loading = (
                np.abs(q0) /
                s_nom.replace(0, np.nan)
                * 100.0
            )

            loading_candidates.extend(
                pd.to_numeric(
                    q_loading,
                    errors="coerce"
                ).to_numpy()
            )

        values = np.asarray(
            loading_candidates,
            dtype=float,
        )

        finite = values[np.isfinite(values)]

        if len(finite) == 0:
            result["status"] = "NO_FINITE_LOADING"
            print("AC power flow : CONVERGED")
            print("Loading data  : NO FINITE VALUES")
            return result

        max_loading = float(np.max(finite))

        result["max_loading_pct"] = max_loading
        result["finite_loading"] = True
        result["status"] = "VALID"

        if hasattr(test.lines_t, "p0"):

            p0 = test.lines_t.p0.loc[SCENARIO]

            line_loading = (
                np.abs(p0) /
                test.lines["s_nom"].replace(
                    0,
                    np.nan
                )
                * 100.0
            )

            overloaded = int(
                (
                    line_loading > 100.0
                ).sum()
            )

            result["overloaded_lines"] = overloaded

        print("AC power flow : VALID")
        print(f"Maximum loading : {max_loading:.6f}%")
        print(
            f"Overloaded lines : "
            f"{result['overloaded_lines']}"
        )

    except Exception as exc:

        result["status"] = "ERROR"

        print("AC power flow : ERROR")
        print(f"Error: {type(exc).__name__}: {exc}")

    return result


# ---------------------------------------------------------------------
# SUMMARY INTERPRETATION
# ---------------------------------------------------------------------

def interpret(generator_df, load_df, controls_df):
    subheader("DIAGNOSTIC INTERPRETATION")

    finite_gen = (
        generator_df["p_set_finite"].sum()
        if len(generator_df)
        else 0
    )

    nonzero_gen = (
        generator_df["p_set_nonzero"].sum()
        if len(generator_df)
        else 0
    )

    finite_load = (
        load_df["p_set_finite"].sum()
        if len(load_df)
        else 0
    )

    nonzero_load = (
        load_df["p_set_nonzero"].sum()
        if len(load_df)
        else 0
    )

    print(
        f"Generators with finite S2 p_set : "
        f"{finite_gen}"
    )

    print(
        f"Generators with non-zero S2 p_set : "
        f"{nonzero_gen}"
    )

    print(
        f"Loads with finite S2 p_set       : "
        f"{finite_load}"
    )

    print(
        f"Loads with non-zero S2 p_set     : "
        f"{nonzero_load}"
    )

    if nonzero_gen == 0:
        print()
        print(
            "WARNING: No generator has non-zero "
            "S2 dispatch."
        )

    if nonzero_load == 0:
        print()
        print(
            "WARNING: No load has non-zero "
            "S2 dispatch."
        )

    if nonzero_gen == 0 and nonzero_load == 0:
        print()
        print(
            "CRITICAL FINDING:"
        )
        print(
            "S2_PEAK_DEMAND does not appear to contain "
            "actual scenario dispatch values."
        )
        print(
            "The next fix should target scenario-data "
            "construction, not transmission reinforcement."
        )

    if len(controls_df) > 0:

        slack_like = controls_df[
            controls_df["control"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "slack",
                    "pv",
                    "generator",
                ]
            )
        ]

        print()
        print(
            f"Generators with explicit control "
            f"settings: {len(slack_like)}"
        )

    print()
    print(
        "Do NOT use the enormous failed-S2 loading "
        "numbers as physical overloads."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    header(
        "IRELAND GRID - S2 SCENARIO DATA DIAGNOSTIC"
    )

    print()
    print("DIAGNOSTIC MODE")
    print()
    print("This script:")
    print("  - DOES NOT modify networks")
    print("  - DOES NOT reinforce lines")
    print("  - DOES NOT declare bottlenecks resolved")
    print("  - Does not trust failed AC loading values")
    print("  - Inspects actual S2 scenario data")
    print()

    optimized = load_network(
        OPTIMIZED_PATH,
        "Optimized network",
    )

    iteration2 = load_network(
        ITERATION2_PATH,
        "Iteration-2 network",
    )

    if not check_scenario(
        optimized,
        "Optimized",
    ):
        raise RuntimeError(
            "S2_PEAK_DEMAND is missing from optimized network."
        )

    if not check_scenario(
        iteration2,
        "Iteration-2",
    ):
        raise RuntimeError(
            "S2_PEAK_DEMAND is missing from iteration-2 network."
        )

    # --------------------------------------------------------------
    # OPTIMIZED
    # --------------------------------------------------------------

    opt_generators = diagnose_generators(
        optimized,
        "Optimized",
    )

    opt_loads = diagnose_loads(
        optimized,
        "Optimized",
    )

    opt_buses = diagnose_buses(
        optimized,
        "Optimized",
    )

    opt_controls = diagnose_controls(
        optimized,
        "Optimized",
    )

    opt_connectivity = diagnose_connectivity(
        optimized,
        "Optimized",
    )

    opt_topology = diagnose_topology(
        optimized,
        "Optimized",
    )

    # --------------------------------------------------------------
    # ITERATION 2
    # --------------------------------------------------------------

    it2_generators = diagnose_generators(
        iteration2,
        "Iteration-2",
    )

    it2_loads = diagnose_loads(
        iteration2,
        "Iteration-2",
    )

    it2_buses = diagnose_buses(
        iteration2,
        "Iteration-2",
    )

    it2_controls = diagnose_controls(
        iteration2,
        "Iteration-2",
    )

    it2_connectivity = diagnose_connectivity(
        iteration2,
        "Iteration-2",
    )

    it2_topology = diagnose_topology(
        iteration2,
        "Iteration-2",
    )

    # --------------------------------------------------------------
    # AC TEST
    # --------------------------------------------------------------

    opt_ac = run_ac_test(
        optimized,
        "Optimized",
    )

    it2_ac = run_ac_test(
        iteration2,
        "Iteration-2",
    )

    # --------------------------------------------------------------
    # INTERPRET
    # --------------------------------------------------------------

    interpret(
        opt_generators,
        opt_loads,
        opt_controls,
    )

    # --------------------------------------------------------------
    # SAVE GENERATOR DATA
    # --------------------------------------------------------------

    generator_df = pd.concat(
        [
            opt_generators,
            it2_generators,
        ],
        ignore_index=True,
    )

    generator_df.to_csv(
        OUTPUT_GENERATORS,
        index=False,
    )

    print()
    print(
        f"OK: Generator scenario data saved."
    )
    print(OUTPUT_GENERATORS)

    # --------------------------------------------------------------
    # SAVE LOAD DATA
    # --------------------------------------------------------------

    load_df = pd.concat(
        [
            opt_loads,
            it2_loads,
        ],
        ignore_index=True,
    )

    load_df.to_csv(
        OUTPUT_LOADS,
        index=False,
    )

    print(
        f"OK: Load scenario data saved."
    )
    print(OUTPUT_LOADS)

    # --------------------------------------------------------------
    # SAVE BUS DATA
    # --------------------------------------------------------------

    bus_df = pd.concat(
        [
            opt_buses,
            it2_buses,
        ],
        ignore_index=True,
    )

    bus_df.to_csv(
        OUTPUT_BUSES,
        index=False,
    )

    print(
        f"OK: Bus diagnostic saved."
    )
    print(OUTPUT_BUSES)

    # --------------------------------------------------------------
    # SAVE TOPOLOGY
    # --------------------------------------------------------------

    topology_df = pd.concat(
        [
            opt_topology,
            it2_topology,
        ],
        ignore_index=True,
    )

    topology_df.to_csv(
        OUTPUT_TOPOLOGY,
        index=False,
    )

    print(
        f"OK: Topology diagnostic saved."
    )
    print(OUTPUT_TOPOLOGY)

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    summary_rows = [
        {
            "network": "optimized",
            "scenario": SCENARIO,
            "generators": len(optimized.generators),
            "loads": len(optimized.loads),
            "generator_finite_p_set": int(
                opt_generators["p_set_finite"].sum()
            ),
            "generator_nonzero_p_set": int(
                opt_generators["p_set_nonzero"].sum()
            ),
            "load_finite_p_set": int(
                opt_loads["p_set_finite"].sum()
            ),
            "load_nonzero_p_set": int(
                opt_loads["p_set_nonzero"].sum()
            ),
            "ac_status": opt_ac["status"],
            "ac_max_loading_pct": opt_ac[
                "max_loading_pct"
            ],
            "ac_overloaded_lines": opt_ac[
                "overloaded_lines"
            ],
        },
        {
            "network": "iteration-2",
            "scenario": SCENARIO,
            "generators": len(iteration2.generators),
            "loads": len(iteration2.loads),
            "generator_finite_p_set": int(
                it2_generators["p_set_finite"].sum()
            ),
            "generator_nonzero_p_set": int(
                it2_generators["p_set_nonzero"].sum()
            ),
            "load_finite_p_set": int(
                it2_loads["p_set_finite"].sum()
            ),
            "load_nonzero_p_set": int(
                it2_loads["p_set_nonzero"].sum()
            ),
            "ac_status": it2_ac["status"],
            "ac_max_loading_pct": it2_ac[
                "max_loading_pct"
            ],
            "ac_overloaded_lines": it2_ac[
                "overloaded_lines"
            ],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    header(
        "S2 SCENARIO DATA DIAGNOSTIC COMPLETE"
    )

    print()
    print("SUMMARY")
    print(
        summary_df[
            [
                "network",
                "generator_finite_p_set",
                "generator_nonzero_p_set",
                "load_finite_p_set",
                "load_nonzero_p_set",
                "ac_status",
                "ac_max_loading_pct",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_GENERATORS}")
    print(f"  {OUTPUT_LOADS}")
    print(f"  {OUTPUT_BUSES}")
    print(f"  {OUTPUT_TOPOLOGY}")

    print()
    print("NEXT:")
    print(
        "If S2 generator/load dispatch is zero or NaN, "
        "fix the scenario construction/data-loading stage "
        "before attempting another reinforcement."
    )

    print(
        "If S2 has valid non-zero dispatch but still "
        "fails, the next step is a controlled AC "
        "solver/topology investigation."
    )


if __name__ == "__main__":
    main()