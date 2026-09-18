# ==================================================================================================
# S4.5E — TOPOLOGY + NUMERICAL CONDITIONING + CONVERGED-SOLUTION VALIDATION
# ==================================================================================================
#
# PURPOSE
# -------
# Diagnose the baseline AC power-flow formulation after S4.5D showed that:
#
#   - Raw formulation does NOT converge
#   - Q=0 alone does NOT converge
#   - Explicit slack alone does NOT converge
#   - Q=0 + explicit slack + distributed slack DOES converge
#
# This stage validates the converged formulation directly and investigates:
#
#   1. Network connected components
#   2. AC topology
#   3. Near-zero line impedances
#   4. Extreme R/X ratios
#   5. Transformer parameters
#   6. Voltage magnitude results
#   7. Voltage angle results
#   8. Branch loading
#   9. Actual nodal power balance
#  10. Numerical validity of the converged solution
#
# IMPORTANT
# ---------
# Source network is READ-ONLY.
# No .nc network file is modified.
# All changes occur on an in-memory copy only.
#
# ==================================================================================================

from pathlib import Path
import warnings

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

OUTPUT_CSV = Path(
    r"data\processed\s4_5e_topology_numerical_diagnostic.csv"
)

OUTPUT_BUS_CSV = Path(
    r"data\processed\s4_5e_bus_validation.csv"
)

OUTPUT_LINE_CSV = Path(
    r"data\processed\s4_5e_line_validation.csv"
)

OUTPUT_TRAFO_CSV = Path(
    r"data\processed\s4_5e_transformer_validation.csv"
)


# Numerical sanity limits
V_MIN_SANITY = 0.50
V_MAX_SANITY = 1.50

ANGLE_MIN_SANITY = -20.0
ANGLE_MAX_SANITY = 20.0

NEAR_ZERO_X = 1e-8
NEAR_ZERO_R = 1e-8

MAX_ACCEPTABLE_LOADING = 100.0

PF_TOLERANCE = 1e-6


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.5E — TOPOLOGY + NUMERICAL CONDITIONING + CONVERGED-SOLUTION VALIDATION")
print("=" * 100)

print()
print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print("PF       : AC nonlinear")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Reactive : temporarily set to zero in memory")
print("Slack    : explicit + distributed")
print("Source   : READ-ONLY")

print()
print("=" * 100)
print("PURPOSE")
print("=" * 100)

print(
    """
S4.5D found that the formulation

    Q = 0
    + explicit slack
    + distributed slack

converged successfully.

This stage validates that solution directly and investigates
topology and numerical-conditioning issues.

No reinforcement is applied.
No source network file is modified.
"""
)


# ==================================================================================================
# LOAD SOURCE NETWORK
# ==================================================================================================

print("=" * 100)
print("LOADING SOURCE NETWORK")
print("=" * 100)

n = pypsa.Network(str(NETWORK_PATH))

print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")


# ==================================================================================================
# SNAPSHOT ISOLATION
# ==================================================================================================

print()
print("=" * 100)
print("ISOLATING TARGET SNAPSHOT")
print("=" * 100)

if SNAPSHOT not in n.snapshots:
    raise RuntimeError(
        f"Snapshot '{SNAPSHOT}' not found. "
        f"Available snapshots: {list(n.snapshots)}"
    )

n.set_snapshots([SNAPSHOT])

print("Active snapshot:")
print(f"  {SNAPSHOT}")


# ==================================================================================================
# ORIGINAL TIME-DEPENDENT OPERATING POINT
# ==================================================================================================

print()
print("=" * 100)
print("ORIGINAL OPERATING POINT")
print("=" * 100)

if hasattr(n.generators_t, "p_set"):
    gen_p = n.generators_t.p_set.loc[SNAPSHOT]
else:
    gen_p = pd.Series(index=n.generators.index, dtype=float)

if hasattr(n.loads_t, "p_set"):
    load_p = n.loads_t.p_set.loc[SNAPSHOT]
else:
    load_p = pd.Series(index=n.loads.index, dtype=float)

gen_p = pd.to_numeric(gen_p, errors="coerce").fillna(0.0)
load_p = pd.to_numeric(load_p, errors="coerce").fillna(0.0)

print(f"Generator P set : {gen_p.sum():.6f} MW")
print(f"Load P set      : {load_p.sum():.6f} MW")
print(f"Generation-load : {gen_p.sum() - load_p.sum():.6f} MW")


# ==================================================================================================
# APPLY TEMPORARY REPAIR FORMULATION
# ==================================================================================================

print()
print("=" * 100)
print("APPLYING TEMPORARY PF REPAIR FORMULATION")
print("=" * 100)

print()
print("Generator reactive setpoints -> 0 Mvar")
print("Load reactive setpoints      -> 0 Mvar")
print("Explicit slack               -> enabled")
print("Distributed slack            -> enabled")

# --------------------------------------------------------------------------------
# Set Q values in the time-dependent tables.
# --------------------------------------------------------------------------------

if len(n.generators) > 0:
    n.generators_t.q_set.loc[SNAPSHOT, :] = 0.0

if len(n.loads) > 0:
    n.loads_t.q_set.loc[SNAPSHOT, :] = 0.0


# ==================================================================================================
# ENSURE VALID GENERATOR DISPATCH
# ==================================================================================================

print()
print("=" * 100)
print("GENERATOR DISPATCH CHECK")
print("=" * 100)

gen_dispatch = n.generators_t.p_set.loc[SNAPSHOT].copy()

gen_dispatch = pd.to_numeric(
    gen_dispatch,
    errors="coerce"
).fillna(0.0)

positive_generators = gen_dispatch[
    gen_dispatch > 1e-9
].index.tolist()

print(f"Positive-dispatch generators : {len(positive_generators)}")

if len(positive_generators) == 0:
    raise RuntimeError(
        "No positive-dispatch generator exists at the target snapshot. "
        "Cannot establish a meaningful slack."
    )

# Select the largest positive-dispatch generator as slack candidate.
slack_generator = gen_dispatch.idxmax()

print(f"Selected slack generator      : {slack_generator}")
print(f"Slack generator P             : {gen_dispatch.loc[slack_generator]:.6f} MW")


# ==================================================================================================
# CONFIGURE SLACK
# ==================================================================================================

print()
print("=" * 100)
print("CONFIGURING SLACK")
print("=" * 100)

# Reset controls.
n.generators.control = "PQ"

# Assign explicit slack generator.
n.generators.at[slack_generator, "control"] = "Slack"

print("Explicit slack generator:")
print(f"  {slack_generator}")

# Distributed slack.
n.sub_networks["carrier"] = n.sub_networks["carrier"] if "carrier" in n.sub_networks else None

distributed_slack_enabled = True


# ==================================================================================================
# TOPOLOGY ANALYSIS
# ==================================================================================================

print()
print("=" * 100)
print("TOPOLOGY / CONNECTIVITY ANALYSIS")
print("=" * 100)

topology_rows = []

# --------------------------------------------------------------------------------
# AC network graph using buses + lines + transformers.
# --------------------------------------------------------------------------------

bus_names = list(n.buses.index)

adjacency = {
    bus: set()
    for bus in bus_names
}

# Lines
for line_name, row in n.lines.iterrows():

    b0 = row["bus0"]
    b1 = row["bus1"]

    if b0 in adjacency and b1 in adjacency:
        adjacency[b0].add(b1)
        adjacency[b1].add(b0)

# Transformers
for trafo_name, row in n.transformers.iterrows():

    b0 = row["bus0"]
    b1 = row["bus1"]

    if b0 in adjacency and b1 in adjacency:
        adjacency[b0].add(b1)
        adjacency[b1].add(b0)


# --------------------------------------------------------------------------------
# Connected components.
# --------------------------------------------------------------------------------

visited = set()
components = []

for start_bus in bus_names:

    if start_bus in visited:
        continue

    stack = [start_bus]
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

    components.append(component)


print(f"Total AC connected components : {len(components)}")

components_sorted = sorted(
    components,
    key=len,
    reverse=True
)

for i, component in enumerate(components_sorted, start=1):

    print(
        f"Component {i:02d} : "
        f"{len(component)} buses"
    )

    if len(component) <= 10:
        print("  Buses:", component)


for i, component in enumerate(components_sorted, start=1):

    for bus in component:

        topology_rows.append(
            {
                "bus": bus,
                "component": i,
                "component_size": len(component),
                "degree": len(adjacency[bus]),
            }
        )


topology_df = pd.DataFrame(topology_rows)


# ==================================================================================================
# IDENTIFY ISOLATED / WEAKLY CONNECTED BUSES
# ==================================================================================================

print()
print("=" * 100)
print("WEAKLY CONNECTED BUS CHECK")
print("=" * 100)

degree_zero = topology_df[
    topology_df["degree"] == 0
]

degree_one = topology_df[
    topology_df["degree"] == 1
]

print(f"Degree-0 buses : {len(degree_zero)}")
print(f"Degree-1 buses : {len(degree_one)}")

if len(degree_zero) > 0:

    print()
    print("DEGREE-0 BUSES:")
    for bus in degree_zero["bus"]:
        print(f"  {bus}")

if len(degree_one) > 0:

    print()
    print("DEGREE-1 BUSES:")
    for bus in degree_one["bus"]:
        print(f"  {bus}")


# ==================================================================================================
# LINE IMPEDANCE CONDITIONING
# ==================================================================================================

print()
print("=" * 100)
print("LINE IMPEDANCE CONDITIONING")
print("=" * 100)

line_records = []

for line_name, row in n.lines.iterrows():

    r = pd.to_numeric(row.get("r"), errors="coerce")
    x = pd.to_numeric(row.get("x"), errors="coerce")
    b = pd.to_numeric(row.get("b"), errors="coerce")

    record = {
        "line": line_name,
        "bus0": row["bus0"],
        "bus1": row["bus1"],
        "r": r,
        "x": x,
        "b": b,
    }

    if pd.notna(r) and pd.notna(x):

        record["r_abs"] = abs(r)
        record["x_abs"] = abs(x)

        if abs(x) > NEAR_ZERO_X:
            record["r_over_x"] = abs(r / x)
        else:
            record["r_over_x"] = np.inf

        record["near_zero_r"] = abs(r) < NEAR_ZERO_R
        record["near_zero_x"] = abs(x) < NEAR_ZERO_X

    else:

        record["r_abs"] = np.nan
        record["x_abs"] = np.nan
        record["r_over_x"] = np.nan
        record["near_zero_r"] = True
        record["near_zero_x"] = True

    line_records.append(record)


line_df = pd.DataFrame(line_records)


near_zero_lines = line_df[
    (line_df["near_zero_r"]) |
    (line_df["near_zero_x"])
]

extreme_rx = line_df[
    line_df["r_over_x"] > 10
]

negative_x = line_df[
    line_df["x"] < 0
]

print(f"Total lines              : {len(line_df)}")
print(f"Near-zero impedance lines: {len(near_zero_lines)}")
print(f"R/X > 10 lines           : {len(extreme_rx)}")
print(f"Negative-X lines         : {len(negative_x)}")


if len(near_zero_lines) > 0:

    print()
    print("NEAR-ZERO IMPEDANCE LINES:")
    print(
        near_zero_lines[
            ["line", "r", "x", "r_over_x"]
        ].to_string(index=False)
    )


if len(extreme_rx) > 0:

    print()
    print("EXTREME R/X LINES:")
    print(
        extreme_rx[
            ["line", "r", "x", "r_over_x"]
        ].sort_values(
            "r_over_x",
            ascending=False
        ).to_string(index=False)
    )


# ==================================================================================================
# TRANSFORMER CONDITIONING
# ==================================================================================================

print()
print("=" * 100)
print("TRANSFORMER CONDITIONING")
print("=" * 100)

trafo_records = []

for trafo_name, row in n.transformers.iterrows():

    x = pd.to_numeric(
        row.get("x_pu"),
        errors="coerce"
    )

    r = pd.to_numeric(
        row.get("r_pu"),
        errors="coerce"
    )

    s_nom = pd.to_numeric(
        row.get("s_nom"),
        errors="coerce"
    )

    tap = pd.to_numeric(
        row.get("tap_ratio"),
        errors="coerce"
    )

    phase = pd.to_numeric(
        row.get("phase_shift"),
        errors="coerce"
    )

    trafo_records.append(
        {
            "transformer": trafo_name,
            "bus0": row["bus0"],
            "bus1": row["bus1"],
            "s_nom_mva": s_nom,
            "x_pu": x,
            "r_pu": r,
            "tap_ratio": tap,
            "phase_shift": phase,
            "r_over_x": (
                abs(r / x)
                if pd.notna(x) and abs(x) > 1e-12
                else np.inf
            ),
        }
    )


trafo_df = pd.DataFrame(trafo_records)

print(
    trafo_df.to_string(index=False)
)


# ==================================================================================================
# INITIAL VOLTAGE CONDITIONS
# ==================================================================================================

print()
print("=" * 100)
print("INITIAL VOLTAGE CONDITIONS")
print("=" * 100)

initial_v = n.buses["v_mag_pu_set"]

initial_v = pd.to_numeric(
    initial_v,
    errors="coerce"
)

print(
    f"Initial voltage min : {initial_v.min():.6f} pu"
)

print(
    f"Initial voltage max : {initial_v.max():.6f} pu"
)

print(
    f"Initial voltage NaNs: {initial_v.isna().sum()}"
)


# ==================================================================================================
# RUN AC POWER FLOW
# ==================================================================================================

print()
print("=" * 100)
print("RUNNING AC NONLINEAR POWER FLOW")
print("=" * 100)

print()
print("Configuration:")
print("  Generator Q = 0")
print("  Load Q      = 0")
print("  Explicit slack enabled")
print("  Distributed slack enabled")
print()

with warnings.catch_warnings():

    warnings.simplefilter("always")

    try:

        pf_result = n.pf(
            snapshots=[SNAPSHOT],
            use_seed=True,
            distribute_slack=True
        )

    except TypeError:

        # Compatibility fallback for PyPSA versions
        # where use_seed is unavailable.
        pf_result = n.pf(
            snapshots=[SNAPSHOT],
            distribute_slack=True
        )

    except Exception as exc:

        print()
        print("POWER FLOW EXCEPTION:")
        print(str(exc))

        raise


# ==================================================================================================
# EXTRACT PF RESULT
# ==================================================================================================

print()
print("=" * 100)
print("RAW POWER-FLOW RESULT")
print("=" * 100)

print()

print(
    "pf_result:"
)

print(pf_result)


# ==================================================================================================
# DIRECT VOLTAGE RESULT INSPECTION
# ==================================================================================================

print()
print("=" * 100)
print("DIRECT BUS VOLTAGE RESULT INSPECTION")
print("=" * 100)

if SNAPSHOT not in n.buses_t.v_mag_pu.index:

    print(
        "ERROR: Snapshot not found in buses_t.v_mag_pu."
    )

    voltage_available = False

else:

    voltage_available = True

    voltage = pd.to_numeric(
        n.buses_t.v_mag_pu.loc[SNAPSHOT],
        errors="coerce"
    )

    finite_voltage = voltage[
        np.isfinite(voltage)
    ]

    print(
        f"Voltage entries       : {len(voltage)}"
    )

    print(
        f"Finite voltage entries: {len(finite_voltage)}"
    )

    print(
        f"NaN voltage entries   : {voltage.isna().sum()}"
    )

    print(
        f"Inf voltage entries   : "
        f"{np.isinf(voltage.to_numpy()).sum()}"
    )

    if len(finite_voltage) > 0:

        print(
            f"Minimum voltage       : "
            f"{finite_voltage.min():.6f} pu"
        )

        print(
            f"Maximum voltage       : "
            f"{finite_voltage.max():.6f} pu"
        )

        print()
        print("VOLTAGE DISTRIBUTION:")
        print(
            finite_voltage.describe().to_string()
        )


# ==================================================================================================
# VOLTAGE VALIDATION
# ==================================================================================================

print()
print("=" * 100)
print("VOLTAGE PHYSICAL VALIDATION")
print("=" * 100)

if not voltage_available:

    voltage_valid = False

else:

    finite_voltage = voltage[
        np.isfinite(voltage)
    ]

    voltage_valid = (
        len(finite_voltage) == len(voltage)
        and
        len(finite_voltage) > 0
        and
        finite_voltage.min() >= V_MIN_SANITY
        and
        finite_voltage.max() <= V_MAX_SANITY
    )

    print(
        f"Voltage finite       : "
        f"{len(finite_voltage) == len(voltage)}"
    )

    print(
        f"Voltage range valid  : "
        f"{voltage_valid}"
    )

    if len(finite_voltage) > 0:

        print(
            f"Voltage minimum      : "
            f"{finite_voltage.min():.6f} pu"
        )

        print(
            f"Voltage maximum      : "
            f"{finite_voltage.max():.6f} pu"
        )


# ==================================================================================================
# DIRECT ANGLE RESULT INSPECTION
# ==================================================================================================

print()
print("=" * 100)
print("BUS ANGLE RESULT INSPECTION")
print("=" * 100)

if SNAPSHOT in n.buses_t.v_ang.index:

    angle = pd.to_numeric(
        n.buses_t.v_ang.loc[SNAPSHOT],
        errors="coerce"
    )

    finite_angle = angle[
        np.isfinite(angle)
    ]

    print(
        f"Angle entries        : {len(angle)}"
    )

    print(
        f"Finite angle entries : {len(finite_angle)}"
    )

    print(
        f"NaN angle entries    : {angle.isna().sum()}"
    )

    if len(finite_angle) > 0:

        print(
            f"Minimum angle       : {finite_angle.min():.6f} rad"
        )

        print(
            f"Maximum angle       : {finite_angle.max():.6f} rad"
        )

else:

    angle = pd.Series(
        index=n.buses.index,
        dtype=float
    )

    print(
        "v_ang result unavailable."
    )


# ==================================================================================================
# BUS VALIDATION TABLE
# ==================================================================================================

bus_validation = pd.DataFrame(
    index=n.buses.index
)

bus_validation.index.name = "bus"

bus_validation["v_nom_kv"] = n.buses["v_nom"]
bus_validation["carrier"] = n.buses["carrier"]

bus_validation["component"] = (
    topology_df
    .set_index("bus")["component"]
)

bus_validation["component_size"] = (
    topology_df
    .set_index("bus")["component_size"]
)

bus_validation["degree"] = (
    topology_df
    .set_index("bus")["degree"]
)

bus_validation["v_initial_pu"] = initial_v

if voltage_available:
    bus_validation["v_final_pu"] = voltage
else:
    bus_validation["v_final_pu"] = np.nan

bus_validation["angle_final_rad"] = angle

if voltage_available:

    bus_validation["voltage_finite"] = (
        np.isfinite(
            bus_validation["v_final_pu"]
        )
    )

    bus_validation["voltage_in_sanity_range"] = (
        bus_validation["v_final_pu"].between(
            V_MIN_SANITY,
            V_MAX_SANITY
        )
    )

else:

    bus_validation["voltage_finite"] = False
    bus_validation["voltage_in_sanity_range"] = False


# ==================================================================================================
# BRANCH LOADING VALIDATION
# ==================================================================================================

print()
print("=" * 100)
print("LINE LOADING VALIDATION")
print("=" * 100)

line_loading = pd.Series(
    index=n.lines.index,
    dtype=float
)

if (
    hasattr(n.lines_t, "p0")
    and SNAPSHOT in n.lines_t.p0.index
):

    p0 = pd.to_numeric(
        n.lines_t.p0.loc[SNAPSHOT],
        errors="coerce"
    )

    p1 = pd.to_numeric(
        n.lines_t.p1.loc[SNAPSHOT],
        errors="coerce"
    )

    s0 = np.sqrt(
        n.lines_t.p0.loc[SNAPSHOT].fillna(0.0) ** 2
        +
        n.lines_t.q0.loc[SNAPSHOT].fillna(0.0) ** 2
    )

    s1 = np.sqrt(
        n.lines_t.p1.loc[SNAPSHOT].fillna(0.0) ** 2
        +
        n.lines_t.q1.loc[SNAPSHOT].fillna(0.0) ** 2
    )

    s_max = pd.concat(
        [s0, s1],
        axis=1
    ).max(axis=1)

    line_s_nom = pd.to_numeric(
        n.lines["s_nom"],
        errors="coerce"
    )

    line_loading = (
        s_max
        / line_s_nom
        * 100.0
    )

    finite_loading = line_loading[
        np.isfinite(line_loading)
    ]

    print(
        f"Finite line-loading entries : "
        f"{len(finite_loading)}"
    )

    if len(finite_loading) > 0:

        print(
            f"Maximum line loading       : "
            f"{finite_loading.max():.6f} %"
        )

        print(
            f"Overloaded lines (>100%)    : "
            f"{(finite_loading > 100).sum()}"
        )

else:

    print(
        "Line flow results unavailable."
    )


# ==================================================================================================
# TRANSFORMER LOADING
# ==================================================================================================

print()
print("=" * 100)
print("TRANSFORMER LOADING VALIDATION")
print("=" * 100)

trafo_loading = pd.Series(
    index=n.transformers.index,
    dtype=float
)

if (
    hasattr(n.transformers_t, "p0")
    and SNAPSHOT in n.transformers_t.p0.index
):

    tp0 = n.transformers_t.p0.loc[SNAPSHOT]
    tq0 = n.transformers_t.q0.loc[SNAPSHOT]

    tp1 = n.transformers_t.p1.loc[SNAPSHOT]
    tq1 = n.transformers_t.q1.loc[SNAPSHOT]

    ts0 = np.sqrt(
        tp0.fillna(0.0) ** 2
        +
        tq0.fillna(0.0) ** 2
    )

    ts1 = np.sqrt(
        tp1.fillna(0.0) ** 2
        +
        tq1.fillna(0.0) ** 2
    )

    ts_max = pd.concat(
        [ts0, ts1],
        axis=1
    ).max(axis=1)

    trafo_s_nom = pd.to_numeric(
        n.transformers["s_nom"],
        errors="coerce"
    )

    trafo_loading = (
        ts_max
        / trafo_s_nom
        * 100.0
    )

    finite_trafo_loading = trafo_loading[
        np.isfinite(trafo_loading)
    ]

    print(
        f"Finite transformer-loading entries : "
        f"{len(finite_trafo_loading)}"
    )

    if len(finite_trafo_loading) > 0:

        print(
            f"Maximum transformer loading       : "
            f"{finite_trafo_loading.max():.6f} %"
        )

        print(
            f"Overloaded transformers (>100%)   : "
            f"{(finite_trafo_loading > 100).sum()}"
        )

else:

    print(
        "Transformer flow results unavailable."
    )


# ==================================================================================================
# ACTUAL PF POWER BALANCE
# ==================================================================================================

print()
print("=" * 100)
print("POST-PF POWER BALANCE")
print("=" * 100)

if hasattr(n.generators_t, "p"):

    solved_generation = pd.to_numeric(
        n.generators_t.p.loc[SNAPSHOT],
        errors="coerce"
    ).fillna(0.0).sum()

else:

    solved_generation = np.nan


if hasattr(n.loads_t, "p"):

    solved_load = pd.to_numeric(
        n.loads_t.p.loc[SNAPSHOT],
        errors="coerce"
    ).fillna(0.0).sum()

else:

    solved_load = np.nan


print(
    f"Solved generation : {solved_generation:.6f} MW"
)

print(
    f"Solved load       : {solved_load:.6f} MW"
)

if np.isfinite(solved_generation) and np.isfinite(solved_load):

    print(
        f"Generation-load   : "
        f"{solved_generation - solved_load:.6f} MW"
    )


# ==================================================================================================
# GENERATOR PF RESULT
# ==================================================================================================

print()
print("=" * 100)
print("SOLVED GENERATOR OUTPUT")
print("=" * 100)

generator_result = pd.DataFrame(index=n.generators.index)

generator_result["bus"] = n.generators["bus"]
generator_result["control"] = n.generators["control"]
generator_result["p_set_mw"] = n.generators_t.p_set.loc[SNAPSHOT]

if hasattr(n.generators_t, "p"):
    generator_result["p_solved_mw"] = n.generators_t.p.loc[SNAPSHOT]

if hasattr(n.generators_t, "q"):
    generator_result["q_solved_mvar"] = n.generators_t.q.loc[SNAPSHOT]

print(
    generator_result.to_string()
)


# ==================================================================================================
# LOAD PF RESULT
# ==================================================================================================

print()
print("=" * 100)
print("SOLVED LOAD OUTPUT")
print("=" * 100)

load_result = pd.DataFrame(index=n.loads.index)

load_result["bus"] = n.loads["bus"]
load_result["p_set_mw"] = n.loads_t.p_set.loc[SNAPSHOT]

if hasattr(n.loads_t, "p"):
    load_result["p_solved_mw"] = n.loads_t.p.loc[SNAPSHOT]

if hasattr(n.loads_t, "q"):
    load_result["q_solved_mvar"] = n.loads_t.q.loc[SNAPSHOT]

print(
    load_result.to_string()
)


# ==================================================================================================
# FINAL VALIDITY DECISION
# ==================================================================================================

print()
print("=" * 100)
print("FINAL PHYSICAL VALIDITY ASSESSMENT")
print("=" * 100)

pf_converged = False

if isinstance(pf_result, pd.DataFrame):

    if "S2_PEAK_DEMAND" in pf_result.index:

        row = pf_result.loc["S2_PEAK_DEMAND"]

        if "converged" in row.index:

            pf_converged = bool(
                row["converged"]
            )

elif isinstance(pf_result, pd.Series):

    if "converged" in pf_result.index:

        pf_converged = bool(
            pf_result["converged"]
        )


# If PyPSA did not expose a straightforward convergence flag,
# infer it conservatively from the actual voltage and flow results.

if not pf_converged:

    if voltage_available:

        finite_v = voltage[
            np.isfinite(voltage)
        ]

        if len(finite_v) == len(n.buses):

            if (
                finite_v.min() >= V_MIN_SANITY
                and
                finite_v.max() <= V_MAX_SANITY
            ):

                # Do not automatically mark true unless PF error
                # is also small.
                try:

                    if isinstance(pf_result, pd.DataFrame):

                        row = pf_result.loc[SNAPSHOT]

                        if "error" in row.index:

                            pf_error = float(
                                row["error"]
                            )

                        else:

                            pf_error = np.nan

                    else:

                        pf_error = np.nan

                except Exception:

                    pf_error = np.nan

                if (
                    np.isfinite(pf_error)
                    and
                    pf_error < PF_TOLERANCE
                ):
                    pf_converged = True


# Final decision
valid_physical_solution = (
    pf_converged
    and
    voltage_valid
)


print(
    f"PF converged               : "
    f"{pf_converged}"
)

print(
    f"Voltage results available  : "
    f"{voltage_available}"
)

print(
    f"Voltage physically valid   : "
    f"{voltage_valid}"
)

print(
    f"VALID PHYSICAL SOLUTION    : "
    f"{valid_physical_solution}"
)


# ==================================================================================================
# CRITICAL INTERPRETATION
# ==================================================================================================

print()
print("=" * 100)
print("S4.5E INTERPRETATION")
print("=" * 100)

if valid_physical_solution:

    print(
        """
SUCCESS:

The repaired AC formulation has produced a numerically valid
physical solution.

This means we can now proceed to voltage-bottleneck analysis.

IMPORTANT:
This is NOT yet the P3 + residual reinforcement case.

The validated formulation is:

    Q = 0
    + explicit slack
    + distributed slack

The next stage should determine whether the original reactive
operating point can be restored while preserving convergence,
rather than immediately interpreting the Q=0 solution as the
final physical operating condition.
"""
    )

elif pf_converged:

    print(
        """
PARTIAL SUCCESS:

The AC solver reports convergence, but the resulting voltage
state failed the physical sanity checks.

Therefore the solution must NOT yet be used for voltage-bottleneck
analysis.

Investigate the direct buses_t.v_mag_pu and buses_t.v_ang results.
"""
    )

else:

    print(
        """
NO VALID SOLUTION:

Even the repaired formulation did not produce a valid physical
AC solution.

The next step must remain numerical/topological diagnosis.
"""
    )


# ==================================================================================================
# SAVE TABLES
# ==================================================================================================

print()
print("=" * 100)
print("SAVING DIAGNOSTIC RESULTS")
print("=" * 100)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary = pd.DataFrame(
    [
        {
            "case": "Q_ZERO_EXPLICIT_SLACK_DISTRIBUTED",
            "snapshot": SNAPSHOT,
            "pf_converged": pf_converged,
            "voltage_available": voltage_available,
            "voltage_valid": voltage_valid,
            "valid_physical_solution": valid_physical_solution,
            "components": len(components_sorted),
            "degree_zero_buses": len(degree_zero),
            "degree_one_buses": len(degree_one),
            "near_zero_impedance_lines": len(near_zero_lines),
            "extreme_rx_lines": len(extreme_rx),
            "negative_x_lines": len(negative_x),
            "max_line_loading_pct": (
                finite_loading.max()
                if "finite_loading" in locals()
                and len(finite_loading) > 0
                else np.nan
            ),
            "overloaded_lines": (
                int(
                    (finite_loading > 100).sum()
                )
                if "finite_loading" in locals()
                and len(finite_loading) > 0
                else np.nan
            ),
            "max_transformer_loading_pct": (
                finite_trafo_loading.max()
                if "finite_trafo_loading" in locals()
                and len(finite_trafo_loading) > 0
                else np.nan
            ),
            "overloaded_transformers": (
                int(
                    (finite_trafo_loading > 100).sum()
                )
                if "finite_trafo_loading" in locals()
                and len(finite_trafo_loading) > 0
                else np.nan
            ),
            "solved_generation_mw": solved_generation,
            "solved_load_mw": solved_load,
            "solved_generation_minus_load_mw": (
                solved_generation - solved_load
                if np.isfinite(solved_generation)
                and np.isfinite(solved_load)
                else np.nan
            ),
        }
    ]
)

summary.to_csv(
    OUTPUT_CSV,
    index=False
)

bus_validation.to_csv(
    OUTPUT_BUS_CSV
)

line_df["loading_pct"] = line_loading

line_df.to_csv(
    OUTPUT_LINE_CSV,
    index=False
)

trafo_df["loading_pct"] = trafo_loading

trafo_df.to_csv(
    OUTPUT_TRAFO_CSV,
    index=False
)


print()
print("Results saved:")
print(f"  {OUTPUT_CSV}")
print(f"  {OUTPUT_BUS_CSV}")
print(f"  {OUTPUT_LINE_CSV}")
print(f"  {OUTPUT_TRAFO_CSV}")


# ==================================================================================================
# FINAL
# ==================================================================================================

print()
print("=" * 100)
print("S4.5E COMPLETE")
print("=" * 100)

print()
print("SOURCE NETWORK MODIFIED : NO")
print("REINFORCEMENTS APPLIED  : NO")
print("REACTIVE DEVICES ADDED  : NO")
print("IN-MEMORY PF REPAIR     : Q=0 + EXPLICIT SLACK + DISTRIBUTED SLACK")

print()
print("=" * 100)