import pypsa
import numpy as np
import pandas as pd

# ==============================================================
# S2 AC BRANCH / TOPOLOGY / ELECTRICAL PARAMETER DIAGNOSTIC
# ==============================================================

NETWORK_PATH = r"data\processed\eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 78)
print("S2 AC BRANCH / TOPOLOGY / ELECTRICAL PARAMETER DIAGNOSTIC")
print("=" * 78)
print()
print(f"Network : {NETWORK_PATH}")
print(f"Snapshot: {SNAPSHOT}")

# --------------------------------------------------------------
# LOAD NETWORK — READ ONLY
# --------------------------------------------------------------

n = pypsa.Network(NETWORK_PATH)

print()
print("NETWORK")
print("-" * 78)
print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")
print(f"Links        : {len(n.links)}")

# ==============================================================
# 1. BUS AUDIT
# ==============================================================

print()
print("=" * 78)
print("1. BUS AUDIT")
print("=" * 78)

print()
print("BUS VOLTAGE LEVELS")
print("-" * 78)
print(n.buses["v_nom"].value_counts(dropna=False).sort_index())

print()
print("INVALID BUS PARAMETERS")
print("-" * 78)

invalid_bus_vnom = n.buses[
    (~np.isfinite(n.buses["v_nom"])) |
    (n.buses["v_nom"] <= 0)
]

print(f"Invalid v_nom buses: {len(invalid_bus_vnom)}")

if len(invalid_bus_vnom):
    print(invalid_bus_vnom[["v_nom"]])

# --------------------------------------------------------------
# Bus control information
# --------------------------------------------------------------

print()
print("BUS COUNT BY SUBNETWORK")
print("-" * 78)

try:
    n.determine_network_topology()

    if hasattr(n.buses, "sub_network"):
        print(n.buses["sub_network"].value_counts(dropna=False))
except Exception as e:
    print(f"Topology determination failed: {type(e).__name__}: {e}")

# ==============================================================
# 2. LINE PARAMETER AUDIT
# ==============================================================

print()
print("=" * 78)
print("2. LINE PARAMETER AUDIT")
print("=" * 78)

if len(n.lines) > 0:

    lines = n.lines.copy()

    required = ["bus0", "bus1", "r", "x", "b", "s_nom"]

    print()
    print("LINE PARAMETER RANGE")
    print("-" * 78)

    for col in ["r", "x", "b", "s_nom"]:
        if col in lines.columns:
            vals = pd.to_numeric(lines[col], errors="coerce")

            print(
                f"{col:8s}: "
                f"min={vals.min():.12g} "
                f"max={vals.max():.12g} "
                f"mean={vals.mean():.12g}"
            )

    # ----------------------------------------------------------
    # NaN / infinite
    # ----------------------------------------------------------

    print()
    print("NON-FINITE LINE PARAMETERS")
    print("-" * 78)

    for col in ["r", "x", "b", "s_nom"]:
        vals = pd.to_numeric(lines[col], errors="coerce")
        bad = ~np.isfinite(vals)

        print(f"{col:8s}: {bad.sum()}")

        if bad.any():
            print(lines.loc[bad, [col, "bus0", "bus1"]])

    # ----------------------------------------------------------
    # Zero / negative
    # ----------------------------------------------------------

    print()
    print("ZERO / NEGATIVE LINE PARAMETERS")
    print("-" * 78)

    for col in ["r", "x", "s_nom"]:

        vals = pd.to_numeric(lines[col], errors="coerce")

        bad = vals <= 0

        print(f"{col:8s}: {bad.sum()}")

        if bad.any():
            print(
                lines.loc[
                    bad,
                    ["bus0", "bus1", col]
                ].to_string()
            )

    # ----------------------------------------------------------
    # Self loops
    # ----------------------------------------------------------

    print()
    print("SELF-LOOP LINES")
    print("-" * 78)

    self_loops = lines[
        lines["bus0"] == lines["bus1"]
    ]

    print(f"Self-loop count: {len(self_loops)}")

    if len(self_loops):
        print(
            self_loops[
                ["bus0", "bus1", "r", "x", "s_nom"]
            ].to_string()
        )

    # ----------------------------------------------------------
    # R/X ratio
    # ----------------------------------------------------------

    lines["r_over_x"] = (
        pd.to_numeric(lines["r"], errors="coerce") /
        pd.to_numeric(lines["x"], errors="coerce")
    )

    print()
    print("R/X RATIO")
    print("-" * 78)

    print(lines["r_over_x"].describe())

    extreme_rx = lines[
        (lines["r_over_x"] > 2.0) |
        (lines["r_over_x"] < 0.01)
    ]

    print()
    print(
        f"Extreme R/X lines (<0.01 or >2.0): "
        f"{len(extreme_rx)}"
    )

    if len(extreme_rx):
        print(
            extreme_rx[
                ["bus0", "bus1", "r", "x", "r_over_x", "s_nom"]
            ].sort_values(
                "r_over_x"
            ).to_string()
        )

    # ----------------------------------------------------------
    # Largest X
    # ----------------------------------------------------------

    print()
    print("HIGHEST X LINES")
    print("-" * 78)

    print(
        lines[
            ["bus0", "bus1", "r", "x", "s_nom"]
        ]
        .sort_values("x", ascending=False)
        .head(15)
        .to_string()
    )

    # ----------------------------------------------------------
    # Lowest X
    # ----------------------------------------------------------

    print()
    print("LOWEST X LINES")
    print("-" * 78)

    print(
        lines[
            ["bus0", "bus1", "r", "x", "s_nom"]
        ]
        .sort_values("x")
        .head(15)
        .to_string()
    )

    # ----------------------------------------------------------
    # Duplicate physical branches
    # ----------------------------------------------------------

    print()
    print("DUPLICATE / PARALLEL BRANCH CHECK")
    print("-" * 78)

    lines["_pair"] = lines.apply(
        lambda row: tuple(
            sorted([str(row["bus0"]), str(row["bus1"])])
        ),
        axis=1
    )

    duplicates = lines[
        lines.duplicated("_pair", keep=False)
    ].sort_values("_pair")

    print(
        f"Branches participating in parallel/duplicate "
        f"connections: {len(duplicates)}"
    )

    if len(duplicates):
        print(
            duplicates[
                ["bus0", "bus1", "r", "x", "b", "s_nom"]
            ].to_string()
        )

else:
    print("NO AC LINES FOUND")

# ==============================================================
# 3. PER-UNIT IMPEDANCE AUDIT
# ==============================================================

print()
print("=" * 78)
print("3. PER-UNIT LINE IMPEDANCE AUDIT")
print("=" * 78)

if len(n.lines):

    lines_pu = n.lines.copy()

    v0 = n.buses.loc[
        lines_pu["bus0"],
        "v_nom"
    ].to_numpy(dtype=float)

    v1 = n.buses.loc[
        lines_pu["bus1"],
        "v_nom"
    ].to_numpy(dtype=float)

    # PyPSA branch impedance is based on the network base
    # and nominal voltage of the connected buses.
    #
    # For diagnostic purposes use the arithmetic mean voltage
    # where both ends have the same voltage.
    #
    # If voltage levels differ, flag the branch separately.

    v_base = np.sqrt(v0 * v1)

    z_base = (v_base ** 2) / n.sn_mva

    lines_pu["r_pu_diag"] = (
        lines_pu["r"].to_numpy(dtype=float) / z_base
    )

    lines_pu["x_pu_diag"] = (
        lines_pu["x"].to_numpy(dtype=float) / z_base
    )

    lines_pu["z_pu_diag"] = np.sqrt(
        lines_pu["r_pu_diag"] ** 2 +
        lines_pu["x_pu_diag"] ** 2
    )

    print()
    print(f"Network apparent-power base: {n.sn_mva} MVA")

    print()
    print("DIAGNOSTIC PER-UNIT IMPEDANCE RANGE")
    print("-" * 78)

    for col in [
        "r_pu_diag",
        "x_pu_diag",
        "z_pu_diag"
    ]:
        print(
            f"{col:14s}: "
            f"min={lines_pu[col].min():.8g} "
            f"max={lines_pu[col].max():.8g} "
            f"mean={lines_pu[col].mean():.8g}"
        )

    print()
    print("LARGEST PER-UNIT IMPEDANCES")
    print("-" * 78)

    print(
        lines_pu[
            [
                "bus0",
                "bus1",
                "r",
                "x",
                "r_pu_diag",
                "x_pu_diag",
                "z_pu_diag"
            ]
        ]
        .sort_values(
            "z_pu_diag",
            ascending=False
        )
        .head(20)
        .to_string()
    )

    print()
    print("SMALLEST PER-UNIT IMPEDANCES")
    print("-" * 78)

    print(
        lines_pu[
            [
                "bus0",
                "bus1",
                "r",
                "x",
                "r_pu_diag",
                "x_pu_diag",
                "z_pu_diag"
            ]
        ]
        .sort_values("z_pu_diag")
        .head(20)
        .to_string()
    )

# ==============================================================
# 4. VOLTAGE LEVEL CONSISTENCY
# ==============================================================

print()
print("=" * 78)
print("4. LINE VOLTAGE-LEVEL CONSISTENCY")
print("=" * 78)

if len(n.lines):

    line_voltage = pd.DataFrame(index=n.lines.index)

    line_voltage["bus0"] = n.lines["bus0"]
    line_voltage["bus1"] = n.lines["bus1"]

    line_voltage["v0"] = n.lines["bus0"].map(
        n.buses["v_nom"]
    )

    line_voltage["v1"] = n.lines["bus1"].map(
        n.buses["v_nom"]
    )

    mismatch = line_voltage[
        line_voltage["v0"] != line_voltage["v1"]
    ]

    print(
        f"Lines connecting different nominal voltage levels: "
        f"{len(mismatch)}"
    )

    if len(mismatch):
        print(
            mismatch.to_string()
        )

# ==============================================================
# 5. TRANSFORMER AUDIT
# ==============================================================

print()
print("=" * 78)
print("5. TRANSFORMER AUDIT")
print("=" * 78)

if len(n.transformers):

    trafo = n.transformers.copy()

    print()
    print(
        trafo[
            [
                "bus0",
                "bus1",
                "r",
                "x",
                "s_nom",
                "tap_ratio",
                "phase_shift"
            ]
        ].to_string()
    )

    print()
    print("TRANSFORMER PARAMETER RANGE")
    print("-" * 78)

    for col in [
        "r",
        "x",
        "s_nom",
        "tap_ratio",
        "phase_shift"
    ]:
        vals = pd.to_numeric(
            trafo[col],
            errors="coerce"
        )

        print(
            f"{col:14s}: "
            f"min={vals.min():.8g} "
            f"max={vals.max():.8g}"
        )

    print()
    print("TRANSFORMER VOLTAGE LEVELS")
    print("-" * 78)

    for name, row in trafo.iterrows():

        v0 = n.buses.at[row.bus0, "v_nom"]
        v1 = n.buses.at[row.bus1, "v_nom"]

        print(
            f"{name}"
            f" | {row.bus0}: {v0:g} kV"
            f" -> {row.bus1}: {v1:g} kV"
            f" | r={row.r}"
            f" | x={row.x}"
            f" | s_nom={row.s_nom}"
            f" | tap={row.tap_ratio}"
        )

else:
    print("NO TRANSFORMERS")

# ==============================================================
# 6. CONNECTIVITY AUDIT
# ==============================================================

print()
print("=" * 78)
print("6. NETWORK CONNECTIVITY AUDIT")
print("=" * 78)

try:

    n.determine_network_topology()

    print()
    print("SUBNETWORKS")
    print("-" * 78)

    if hasattr(n, "sub_networks"):
        print(n.sub_networks)

except Exception as e:

    print(
        f"Topology audit failed: "
        f"{type(e).__name__}: {e}"
    )

# --------------------------------------------------------------
# Manual graph connectivity
# --------------------------------------------------------------

print()
print("MANUAL AC GRAPH CONNECTIVITY")
print("-" * 78)

# Build graph using lines + transformers.
#
# This avoids relying exclusively on PyPSA's internal
# subnetwork representation.

graph = {
    bus: set()
    for bus in n.buses.index
}

for name, row in n.lines.iterrows():

    b0 = row.bus0
    b1 = row.bus1

    if b0 in graph and b1 in graph:

        graph[b0].add(b1)
        graph[b1].add(b0)

for name, row in n.transformers.iterrows():

    b0 = row.bus0
    b1 = row.bus1

    if b0 in graph and b1 in graph:

        graph[b0].add(b1)
        graph[b1].add(b0)

components = []

remaining = set(graph.keys())

while remaining:

    start = next(iter(remaining))

    stack = [start]
    component = set()

    while stack:

        bus = stack.pop()

        if bus in component:
            continue

        component.add(bus)

        for neighbour in graph[bus]:

            if neighbour not in component:
                stack.append(neighbour)

    components.append(component)

    remaining -= component

components = sorted(
    components,
    key=len,
    reverse=True
)

print(
    f"Connected AC components: {len(components)}"
)

for i, component in enumerate(components):

    voltage_levels = sorted(
        set(
            n.buses.loc[
                list(component),
                "v_nom"
            ].dropna().tolist()
        )
    )

    print(
        f"Component {i}: "
        f"{len(component)} buses "
        f"| voltage levels={voltage_levels}"
    )

    if len(component) <= 10:
        print(
            "  ",
            sorted(component)
        )

# ==============================================================
# 7. LOAD CONNECTIVITY
# ==============================================================

print()
print("=" * 78)
print("7. LOAD CONNECTIVITY")
print("=" * 78)

for name, row in n.loads.iterrows():

    bus = row.bus

    neighbours = graph.get(bus, set())

    print(
        f"{name}"
        f" | bus={bus}"
        f" | v_nom={n.buses.at[bus, 'v_nom']}"
        f" | graph_degree={len(neighbours)}"
    )

# ==============================================================
# 8. GENERATOR CONNECTIVITY
# ==============================================================

print()
print("=" * 78)
print("8. GENERATOR CONNECTIVITY")
print("=" * 78)

for name, row in n.generators.iterrows():

    bus = row.bus

    neighbours = graph.get(bus, set())

    p = (
        n.generators_t.p.loc[
            SNAPSHOT,
            name
        ]
        if SNAPSHOT in n.generators_t.p.index
        else np.nan
    )

    print(
        f"{name}"
        f" | bus={bus}"
        f" | v_nom={n.buses.at[bus, 'v_nom']}"
        f" | degree={len(neighbours)}"
        f" | P={p}"
    )

# ==============================================================
# 9. BUS DEGREE AUDIT
# ==============================================================

print()
print("=" * 78)
print("9. LOW-DEGREE / ISOLATED BUS AUDIT")
print("=" * 78)

degree_records = []

for bus in graph:

    degree_records.append(
        {
            "bus": bus,
            "degree": len(graph[bus]),
            "v_nom": n.buses.at[bus, "v_nom"]
        }
    )

degree_df = pd.DataFrame(
    degree_records
).sort_values("degree")

print(
    degree_df.head(30).to_string(index=False)
)

# ==============================================================
# 10. BRANCHES CONNECTED TO LOW-DEGREE BUSES
# ==============================================================

print()
print("=" * 78)
print("10. BRANCHES CONNECTED TO LOW-DEGREE BUSES")
print("=" * 78)

low_degree = set(
    degree_df[
        degree_df["degree"] <= 1
    ]["bus"]
)

print(
    f"Low-degree buses (degree <= 1): "
    f"{len(low_degree)}"
)

if low_degree:

    for bus in sorted(low_degree):

        print()
        print(f"BUS: {bus}")

        connected_lines = n.lines[
            (n.lines["bus0"] == bus) |
            (n.lines["bus1"] == bus)
        ]

        if len(connected_lines):

            print(
                connected_lines[
                    [
                        "bus0",
                        "bus1",
                        "r",
                        "x",
                        "b",
                        "s_nom"
                    ]
                ].to_string()
            )

        connected_trafos = n.transformers[
            (n.transformers["bus0"] == bus) |
            (n.transformers["bus1"] == bus)
        ]

        if len(connected_trafos):

            print("Transformers:")

            print(
                connected_trafos[
                    [
                        "bus0",
                        "bus1",
                        "r",
                        "x",
                        "s_nom"
                    ]
                ].to_string()
            )

# ==============================================================
# 11. SUSPECT PARAMETER FLAGS
# ==============================================================

print()
print("=" * 78)
print("11. SUSPECT PARAMETER FLAGS")
print("=" * 78)

suspects = []

if len(n.lines):

    for name, row in n.lines.iterrows():

        flags = []

        r = float(row.r)
        x = float(row.x)
        b = float(row.b)
        s = float(row.s_nom)

        if not np.isfinite(r):
            flags.append("r_nonfinite")

        if not np.isfinite(x):
            flags.append("x_nonfinite")

        if not np.isfinite(b):
            flags.append("b_nonfinite")

        if not np.isfinite(s):
            flags.append("s_nom_nonfinite")

        if r <= 0:
            flags.append("r_nonpositive")

        if x <= 0:
            flags.append("x_nonpositive")

        if s <= 0:
            flags.append("s_nom_nonpositive")

        if x > 100:
            flags.append("very_high_x")

        if r > 20:
            flags.append("very_high_r")

        if x > 0 and r / x > 2:
            flags.append("very_high_r_over_x")

        if x > 0 and r / x < 0.005:
            flags.append("very_low_r_over_x")

        if row.bus0 == row.bus1:
            flags.append("self_loop")

        if flags:

            suspects.append(
                {
                    "name": name,
                    "bus0": row.bus0,
                    "bus1": row.bus1,
                    "r": r,
                    "x": x,
                    "b": b,
                    "s_nom": s,
                    "flags": ", ".join(flags)
                }
            )

if suspects:

    suspect_df = pd.DataFrame(suspects)

    print(
        suspect_df.to_string(index=False)
    )

else:

    print("No obvious line-parameter flags found.")

# ==============================================================
# 12. MOST IMPORTANT CHECK:
#    ARE LINE PARAMETERS POSSIBLY IN WRONG UNITS?
# ==============================================================

print()
print("=" * 78)
print("12. UNIT / SCALING SANITY CHECK")
print("=" * 78)

print()
print(
    "IMPORTANT:"
)
print(
    "PyPSA AC line r/x values are electrical impedances in ohms,"
)
print(
    "not per-unit values."
)

print()
print(
    "For the current network, a typical 220 kV line with"
)
print(
    "x around 0.18–51 ohm is not automatically wrong."
)

print()
print(
    "However, the AC PF exploding to ~10^42 MW indicates that"
)
print(
    "we must verify how these values were generated."
)

print()
print("Network base:")
print(f"sn_mva = {n.sn_mva}")

print()
print("First 10 lines:")
print(
    n.lines[
        [
            "bus0",
            "bus1",
            "r",
            "x",
            "b",
            "s_nom"
        ]
    ].head(10).to_string()
)

# ==============================================================
# 13. SNAPSHOT OPERATING POINT
# ==============================================================

print()
print("=" * 78)
print("13. OPERATING POINT")
print("=" * 78)

if SNAPSHOT in n.generators_t.p.index:

    generation = (
        n.generators_t.p
        .loc[SNAPSHOT]
        .sum()
    )

else:

    generation = np.nan

if SNAPSHOT in n.loads_t.p.index:

    load = (
        n.loads_t.p
        .loc[SNAPSHOT]
        .sum()
    )

else:

    load = np.nan

print(f"Generation : {generation:.6f} MW")
print(f"Load       : {load:.6f} MW")
print(
    f"Difference : {generation - load:.6f} MW"
)

# ==============================================================
# 14. FINAL DIAGNOSIS
# ==============================================================

print()
print("=" * 78)
print("FINAL DIAGNOSTIC INTERPRETATION")
print("=" * 78)

print()
print(
    "The AC solver is not failing because of the 123.679 MW"
)
print(
    "active-power imbalance."
)

print()
print(
    "The balanced test also failed with enormous numerical"
)
print(
    "voltage and branch-flow values."
)

print()
print(
    "Line charging removal reduced the numerical explosion,"
)
print(
    "but did NOT restore convergence."
)

print()
print(
    "Therefore:"
)

print(
    "  1. DO NOT change reinforcement."
)

print(
    "  2. DO NOT tune generation further."
)

print(
    "  3. DO NOT artificially reduce loads."
)

print(
    "  4. DO NOT accept the current AC PF results."
)

print(
    "  5. Inspect this diagnostic for bad electrical parameters,"
)
print(
    "     topology errors, unit/scaling errors, and problematic"
)
print(
    "     branches."
)

print()
print("=" * 78)
print("S2 AC BRANCH DIAGNOSTIC COMPLETE")
print("=" * 78)