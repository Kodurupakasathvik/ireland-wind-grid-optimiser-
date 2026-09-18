import pypsa
import numpy as np

NETWORK = r"data\processed\eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("S2 AC STRUCTURE DIAGNOSTIC")
print("=" * 70)

n = pypsa.Network(NETWORK)

# ------------------------------------------------------------
# 1. BASIC NETWORK
# ------------------------------------------------------------

print("\nNETWORK")
print("-" * 70)

print("Buses        :", len(n.buses))
print("Lines        :", len(n.lines))
print("Transformers :", len(n.transformers))
print("Links        :", len(n.links))
print("Generators   :", len(n.generators))
print("Loads        :", len(n.loads))

# ------------------------------------------------------------
# 2. BUS CONNECTIVITY
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("BUS CONNECTIVITY")
print("-" * 70)

degree = {}

for bus in n.buses.index:
    line_count = (
        (n.lines.bus0 == bus).sum()
        + (n.lines.bus1 == bus).sum()
    )

    transformer_count = (
        (n.transformers.bus0 == bus).sum()
        + (n.transformers.bus1 == bus).sum()
    )

    link_count = (
        (n.links.bus0 == bus).sum()
        + (n.links.bus1 == bus).sum()
    )

    degree[bus] = line_count + transformer_count + link_count

isolated = [b for b, d in degree.items() if d == 0]

print("Isolated buses:", len(isolated))

for bus in isolated:
    print("  ", bus)

print("\nLowest-degree buses:")

for bus, d in sorted(degree.items(), key=lambda x: x[1])[:15]:
    print(f"{bus:50s} degree={d}")

# ------------------------------------------------------------
# 3. LINE PARAMETERS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LINE PARAMETER RANGE")
print("-" * 70)

for col in ["r", "x", "b", "s_nom"]:

    if col not in n.lines.columns:
        print(f"{col}: NOT PRESENT")
        continue

    values = n.lines[col].astype(float)

    finite = values[np.isfinite(values)]

    print(f"\n{col}")
    print("  min :", finite.min())
    print("  max :", finite.max())
    print("  mean:", finite.mean())

# ------------------------------------------------------------
# 4. VERY LOW IMPEDANCE LINES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LOW IMPEDANCE LINES")
print("-" * 70)

lines = n.lines.copy()

lines["abs_x"] = np.abs(lines.x)

print(
    lines[
        ["bus0", "bus1", "r", "x", "s_nom", "abs_x"]
    ]
    .sort_values("abs_x")
    .head(20)
    .to_string()
)

# ------------------------------------------------------------
# 5. TRANSFORMERS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("TRANSFORMER PARAMETERS")
print("-" * 70)

print(
    n.transformers[
        [
            "bus0",
            "bus1",
            "r",
            "x",
            "s_nom",
            "tap_ratio",
            "phase_shift",
        ]
    ].to_string()
)

# ------------------------------------------------------------
# 6. BUS NOMINAL VOLTAGES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("BUS VOLTAGE LEVELS")
print("-" * 70)

print(
    n.buses[
        ["v_nom", "v_mag_pu_set"]
    ].groupby("v_nom").size()
)

# ------------------------------------------------------------
# 7. GENERATOR LOCATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("GENERATOR LOCATIONS")
print("-" * 70)

print(
    n.generators[
        ["bus", "carrier", "control"]
    ].to_string()
)

# ------------------------------------------------------------
# 8. LOAD LOCATION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LOAD LOCATIONS")
print("-" * 70)

print(
    n.loads[
        ["bus"]
    ].to_string()
)

# ------------------------------------------------------------
# 9. SUBNETWORK STRUCTURE
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SUBNETWORK STRUCTURE")
print("-" * 70)

try:

    n.determine_network_topology()

    for i, sub in enumerate(n.sub_networks.itertuples()):

        print(
            f"Subnetwork {i}: "
            f"buses={len(sub.buses)}"
        )

except Exception as exc:

    print("Topology determination failed:")
    print(type(exc).__name__, exc)

# ------------------------------------------------------------
# 10. S2 ACTIVE POWER AT BUSES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("S2 ACTIVE POWER BY BUS")
print("-" * 70)

bus_power = {}

for bus in n.buses.index:

    generation = 0.0
    load = 0.0

    gens = n.generators.index[
        n.generators.bus == bus
    ]

    loads = n.loads.index[
        n.loads.bus == bus
    ]

    if len(gens) > 0:
        generation = float(
            n.generators_t.p_set.loc[
                SNAPSHOT, gens
            ].sum()
        )

    if len(loads) > 0:
        load = float(
            n.loads_t.p_set.loc[
                SNAPSHOT, loads
            ].sum()
        )

    if abs(generation) > 1e-9 or abs(load) > 1e-9:

        bus_power[bus] = (
            generation,
            load,
            generation - load
        )

for bus, values in bus_power.items():

    print(
        f"{bus:50s} "
        f"generation={values[0]:10.3f} "
        f"load={values[1]:10.3f} "
        f"net={values[2]:10.3f}"
    )

# ------------------------------------------------------------
# 11. CHECK FOR EXTREME PARAMETERS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("EXTREME PARAMETER CHECK")
print("-" * 70)

checks = {
    "line x < 0.01": (n.lines.x.abs() < 0.01).sum(),
    "line x > 100": (n.lines.x.abs() > 100).sum(),
    "line r < 0.001": (n.lines.r.abs() < 0.001).sum(),
    "line r > 100": (n.lines.r.abs() > 100).sum(),
    "line b huge": (n.lines.b.abs() > 100).sum(),
}

for name, count in checks.items():
    print(f"{name:25s}: {count}")

print("\n" + "=" * 70)
print("S2 AC STRUCTURE DIAGNOSTIC COMPLETE")
print("=" * 70)