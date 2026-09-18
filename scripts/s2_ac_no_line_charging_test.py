from pathlib import Path
import pypsa
import numpy as np

NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("S2 AC POWER FLOW - NO LINE CHARGING TEST")
print("=" * 70)

n = pypsa.Network(NETWORK)

# ------------------------------------------------------------
# Diagnostic copy
# ------------------------------------------------------------

# Remove line charging / shunt susceptance
print("\nDISABLING LINE CHARGING")
print("-----------------------")

original_b = n.lines.b.copy()

n.lines["b"] = 0.0

print(
    f"Original total |b| : {original_b.abs().sum():.8f}"
)

print(
    f"New total |b|      : {n.lines.b.abs().sum():.8f}"
)

# ------------------------------------------------------------
# Remove reactive-power setpoints
# ------------------------------------------------------------

if SNAPSHOT in n.loads_t.q_set.index:
    n.loads_t.q_set.loc[SNAPSHOT, :] = 0.0

if SNAPSHOT in n.generators_t.q_set.index:
    n.generators_t.q_set.loc[SNAPSHOT, :] = 0.0

# ------------------------------------------------------------
# Disconnect interconnectors
# ------------------------------------------------------------

for g in ["ewic_import", "greenlink_import"]:
    if g in n.generators.index:
        n.generators_t.p_set.loc[SNAPSHOT, g] = 0.0

for link in ["EWIC_interface", "Greenlink_interface"]:
    if link in n.links.index:
        n.links_t.p_set.loc[SNAPSHOT, link] = 0.0

# ------------------------------------------------------------
# Wind generation
# ------------------------------------------------------------

wind_generators = [
    "eirgrid_wind_way/88462768-220",
    "eirgrid_wind_way/104388595-220",
    "eirgrid_wind_way/516651650-220",
    "eirgrid_wind_way/88144450-220",
    "eirgrid_wind_way/254158424-220",
]

wind_generation = sum(
    n.generators_t.p_set.loc[SNAPSHOT, g]
    for g in wind_generators
)

total_load = n.loads_t.p_set.loc[SNAPSHOT].sum()

required_slack = total_load - wind_generation

SLACK = "eirgrid_non_wind_generation"

n.generators.loc[SLACK, "control"] = "Slack"

# Give slack enough capacity
n.generators.loc[SLACK, "p_nom"] = required_slack * 1.20

n.generators_t.p_set.loc[
    SNAPSHOT,
    SLACK
] = required_slack

# ------------------------------------------------------------
# Topology
# ------------------------------------------------------------

print("\nDETERMINING TOPOLOGY")
print("--------------------")

n.determine_network_topology()

print(
    n.sub_networks[
        ["carrier", "slack_bus"]
    ]
)

# ------------------------------------------------------------
# Run AC PF
# ------------------------------------------------------------

print("\nRUNNING AC POWER FLOW")
print("---------------------")

result = n.pf(
    snapshots=[SNAPSHOT]
)

print("\nPF RESULT")
print("---------")
print(result)

# ------------------------------------------------------------
# Convergence
# ------------------------------------------------------------

print("\nCONVERGENCE")
print("-----------")

print(result["converged"])

# ------------------------------------------------------------
# Voltage
# ------------------------------------------------------------

v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

finite_v = v[np.isfinite(v)]

print("\nBUS VOLTAGES")
print("------------")

print(
    f"Minimum voltage : {finite_v.min():.6f} pu"
)

print(
    f"Maximum voltage : {finite_v.max():.6f} pu"
)

print(
    f"Voltage range   : {(finite_v.max() - finite_v.min()):.6f} pu"
)

# ------------------------------------------------------------
# Suspicious voltages
# ------------------------------------------------------------

suspicious = v[
    (v < 0.8) |
    (v > 1.2)
]

print("\nSUSPICIOUS VOLTAGES")
print("-------------------")

if len(suspicious):
    print(
        suspicious.sort_values()
    )
else:
    print("None")

# ------------------------------------------------------------
# Line flows
# ------------------------------------------------------------

p0 = n.lines_t.p0.loc[SNAPSHOT]
p1 = n.lines_t.p1.loc[SNAPSHOT]

finite_p0 = p0[np.isfinite(p0)]
finite_p1 = p1[np.isfinite(p1)]

print("\nLINE FLOWS")
print("----------")

print(
    f"Maximum |P0| : {finite_p0.abs().max():.6f} MW"
)

print(
    f"Maximum |P1| : {finite_p1.abs().max():.6f} MW"
)

# ------------------------------------------------------------
# Generator Q
# ------------------------------------------------------------

print("\nGENERATOR REACTIVE POWER")
print("------------------------")

print(
    n.generators_t.q.loc[SNAPSHOT]
)

print("\n" + "=" * 70)
print("NO LINE CHARGING AC TEST COMPLETE")
print("=" * 70)