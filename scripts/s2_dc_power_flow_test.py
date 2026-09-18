from pathlib import Path
import pypsa
import numpy as np

NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("S2 DC / LINEAR POWER-FLOW DIAGNOSTIC")
print("=" * 70)

n = pypsa.Network(NETWORK)

# ------------------------------------------------------------
# Diagnostic copy
# ------------------------------------------------------------

# Remove reactive-power effects for this diagnostic
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
# Main-grid generation
# ------------------------------------------------------------

SLACK = "eirgrid_non_wind_generation"

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

n.generators.loc[SLACK, "control"] = "Slack"
n.generators.loc[SLACK, "p_nom"] = required_slack * 1.20

n.generators_t.p_set.loc[
    SNAPSHOT,
    SLACK
] = required_slack

print("\nOPERATING POINT")
print("----------------")
print(f"Wind generation : {wind_generation:.6f} MW")
print(f"Load             : {total_load:.6f} MW")
print(f"Slack setpoint   : {required_slack:.6f} MW")

# ------------------------------------------------------------
# Topology
# ------------------------------------------------------------

print("\nDETERMINING TOPOLOGY")
print("--------------------")

n.determine_network_topology()

print(n.sub_networks[["carrier", "slack_bus"]])

# ------------------------------------------------------------
# LINEAR POWER FLOW
# ------------------------------------------------------------

print("\nRUNNING LINEAR POWER FLOW")
print("-------------------------")

try:
    result = n.lpf(
        snapshots=[SNAPSHOT]
    )

    print("\nLPF RETURN")
    print("----------")
    print(result)

except Exception as e:
    print("\nLPF FAILED")
    print("----------")
    print(type(e).__name__, str(e))
    raise

# ------------------------------------------------------------
# BUS ANGLES
# ------------------------------------------------------------

theta = n.buses_t.v_ang.loc[SNAPSHOT]

finite_theta = theta[np.isfinite(theta)]

print("\nBUS ANGLES")
print("----------")

if len(finite_theta):
    print(f"Minimum angle : {finite_theta.min():.6f} rad")
    print(f"Maximum angle : {finite_theta.max():.6f} rad")
    print(f"Angle range   : {(finite_theta.max() - finite_theta.min()):.6f} rad")

# ------------------------------------------------------------
# LINE FLOWS
# ------------------------------------------------------------

p0 = n.lines_t.p0.loc[SNAPSHOT]
p1 = n.lines_t.p1.loc[SNAPSHOT]

finite_p0 = p0[np.isfinite(p0)]
finite_p1 = p1[np.isfinite(p1)]

print("\nLINE FLOWS")
print("----------")

if len(finite_p0):
    print(
        f"Maximum |P0| : {finite_p0.abs().max():.6f} MW"
    )

if len(finite_p1):
    print(
        f"Maximum |P1| : {finite_p1.abs().max():.6f} MW"
    )

# ------------------------------------------------------------
# LINE LOADINGS
# ------------------------------------------------------------

loading0 = p0.abs() / n.lines.s_nom * 100
loading1 = p1.abs() / n.lines.s_nom * 100

loading = np.maximum(loading0, loading1)

finite_loading = loading[np.isfinite(loading)]

print("\nLINE LOADING")
print("------------")

if len(finite_loading):
    print(
        f"Maximum : {finite_loading.max():.6f}%"
    )

    print("\nTop 10:")
    print(
        loading.sort_values(ascending=False).head(10)
    )

# ------------------------------------------------------------
# POWER BALANCE
# ------------------------------------------------------------

print("\nACTIVE POWER")
print("------------")

print(
    n.generators_t.p.loc[SNAPSHOT]
)

print("\n" + "=" * 70)
print("DC / LINEAR POWER-FLOW TEST COMPLETE")
print("=" * 70)