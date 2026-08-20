from pathlib import Path
import pypsa
import numpy as np

NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("S2 MAIN-GRID ISOLATION TEST")
print("=" * 70)

n = pypsa.Network(NETWORK)

# ------------------------------------------------------------
# S2 REACTIVE POWER = 0
# ------------------------------------------------------------

n.loads_t.q_set.loc[SNAPSHOT, :] = 0.0
n.generators_t.q_set.loc[SNAPSHOT, :] = 0.0

# ------------------------------------------------------------
# DISABLE INTERCONNECTOR GENERATORS FOR THIS TEST
# ------------------------------------------------------------

for g in ["ewic_import", "greenlink_import"]:
    n.generators_t.p_set.loc[SNAPSHOT, g] = 0.0
    n.generators_t.q_set.loc[SNAPSHOT, g] = 0.0

# ------------------------------------------------------------
# DISABLE INTERCONNECTOR LINKS FOR THIS TEST
# ------------------------------------------------------------

for link in ["EWIC_interface", "Greenlink_interface"]:
    n.links_t.p_set.loc[SNAPSHOT, link] = 0.0

# ------------------------------------------------------------
# MAIN-GRID GENERATORS
# ------------------------------------------------------------

SLACK = "eirgrid_non_wind_generation"

n.generators.loc[:, "control"] = "PQ"
n.generators.at[SLACK, "control"] = "Slack"

# Calculate required slack from actual S2 load
wind_names = [
    "eirgrid_wind_way/88462768-220",
    "eirgrid_wind_way/104388595-220",
    "eirgrid_wind_way/516651650-220",
    "eirgrid_wind_way/88144450-220",
    "eirgrid_wind_way/254158424-220",
]

wind_generation = sum(
    n.generators_t.p_set.loc[SNAPSHOT, g]
    for g in wind_names
)

total_load = n.loads_t.p_set.loc[SNAPSHOT].sum()

required_slack = total_load - wind_generation

print("\nOPERATING POINT")
print("----------------")
print(f"Wind generation : {wind_generation:.6f} MW")
print(f"Total load      : {total_load:.6f} MW")
print(f"Required slack  : {required_slack:.6f} MW")

n.generators_t.p_set.loc[
    SNAPSHOT,
    SLACK
] = required_slack

# Give diagnostic slack sufficient nominal capacity
n.generators.at[SLACK, "p_nom"] = required_slack * 1.20

# ------------------------------------------------------------
# TOPOLOGY
# ------------------------------------------------------------

print("\nDETERMINING TOPOLOGY")
print("--------------------")

n.determine_network_topology()

print(n.sub_networks)

# ------------------------------------------------------------
# RUN PF
# ------------------------------------------------------------

print("\nRUNNING AC POWER FLOW")
print("---------------------")

result = n.pf(
    snapshots=[SNAPSHOT],
    distribute_slack=False,
    use_seed=False
)

print("\nPF RESULT")
print("---------")
print(result)

# ------------------------------------------------------------
# MAIN SUBNETWORK CONVERGENCE
# ------------------------------------------------------------

print("\nCONVERGENCE")
print("-----------")

print(result["converged"])

# ------------------------------------------------------------
# VOLTAGES
# ------------------------------------------------------------

v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

finite_v = v[np.isfinite(v)]

print("\nVOLTAGE")
print("-------")

if len(finite_v):
    print(f"Minimum : {finite_v.min():.6f} pu")
    print(f"Maximum : {finite_v.max():.6f} pu")

# ------------------------------------------------------------
# GENERATOR RESULT
# ------------------------------------------------------------

print("\nGENERATOR ACTIVE POWER")
print("----------------------")

print(
    n.generators_t.p.loc[SNAPSHOT]
)

print("\nGENERATOR REACTIVE POWER")
print("------------------------")

print(
    n.generators_t.q.loc[SNAPSHOT]
)

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
# LOADING
# ------------------------------------------------------------

loading0 = p0.abs() / n.lines.s_nom * 100
loading1 = p1.abs() / n.lines.s_nom * 100

loading = np.maximum(loading0, loading1)

finite_loading = loading[np.isfinite(loading)]

if len(finite_loading):
    print(
        f"Maximum line loading : {finite_loading.max():.6f}%"
    )

print("\n" + "=" * 70)
print("MAIN-GRID ISOLATION TEST COMPLETE")
print("=" * 70)