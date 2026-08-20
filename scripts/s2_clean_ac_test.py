from pathlib import Path
import pypsa
import numpy as np
import pandas as pd

NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("S2 CLEAN AC POWER-FLOW TEST")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

n = pypsa.Network(NETWORK)

print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")
print(f"Links        : {len(n.links)}")

# ------------------------------------------------------------
# WORK ON COPY ONLY
# ------------------------------------------------------------

print("\nCreating in-memory diagnostic copy...")

# ------------------------------------------------------------
# S2 INTERCONNECTOR FLOWS
# ------------------------------------------------------------

n.links_t.p_set.loc[SNAPSHOT, "EWIC_interface"] = 529.970
n.links_t.p_set.loc[SNAPSHOT, "Greenlink_interface"] = 513.201

print("\nINTERCONNECTOR FLOWS")
print("--------------------")
print(
    "EWIC      :",
    n.links_t.p_set.loc[SNAPSHOT, "EWIC_interface"],
    "MW"
)
print(
    "Greenlink :",
    n.links_t.p_set.loc[SNAPSHOT, "Greenlink_interface"],
    "MW"
)

# ------------------------------------------------------------
# EXPLICIT REACTIVE POWER = 0 FOR DIAGNOSTIC TEST
# ------------------------------------------------------------

print("\nSETTING Q = 0 FOR DIAGNOSTIC TEST")

n.loads_t.q_set.loc[SNAPSHOT, :] = 0.0
n.generators_t.q_set.loc[SNAPSHOT, :] = 0.0

# ------------------------------------------------------------
# GENERATOR CONTROLS
# ------------------------------------------------------------

print("\nGENERATOR CONTROLS BEFORE TEST")
print("--------------------------------")
print(n.generators[["bus", "carrier", "control", "p_nom"]])

# ------------------------------------------------------------
# ONE SLACK
# ------------------------------------------------------------

SLACK = "eirgrid_non_wind_generation"

print("\nSETTING SLACK")
print("-------------")
print(f"Slack generator: {SLACK}")

n.generators.loc[:, "control"] = "PQ"

n.generators.at[SLACK, "control"] = "Slack"

# Diagnostic-only capacity.
# This is NOT a reinforcement decision.
n.generators.at[SLACK, "p_nom"] = 10000.0

print(n.generators.loc[
    SLACK,
    ["bus", "carrier", "control", "p_nom"]
])

# ------------------------------------------------------------
# TOPOLOGY
# ------------------------------------------------------------

print("\nDETERMINING NETWORK TOPOLOGY")
print("-----------------------------")

n.determine_network_topology()

print("\nSUBNETWORKS")
print("-----------")
print(n.sub_networks)

print("\nBUS CONTROLS")
print("------------")
print(n.buses[["control", "generator", "sub_network"]])

# ------------------------------------------------------------
# ACTIVE POWER BALANCE BEFORE PF
# ------------------------------------------------------------

gen = n.generators_t.p_set.loc[SNAPSHOT].sum()
load = n.loads_t.p_set.loc[SNAPSHOT].sum()

print("\nACTIVE POWER BALANCE BEFORE PF")
print("------------------------------")
print(f"Generation : {gen:.6f} MW")
print(f"Load       : {load:.6f} MW")
print(f"Difference : {gen - load:.6f} MW")

# ------------------------------------------------------------
# RUN AC POWER FLOW
# ------------------------------------------------------------

print("\nRUNNING AC POWER FLOW")
print("---------------------")

result = n.pf(
    snapshots=[SNAPSHOT],
    distribute_slack=False,
    use_seed=False
)

print("\nPF RETURN")
print("---------")
print(result)

# ------------------------------------------------------------
# CONVERGENCE
# ------------------------------------------------------------

print("\nCONVERGENCE")
print("-----------")

try:
    print(result["converged"])
except Exception:
    print("Could not read convergence result.")

# ------------------------------------------------------------
# VOLTAGE
# ------------------------------------------------------------

print("\nBUS VOLTAGES")
print("------------")

v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

finite_v = v[np.isfinite(v)]

if len(finite_v):
    print(f"Minimum voltage : {finite_v.min():.6f} pu")
    print(f"Maximum voltage : {finite_v.max():.6f} pu")

    bad_v = v[(~np.isfinite(v)) | (v <= 0) | (v > 2)]

    print(f"Suspicious buses: {len(bad_v)}")

    if len(bad_v):
        print(bad_v.sort_values())

# ------------------------------------------------------------
# LINE FLOWS
# ------------------------------------------------------------

print("\nLINE FLOWS")
print("----------")

p0 = n.lines_t.p0.loc[SNAPSHOT]
p1 = n.lines_t.p1.loc[SNAPSHOT]

finite_p0 = p0[np.isfinite(p0)]
finite_p1 = p1[np.isfinite(p1)]

if len(finite_p0):
    print(
        "Maximum |P0| :",
        f"{finite_p0.abs().max():.6f} MW"
    )

if len(finite_p1):
    print(
        "Maximum |P1| :",
        f"{finite_p1.abs().max():.6f} MW"
    )

# ------------------------------------------------------------
# LINE LOADING
# ------------------------------------------------------------

loading0 = p0.abs() / n.lines.s_nom * 100
loading1 = p1.abs() / n.lines.s_nom * 100

loading = pd = np.maximum(loading0, loading1)

finite_loading = loading[np.isfinite(loading)]

if len(finite_loading):
    print(
        "Maximum line loading :",
        f"{finite_loading.max():.6f}%"
    )

# ------------------------------------------------------------
# GENERATOR RESULT
# ------------------------------------------------------------

print("\nGENERATOR RESULTS")
print("-----------------")

print(
    n.generators_t.p.loc[
        SNAPSHOT
    ]
)

print("\nReactive power:")
print(
    n.generators_t.q.loc[
        SNAPSHOT
    ]
)

print("\n" + "=" * 70)
print("CLEAN S2 AC TEST COMPLETE")
print("=" * 70)