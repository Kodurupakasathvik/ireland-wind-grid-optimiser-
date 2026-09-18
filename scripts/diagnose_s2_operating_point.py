import pypsa
import numpy as np
import pandas as pd

NETWORK = r"data\processed\eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("S2 PEAK DEMAND - OPERATING POINT AUDIT")
print("=" * 70)

n = pypsa.Network(NETWORK)

print("\nNETWORK")
print(f"Snapshots: {list(n.snapshots)}")

if SNAPSHOT not in n.snapshots:
    raise ValueError(f"{SNAPSHOT} not found")

# ------------------------------------------------------------
# GENERATORS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("GENERATOR SNAPSHOT DATA")
print("-" * 70)

g = n.generators.copy()

for name, row in g.iterrows():

    p_set = (
        n.generators_t.p_set.loc[SNAPSHOT, name]
        if name in n.generators_t.p_set.columns
        else np.nan
    )

    q_set = (
        n.generators_t.q_set.loc[SNAPSHOT, name]
        if name in n.generators_t.q_set.columns
        else np.nan
    )

    print(
        f"{name}\n"
        f"  bus       = {row.bus}\n"
        f"  carrier   = {row.carrier}\n"
        f"  control   = {row.control}\n"
        f"  p_nom     = {row.p_nom}\n"
        f"  static p_set = {row.p_set}\n"
        f"  S2 p_set   = {p_set}\n"
        f"  S2 q_set   = {q_set}\n"
    )

# ------------------------------------------------------------
# LOADS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LOAD SNAPSHOT DATA")
print("-" * 70)

total_load = 0.0

for name, row in n.loads.iterrows():

    p_set = (
        n.loads_t.p_set.loc[SNAPSHOT, name]
        if name in n.loads_t.p_set.columns
        else np.nan
    )

    q_set = (
        n.loads_t.q_set.loc[SNAPSHOT, name]
        if name in n.loads_t.q_set.columns
        else np.nan
    )

    print(
        f"{name}\n"
        f"  bus        = {row.bus}\n"
        f"  static p_set = {row.p_set}\n"
        f"  S2 p_set   = {p_set}\n"
        f"  S2 q_set   = {q_set}\n"
    )

    if pd.notna(p_set):
        total_load += float(p_set)

# ------------------------------------------------------------
# GENERATION TOTAL
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("S2 ACTIVE POWER BALANCE")
print("-" * 70)

total_generation = 0.0

for name in n.generators.index:

    if name in n.generators_t.p_set.columns:

        value = n.generators_t.p_set.loc[SNAPSHOT, name]

        if pd.notna(value):
            total_generation += float(value)

print(f"Total generator S2 p_set : {total_generation:.6f} MW")
print(f"Total load S2 p_set      : {total_load:.6f} MW")
print(f"Generation - Load        : {total_generation-total_load:.6f} MW")

# ------------------------------------------------------------
# LINKS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("INTERCONNECTOR DATA")
print("-" * 70)

for name, row in n.links.iterrows():

    p_set = (
        n.links_t.p_set.loc[SNAPSHOT, name]
        if name in n.links_t.p_set.columns
        else np.nan
    )

    print(
        f"{name}\n"
        f"  bus0     = {row.bus0}\n"
        f"  bus1     = {row.bus1}\n"
        f"  p_nom    = {row.p_nom}\n"
        f"  static p_set = {row.p_set}\n"
        f"  S2 p_set = {p_set}\n"
    )

# ------------------------------------------------------------
# BUS VOLTAGE SETTINGS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("BUS VOLTAGE / CONTROL DATA")
print("-" * 70)

for name, row in n.buses.iterrows():

    vset = row.v_mag_pu_set

    if pd.isna(vset):
        vset = "NaN"

    print(
        f"{name:45s} "
        f"v_nom={row.v_nom:6.1f} "
        f"v_mag_pu_set={vset}"
    )

# ------------------------------------------------------------
# NaN CHECK
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("S2 NaN / INF CHECK")
print("-" * 70)

def check_df(label, df):

    if SNAPSHOT not in df.index:
        print(f"{label}: SNAPSHOT MISSING")
        return

    row = df.loc[SNAPSHOT]

    nan_count = row.isna().sum()

    inf_count = np.isinf(
        pd.to_numeric(row, errors="coerce").fillna(0)
    ).sum()

    print(
        f"{label:25s} "
        f"columns={len(row):3d} "
        f"NaN={nan_count:3d} "
        f"Inf={inf_count:3d}"
    )

check_df("generators_t.p_set", n.generators_t.p_set)
check_df("generators_t.q_set", n.generators_t.q_set)
check_df("loads_t.p_set", n.loads_t.p_set)
check_df("loads_t.q_set", n.loads_t.q_set)
check_df("links_t.p_set", n.links_t.p_set)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)