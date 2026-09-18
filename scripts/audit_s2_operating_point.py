import pypsa
import numpy as np
import pandas as pd

NETWORK = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 70)
print("IRELAND GRID - S2 CONTROLLED OPERATING-POINT AUDIT")
print("=" * 70)

n = pypsa.Network(NETWORK)

print("\nNETWORK")
print(f"  Buses        : {len(n.buses)}")
print(f"  Lines        : {len(n.lines)}")
print(f"  Transformers : {len(n.transformers)}")
print(f"  Generators   : {len(n.generators)}")
print(f"  Loads        : {len(n.loads)}")

print("\n" + "-" * 70)
print("S2 DISPATCH")
print("-" * 70)

gen = n.generators_t.p_set.loc[SNAPSHOT].copy()
load = n.loads_t.p_set.loc[SNAPSHOT].copy()

gen_total = gen.fillna(0).sum()
load_total = load.fillna(0).sum()

print(f"Total generator p_set : {gen_total:.6f} MW")
print(f"Total load p_set      : {load_total:.6f} MW")
print(f"Difference            : {gen_total - load_total:.6f} MW")

print("\nGENERATORS")
for name in n.generators.index:
    print(
        f"{name} | "
        f"bus={n.generators.at[name, 'bus']} | "
        f"carrier={n.generators.at[name, 'carrier']} | "
        f"p_set={gen.get(name, np.nan):.6f} MW"
    )

print("\nLOADS")
for name in n.loads.index:
    print(
        f"{name} | "
        f"bus={n.loads.at[name, 'bus']} | "
        f"p_set={load.get(name, np.nan):.6f} MW"
    )

print("\n" + "-" * 70)
print("GENERATION BY BUS")
print("-" * 70)

gen_bus = (
    pd.DataFrame({
        "bus": n.generators.bus,
        "p_set": gen.reindex(n.generators.index).fillna(0).values
    })
    .groupby("bus")["p_set"]
    .sum()
)

load_bus = (
    pd.DataFrame({
        "bus": n.loads.bus,
        "p_set": load.reindex(n.loads.index).fillna(0).values
    })
    .groupby("bus")["p_set"]
    .sum()
)

bus_audit = pd.DataFrame(index=n.buses.index)
bus_audit["generation_MW"] = gen_bus.reindex(bus_audit.index).fillna(0)
bus_audit["load_MW"] = load_bus.reindex(bus_audit.index).fillna(0)
bus_audit["net_injection_MW"] = (
    bus_audit["generation_MW"] - bus_audit["load_MW"]
)

print(
    bus_audit[
        (bus_audit["generation_MW"] != 0)
        | (bus_audit["load_MW"] != 0)
    ]
    .sort_values("net_injection_MW", ascending=False)
    .to_string()
)

print("\n" + "-" * 70)
print("CONTROLLED AC PF — DISTRIBUTED SLACK")
print("-" * 70)

# Work on a copy so the saved network is never modified.
test = n.copy()

result = test.pf(
    snapshots=[SNAPSHOT],
    distribute_slack=True
)

converged = bool(result.converged.loc[SNAPSHOT].all())

print(f"All subnetworks converged : {converged}")

if not converged:
    print("ERROR: Distributed-slack PF did not converge.")
    raise SystemExit(1)

print(f"Maximum PF error          : "
      f"{float(result.error.loc[SNAPSHOT].max()):.6e}")
print(f"Maximum iterations        : "
      f"{float(result.n_iter.loc[SNAPSHOT].max()):.0f}")

print("\n" + "-" * 70)
print("BUS VOLTAGE AUDIT")
print("-" * 70)

v = test.buses_t.v_mag_pu.loc[SNAPSHOT].dropna()

voltage = pd.DataFrame({"v_pu": v})
voltage["deviation_from_1pu"] = abs(voltage["v_pu"] - 1.0)

print("\nLOWEST 15 VOLTAGES")
print(voltage.sort_values("v_pu").head(15).to_string())

print("\nHIGHEST 15 VOLTAGES")
print(voltage.sort_values("v_pu", ascending=False).head(15).to_string())

print(f"\nMinimum voltage : {v.min():.6f} pu")
print(f"Maximum voltage : {v.max():.6f} pu")

print("\n" + "-" * 70)
print("LINE LOADING AUDIT")
print("-" * 70)

p0 = test.lines_t.p0.loc[SNAPSHOT].abs()
p1 = test.lines_t.p1.loc[SNAPSHOT].abs()

s_nom = test.lines.s_nom.replace(0, np.nan)

loading_p0 = p0 / s_nom * 100
loading_p1 = p1 / s_nom * 100

line_loading = pd.DataFrame({
    "loading_p0_pct": loading_p0,
    "loading_p1_pct": loading_p1
})

line_loading["max_loading_pct"] = line_loading.max(axis=1)

print("\nTOP 15 LINE LOADINGS")
print(
    line_loading
    .sort_values("max_loading_pct", ascending=False)
    .head(15)
    .to_string()
)

print("\nLINE DETAILS FOR TOP 15")

top_lines = (
    line_loading
    .sort_values("max_loading_pct", ascending=False)
    .head(15)
    .index
)

details = test.lines.loc[
    top_lines,
    ["bus0", "bus1", "r", "x", "s_nom"]
].copy()

details["loading_pct"] = line_loading.loc[
    top_lines, "max_loading_pct"
]

print(
    details
    .sort_values("loading_pct", ascending=False)
    .to_string()
)

print("\n" + "-" * 70)
print("SUMMARY")
print("-" * 70)

print(f"Total generation : {gen_total:.3f} MW")
print(f"Total load       : {load_total:.3f} MW")
print(f"PF converged     : {converged}")
print(f"Minimum V        : {v.min():.6f} pu")
print(f"Maximum V        : {v.max():.6f} pu")
print(
    f"Lines >100%      : "
    f"{int((line_loading['max_loading_pct'] > 100).sum())}"
)
print(
    f"Lines >110%      : "
    f"{int((line_loading['max_loading_pct'] > 110).sum())}"
)
print(
    f"Lines >120%      : "
    f"{int((line_loading['max_loading_pct'] > 120).sum())}"
)

print("\n" + "=" * 70)
print("S2 CONTROLLED OPERATING-POINT AUDIT COMPLETE")
print("=" * 70)