import pypsa
import pandas as pd
import numpy as np
import networkx as nx

NETWORK = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 78)
print("S2 AC INJECTION / TOPOLOGY / DC STRESS DIAGNOSTIC")
print("=" * 78)

n = pypsa.Network(NETWORK)
snap = SNAPSHOT

print("\nNETWORK")
print("-" * 78)
print("Buses        :", len(n.buses))
print("Lines        :", len(n.lines))
print("Transformers:", len(n.transformers))
print("Generators  :", len(n.generators))
print("Loads       :", len(n.loads))
print("Links       :", len(n.links))


# ============================================================================
# 1. GENERATOR AUDIT
# ============================================================================

print("\n" + "=" * 78)
print("1. GENERATOR PLACEMENT")
print("=" * 78)

gen = pd.DataFrame({
    "bus": n.generators.bus,
    "p_nom": n.generators.p_nom,
    "p_set": n.generators_t.p_set.loc[snap],
    "control": n.generators.control,
    "carrier": n.generators.carrier,
})

print(gen.sort_values("p_set", ascending=False).to_string())

print("\nGENERATION BY BUS")
print("-" * 78)

gen_by_bus = (
    gen.groupby("bus")
       .agg(
           generators=("p_set", "size"),
           p_set_MW=("p_set", "sum"),
           p_nom_MW=("p_nom", "sum"),
       )
       .sort_values("p_set_MW", ascending=False)
)

print(gen_by_bus.to_string())


# ============================================================================
# 2. LOAD AUDIT
# ============================================================================

print("\n" + "=" * 78)
print("2. LOAD PLACEMENT")
print("=" * 78)

load = pd.DataFrame({
    "bus": n.loads.bus,
    "p_set": n.loads_t.p_set.loc[snap],
    "carrier": n.loads.carrier,
})

print(load.sort_values("p_set", ascending=False).to_string())

print("\nLOAD BY BUS")
print("-" * 78)

load_by_bus = (
    load.groupby("bus")
        .agg(
            loads=("p_set", "size"),
            p_load_MW=("p_set", "sum"),
        )
        .sort_values("p_load_MW", ascending=False)
)

print(load_by_bus.to_string())


# ============================================================================
# 3. NET INJECTION BY BUS
# ============================================================================

print("\n" + "=" * 78)
print("3. NET BUS INJECTION")
print("=" * 78)

all_buses = n.buses.index

p_gen = gen.groupby("bus")["p_set"].sum()
p_load = load.groupby("bus")["p_set"].sum()

inj = pd.DataFrame(index=all_buses)
inj["v_nom"] = n.buses.v_nom
inj["generation_MW"] = p_gen.reindex(all_buses).fillna(0)
inj["load_MW"] = p_load.reindex(all_buses).fillna(0)
inj["net_injection_MW"] = (
    inj["generation_MW"] - inj["load_MW"]
)
inj["abs_injection_MW"] = inj["net_injection_MW"].abs()

print(
    inj.sort_values(
        "abs_injection_MW",
        ascending=False
    ).head(30).to_string()
)

print("\nTOTALS")
print("-" * 78)
print("Generation:", inj["generation_MW"].sum())
print("Load      :", inj["load_MW"].sum())
print("Net       :", inj["net_injection_MW"].sum())


# ============================================================================
# 4. BUS DEGREE / TOPOLOGY
# ============================================================================

print("\n" + "=" * 78)
print("4. BUS TOPOLOGY")
print("=" * 78)

G = nx.Graph()

for bus in n.buses.index:
    G.add_node(bus)

for name, row in n.lines.iterrows():
    G.add_edge(row.bus0, row.bus1, element="line", name=name)

for name, row in n.transformers.iterrows():
    G.add_edge(row.bus0, row.bus1, element="transformer", name=name)

degree = dict(G.degree())

degree_df = pd.DataFrame({
    "v_nom": n.buses.v_nom,
    "degree": pd.Series(degree),
    "net_injection_MW": inj["net_injection_MW"],
})

print("\nDEGREE DISTRIBUTION")
print(degree_df["degree"].value_counts().sort_index())

print("\nDEAD-END / RADIAL BUSES")
print("-" * 78)

radial = degree_df[degree_df["degree"] <= 1].sort_values(
    "degree"
)

print(radial.to_string())


# ============================================================================
# 5. ARTICULATION BUSES
# ============================================================================

print("\n" + "=" * 78)
print("5. ARTICULATION / BRIDGE AUDIT")
print("=" * 78)

components = list(nx.connected_components(G))

print("Graph components:", len(components))

for i, comp in enumerate(components):
    print(f"Component {i}: {len(comp)} buses")

main_component = max(components, key=len)

Gmain = G.subgraph(main_component).copy()

articulation = list(nx.articulation_points(Gmain))
bridges = list(nx.bridges(Gmain))

print("\nMain component buses:", len(Gmain))
print("Articulation buses  :", len(articulation))
print("Bridges             :", len(bridges))

print("\nARTICULATION BUSES")
print("-" * 78)

for bus in articulation:
    print(
        bus,
        "degree=", Gmain.degree(bus),
        "injection=",
        inj.loc[bus, "net_injection_MW"]
        if bus in inj.index else np.nan
    )


# ============================================================================
# 6. BRIDGE BRANCHES
# ============================================================================

print("\n" + "=" * 78)
print("6. BRIDGE BRANCHES")
print("=" * 78)

bridge_rows = []

for u, v in bridges:

    found = False

    for name, row in n.lines.iterrows():
        if (
            (row.bus0 == u and row.bus1 == v)
            or
            (row.bus0 == v and row.bus1 == u)
        ):
            bridge_rows.append({
                "type": "line",
                "name": name,
                "bus0": row.bus0,
                "bus1": row.bus1,
                "r": row.r,
                "x": row.x,
                "s_nom": row.s_nom,
            })
            found = True

    for name, row in n.transformers.iterrows():
        if (
            (row.bus0 == u and row.bus1 == v)
            or
            (row.bus0 == v and row.bus1 == u)
        ):
            bridge_rows.append({
                "type": "transformer",
                "name": name,
                "bus0": row.bus0,
                "bus1": row.bus1,
                "r": row.r,
                "x": row.x,
                "s_nom": row.s_nom,
            })
            found = True

print(pd.DataFrame(bridge_rows).to_string(index=False))


# ============================================================================
# 7. DC / LINEAR FLOW STRESS
# ============================================================================

print("\n" + "=" * 78)
print("7. DC / LINEAR FLOW STRESS")
print("=" * 78)

# Copy network so original is untouched
ndc = pypsa.Network(NETWORK)

try:
    ndc.lpf(snapshots=[snap])

    print("Linear PF completed.")

    line_flow = pd.DataFrame({
        "p0_MW": ndc.lines_t.p0.loc[snap],
        "p1_MW": ndc.lines_t.p1.loc[snap],
        "s_nom_MW": ndc.lines.s_nom,
    })

    line_flow["max_abs_flow_MW"] = line_flow[
        ["p0_MW", "p1_MW"]
    ].abs().max(axis=1)

    line_flow["loading_pct"] = (
        line_flow["max_abs_flow_MW"]
        / line_flow["s_nom_MW"]
        * 100
    )

    print("\nMOST STRESSED LINES")
    print("-" * 78)

    print(
        line_flow
        .sort_values("loading_pct", ascending=False)
        .head(25)
        .to_string()
    )

    print("\nMAX LINE LOADING:")
    print(
        line_flow["loading_pct"].max(),
        "%"
    )

except Exception as e:
    print("LINEAR PF FAILED:", type(e).__name__, e)


# ============================================================================
# 8. DC FLOW + TOPOLOGY CROSS-CHECK
# ============================================================================

print("\n" + "=" * 78)
print("8. HIGHEST-INJECTION BUSES")
print("=" * 78)

for bus, row in (
    inj.sort_values(
        "abs_injection_MW",
        ascending=False
    ).head(15).iterrows()
):

    print(
        f"{bus:45s} "
        f"V={row.v_nom:6.1f} kV "
        f"Gen={row.generation_MW:10.3f} MW "
        f"Load={row.load_MW:10.3f} MW "
        f"Net={row.net_injection_MW:10.3f} MW "
        f"Degree={degree.get(bus, 0)}"
    )


print("\n" + "=" * 78)
print("DIAGNOSTIC COMPLETE")
print("=" * 78)