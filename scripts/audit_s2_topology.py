import pypsa
import numpy as np

NETWORK = r"data\processed\eirgrid_optimized_network.nc"

print("=" * 70)
print("S2 PEAK DEMAND - TOPOLOGY / INTERCONNECTOR AUDIT")
print("=" * 70)

n = pypsa.Network(NETWORK)

snap = "S2_PEAK_DEMAND"

print("\nNETWORK")
print("Buses        :", len(n.buses))
print("Lines        :", len(n.lines))
print("Links        :", len(n.links))
print("Transformers :", len(n.transformers))
print("Generators   :", len(n.generators))
print("Loads        :", len(n.loads))

print("\n" + "-" * 70)
print("INTERCONNECTOR GENERATORS")
print("-" * 70)

for name in ["ewic_import", "greenlink_import"]:

    if name not in n.generators.index:
        print(f"{name}: NOT FOUND")
        continue

    g = n.generators.loc[name]

    print(f"\n{name}")
    print("bus     :", g.bus)
    print("carrier :", g.carrier)
    print("control :", g.control)
    print("p_nom   :", g.p_nom)

    if snap in n.generators_t.p_set.columns:
        print("S2 p_set:", n.generators_t.p_set.at[snap, name])

print("\n" + "-" * 70)
print("INTERCONNECTOR LINKS")
print("-" * 70)

for name in ["EWIC_interface", "Greenlink_interface"]:

    if name not in n.links.index:
        print(f"{name}: NOT FOUND")
        continue

    link = n.links.loc[name]

    print(f"\n{name}")
    print("bus0    :", link.bus0)
    print("bus1    :", link.bus1)
    print("carrier :", link.carrier)
    print("p_nom   :", link.p_nom)
    print("efficiency :", link.efficiency)

    if hasattr(n.links_t, "p_set") and name in n.links_t.p_set.columns:
        print("S2 p_set:", n.links_t.p_set.at[snap, name])
    else:
        print("S2 p_set: NO TIME-SERIES COLUMN")

print("\n" + "-" * 70)
print("S2 GENERATION")
print("-" * 70)

if snap in n.generators_t.p_set.index:
    s2_generation = n.generators_t.p_set.loc[snap].sum()
else:
    s2_generation = 0

print("Total generation:", s2_generation, "MW")

print("\n" + "-" * 70)
print("S2 LOAD")
print("-" * 70)

if snap in n.loads_t.p_set.index:
    s2_load = n.loads_t.p_set.loc[snap].sum()
else:
    s2_load = 0

print("Total load:", s2_load, "MW")

print("\n" + "-" * 70)
print("S2 BALANCE")
print("-" * 70)

print("Generation :", s2_generation)
print("Load       :", s2_load)
print("Deficit    :", s2_load - s2_generation)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)