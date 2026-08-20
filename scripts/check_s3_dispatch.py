import pypsa

NETWORK_FILE = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 100)
print("S3 DISPATCH STORAGE DIAGNOSTIC")
print("=" * 100)

n = pypsa.Network(NETWORK_FILE)

print("\nSNAPSHOTS")
print("-" * 100)
print(n.snapshots)

print("\nGENERATORS")
print("-" * 100)
print(n.generators[
    ["carrier", "p_nom", "control"]
].to_string())

print("\nGENERATORS_T TABLES")
print("-" * 100)

print("\ngenerators_t.p:")
print(n.generators_t.p)

print("\ngenerators_t.p_set:")
print(n.generators_t.p_set)

print("\ngenerators_t.p_max_pu:")
print(n.generators_t.p_max_pu)

print("\nLOADS")
print("-" * 100)
print(n.loads[
    ["bus", "p_set"]
].to_string())

print("\nLOAD TIME SERIES")
print("-" * 100)

print("loads_t.p:")
print(n.loads_t.p)

print("\nloads_t.p_set:")
print(n.loads_t.p_set)

print("\nSUMMARY")
print("-" * 100)

print("Generator static p_set:")
print(n.generators["p_set"])

print("\nGenerator static p_nom:")
print(n.generators["p_nom"])

print("\nLoad static p_set:")
print(n.loads["p_set"])

print("\nDone.")