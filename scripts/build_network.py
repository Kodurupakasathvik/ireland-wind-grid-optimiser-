import pandas as pd
import pypsa
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

BUS_FILE = BASE / "data" / "processed" / "ireland_buses_model.csv"
LINE_FILE = BASE / "data" / "processed" / "ireland_lines_model.csv"
TRAFO_FILE = BASE / "data" / "processed" / "ireland_transformers_model.csv"

OUTPUT = BASE / "data" / "processed" / "ireland_network.nc"


# ============================================================
# LOAD DATA
# ============================================================

buses = pd.read_csv(BUS_FILE)
lines = pd.read_csv(LINE_FILE)
transformers = pd.read_csv(TRAFO_FILE)

print("Loaded model data:")
print("  Buses:", len(buses))
print("  Lines:", len(lines))
print("  Transformers:", len(transformers))


# ============================================================
# CREATE NETWORK
# ============================================================

n = pypsa.Network()

n.name = "Ireland Wind Grid Optimiser"


# ============================================================
# ADD BUSES
# ============================================================

for _, row in buses.iterrows():

    n.add(
        "Bus",
        row["bus_id"],
        v_nom=float(row["voltage"]),
        x=float(row["x"]),
        y=float(row["y"]),
        country=str(row["country"]),
    )


# ============================================================
# ADD LINES
# ============================================================

for _, row in lines.iterrows():

    n.add(
        "Line",
        row["line_id"],
        bus0=row["bus0"],
        bus1=row["bus1"],
        x=float(row["x"]),
        r=float(row["r"]),
        b=float(row["b"]),
        s_nom=float(row["s_nom"]),
        length=float(row["length"]) / 1000.0,
        num_parallel=int(row["circuits"]),
    )


# ============================================================
# ADD TRANSFORMERS
# ============================================================

for _, row in transformers.iterrows():

    n.add(
        "Transformer",
        row["transformer_id"],
        bus0=row["bus0"],
        bus1=row["bus1"],
        s_nom=float(row["s_nom"]),
        r=0.01,
        x=0.10,
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=== PYPSA NETWORK ===")

print("Buses:", len(n.buses))
print("Lines:", len(n.lines))
print("Transformers:", len(n.transformers))

print()
print("Bus voltage levels:")
print(n.buses["v_nom"].value_counts().sort_index())

print()
print("Line nominal capacities:")
print(
    n.lines["s_nom"]
    .describe()
    .to_string()
)

print()
print("Transformer nominal capacities:")
print(
    n.transformers["s_nom"]
    .describe()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

n.export_to_netcdf(OUTPUT)

print()
print("Network saved:")
print(OUTPUT)