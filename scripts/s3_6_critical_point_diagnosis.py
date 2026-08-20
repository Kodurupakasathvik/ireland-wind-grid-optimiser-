from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pypsa

# ============================================================
# S3.6 v2 — CRITICAL-POINT WEAK-BUS + CONGESTION DIAGNOSIS
# ============================================================

NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_second_reinforced_network.nc"
)

OUTPUT = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed"
    r"s3_6_critical_point_diagnosis.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

# Last VALID continuation point from S3.5
LAMBDA_CRITICAL = 0.953125

# Weak bus identified by S3.5
WEAK_BUS = "way/104388595-220"

# ------------------------------------------------------------
# LOAD NETWORK
# ------------------------------------------------------------

print("=" * 110)
print("S3.6 v2 — CRITICAL-POINT WEAK-BUS + CONGESTION DIAGNOSIS")
print("=" * 110)

print(f"Network  : {NETWORK}")
print(f"Snapshot : {SNAPSHOT}")
print(f"Lambda   : {LAMBDA_CRITICAL}")
print(f"Weak bus : {WEAK_BUS}")

n = pypsa.Network(NETWORK)

print("\nNETWORK")
print("-" * 110)
print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")

# ============================================================
# APPLY CRITICAL LOAD / GENERATION SCALING
# ============================================================

# Preserve original values
load_p_original = n.loads_t.p_set.loc[SNAPSHOT].copy()
gen_p_original = n.generators_t.p_set.loc[SNAPSHOT].copy()

# Scale load around the continuation base
n.loads_t.p_set.loc[SNAPSHOT] = load_p_original * LAMBDA_CRITICAL

# Scale fixed generation consistently
n.generators_t.p_set.loc[SNAPSHOT] = gen_p_original * LAMBDA_CRITICAL

# ============================================================
# RUN AC POWER FLOW
# ============================================================

print("\n" + "=" * 110)
print("CRITICAL-POINT AC POWER FLOW")
print("=" * 110)

print(f"Running at lambda = {LAMBDA_CRITICAL}")

try:
    n.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-8,
        use_seed=True,
    )
except Exception as e:
    print("\nPOWER FLOW EXCEPTION:")
    print(e)

# ============================================================
# VALIDATE RESULT BEFORE ANALYSING
# ============================================================

v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

finite_voltage = np.isfinite(v).all()
reasonable_voltage = ((v.abs() < 2.0).all())

if not finite_voltage or not reasonable_voltage:
    print("\n" + "=" * 110)
    print("CRITICAL ERROR — INVALID AC STATE")
    print("=" * 110)
    print("The power-flow state is numerically invalid.")
    print("S3.6 v2 will NOT analyse the resulting flows.")
    print("Do not use this output as an engineering result.")
    raise RuntimeError(
        "Critical-point AC solution is invalid. "
        "Need to restore the exact S3.5 continuation state."
    )

print("\nVALID AC SOLUTION CONFIRMED")

# ============================================================
# 1. BUS VOLTAGE RANKING
# ============================================================

print("\n" + "=" * 110)
print("1. LOWEST-VOLTAGE BUSES")
print("=" * 110)

bus_voltage = pd.DataFrame({
    "bus": v.index,
    "voltage_pu": v.values,
})

bus_voltage["voltage_deviation"] = (
    1.0 - bus_voltage["voltage_pu"].abs()
)

print(
    bus_voltage
    .sort_values("voltage_pu")
    .head(20)
    .to_string(index=False)
)

# ============================================================
# 2. WEAK BUS
# ============================================================

print("\n" + "=" * 110)
print("2. CRITICAL WEAK BUS")
print("=" * 110)

if WEAK_BUS not in n.buses.index:
    raise KeyError(f"Weak bus {WEAK_BUS} not found.")

weak_voltage = float(v.loc[WEAK_BUS])

print(f"Bus ID       : {WEAK_BUS}")
print(f"Voltage      : {weak_voltage:.6f} pu")
print(f"Nominal kV   : {n.buses.at[WEAK_BUS, 'v_nom']}")
print(f"Carrier      : {n.buses.at[WEAK_BUS, 'carrier']}")
print(f"Country      : {n.buses.at[WEAK_BUS, 'country']}")

# ============================================================
# 3. LINES CONNECTED TO WEAK BUS
# ============================================================

print("\n" + "=" * 110)
print("3. LINES CONNECTED TO WEAK BUS")
print("=" * 110)

connected_lines = n.lines[
    (n.lines.bus0 == WEAK_BUS) |
    (n.lines.bus1 == WEAK_BUS)
].copy()

line_loading = (
    n.lines_t.p0.loc[SNAPSHOT].abs()
    / n.lines.s_nom
    * 100.0
)

connected_lines["loading_pct"] = (
    line_loading.loc[connected_lines.index]
)

connected_lines["p0_MW"] = (
    n.lines_t.p0.loc[SNAPSHOT]
    .loc[connected_lines.index]
)

connected_lines["q0_Mvar"] = (
    n.lines_t.q0.loc[SNAPSHOT]
    .loc[connected_lines.index]
)

cols = [
    "bus0",
    "bus1",
    "s_nom",
    "r",
    "x",
    "length",
    "loading_pct",
    "p0_MW",
    "q0_Mvar",
]

print(
    connected_lines[cols]
    .sort_values("loading_pct", ascending=False)
    .to_string()
)

# ============================================================
# 4. GLOBAL CONGESTION
# ============================================================

print("\n" + "=" * 110)
print("4. GLOBAL LINE CONGESTION")
print("=" * 110)

line_results = n.lines.copy()

line_results["p0_MW"] = n.lines_t.p0.loc[SNAPSHOT]
line_results["q0_Mvar"] = n.lines_t.q0.loc[SNAPSHOT]

line_results["loading_pct"] = (
    np.sqrt(
        line_results["p0_MW"] ** 2
        + line_results["q0_Mvar"] ** 2
    )
    / line_results["s_nom"]
    * 100.0
)

line_results["overloaded"] = (
    line_results["loading_pct"] > 100.0
)

top_lines = (
    line_results
    .sort_values("loading_pct", ascending=False)
    .head(20)
)

print(
    top_lines[
        [
            "bus0",
            "bus1",
            "s_nom",
            "p0_MW",
            "q0_Mvar",
            "loading_pct",
            "overloaded",
        ]
    ].to_string()
)

# ============================================================
# 5. TRANSFORMER LOADING
# ============================================================

print("\n" + "=" * 110)
print("5. TRANSFORMER LOADING")
print("=" * 110)

if len(n.transformers) > 0:

    trafo_results = n.transformers.copy()

    trafo_results["p0_MW"] = (
        n.transformers_t.p0.loc[SNAPSHOT]
    )

    trafo_results["q0_Mvar"] = (
        n.transformers_t.q0.loc[SNAPSHOT]
    )

    trafo_results["loading_pct"] = (
        np.sqrt(
            trafo_results["p0_MW"] ** 2
            + trafo_results["q0_Mvar"] ** 2
        )
        / trafo_results["s_nom"]
        * 100.0
    )

    print(
        trafo_results[
            [
                "bus0",
                "bus1",
                "s_nom",
                "p0_MW",
                "q0_Mvar",
                "loading_pct",
            ]
        ]
        .sort_values("loading_pct", ascending=False)
        .to_string()
    )

else:
    trafo_results = pd.DataFrame()

# ============================================================
# 6. GENERATION / LOAD AT WEAK BUS
# ============================================================

print("\n" + "=" * 110)
print("6. POWER INJECTION AT WEAK BUS")
print("=" * 110)

weak_generators = n.generators[
    n.generators.bus == WEAK_BUS
]

weak_loads = n.loads[
    n.loads.bus == WEAK_BUS
]

if len(weak_generators):
    print("\nGENERATION")
    print(
        pd.DataFrame({
            "generator": weak_generators.index,
            "carrier": weak_generators.carrier,
            "p_nom_MW": weak_generators.p_nom,
            "dispatch_MW":
                n.generators_t.p.loc[
                    SNAPSHOT,
                    weak_generators.index
                ],
        }).to_string(index=False)
    )
else:
    print("No generators directly connected.")

if len(weak_loads):
    print("\nLOAD")
    print(
        pd.DataFrame({
            "load": weak_loads.index,
            "p_MW":
                n.loads_t.p.loc[
                    SNAPSHOT,
                    weak_loads.index
                ],
        }).to_string(index=False)
    )
else:
    print("No loads directly connected.")

# ============================================================
# 7. NETWORK-WIDE SUMMARY
# ============================================================

max_line = line_results["loading_pct"].idxmax()
max_line_loading = float(
    line_results.loc[max_line, "loading_pct"]
)

overloaded_count = int(
    line_results["overloaded"].sum()
)

if len(trafo_results):
    max_trafo = trafo_results["loading_pct"].idxmax()
    max_trafo_loading = float(
        trafo_results.loc[max_trafo, "loading_pct"]
    )
else:
    max_trafo = ""
    max_trafo_loading = np.nan

print("\n" + "=" * 110)
print("S3.6 v2 SUMMARY")
print("=" * 110)

print(f"Weak bus              : {WEAK_BUS}")
print(f"Weak-bus voltage      : {weak_voltage:.6f} pu")
print(f"Maximum line loading  : {max_line_loading:.6f} %")
print(f"Maximum loaded line   : {max_line}")
print(f"Overloaded lines      : {overloaded_count}")
print(f"Maximum transformer   : {max_trafo_loading:.6f} %")
print(f"Maximum transformer   : {max_trafo}")

# ============================================================
# 8. SAVE MACHINE-READABLE RESULTS
# ============================================================

rows = []

for bus, voltage in bus_voltage.iterrows():

    rows.append({
        "lambda": LAMBDA_CRITICAL,
        "type": "bus",
        "id": bus,
        "metric": "voltage_pu",
        "value": voltage["voltage_pu"],
    })

for line, row in line_results.iterrows():

    rows.append({
        "lambda": LAMBDA_CRITICAL,
        "type": "line",
        "id": line,
        "metric": "loading_pct",
        "value": row["loading_pct"],
    })

if len(trafo_results):

    for trafo, row in trafo_results.iterrows():

        rows.append({
            "lambda": LAMBDA_CRITICAL,
            "type": "transformer",
            "id": trafo,
            "metric": "loading_pct",
            "value": row["loading_pct"],
        })

results = pd.DataFrame(rows)

results.to_csv(OUTPUT, index=False)

print("\nResults saved to:")
print(OUTPUT)

print("\n" + "=" * 110)
print("S3.6 v2 COMPLETE")
print("=" * 110)