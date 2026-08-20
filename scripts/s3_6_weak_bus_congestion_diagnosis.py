from pathlib import Path
import pandas as pd
import numpy as np
import pypsa


# ============================================================
# S3.6 — WEAK-BUS + CONGESTION DIAGNOSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NETWORK_FILE = ROOT / "data" / "processed" / "eirgrid_second_reinforced_network.nc"
OUTPUT_FILE = ROOT / "data" / "processed" / "s3_6_weak_bus_congestion_diagnosis.csv"

SNAPSHOT = "S2_PEAK_DEMAND"
WEAK_BUS = "way/104388595-220"


print("=" * 110)
print("S3.6 — WEAK-BUS + CONGESTION DIAGNOSIS")
print("=" * 110)

print(f"Network : {NETWORK_FILE}")
print(f"Snapshot: {SNAPSHOT}")
print(f"Weak bus: {WEAK_BUS}")
print()


# ============================================================
# LOAD NETWORK
# ============================================================

n = pypsa.Network(NETWORK_FILE)

print("NETWORK")
print("-" * 110)
print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")
print()


# ============================================================
# RUN BASE AC POWER FLOW
# ============================================================

print("=" * 110)
print("BASE AC POWER FLOW")
print("=" * 110)

try:
    n.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-8,
        use_seed=True,
    )
    print("Power flow completed.")
except Exception as e:
    print(f"Power-flow warning/error: {e}")

print()


# ============================================================
# 1. BUS VOLTAGE DIAGNOSTIC
# ============================================================

print("=" * 110)
print("1. LOWEST-VOLTAGE BUSES")
print("=" * 110)

v = n.buses_t.v_mag_pu.loc[SNAPSHOT].copy()

voltage_df = pd.DataFrame({
    "bus": v.index,
    "voltage_pu": v.values,
})

voltage_df = voltage_df.sort_values("voltage_pu")

print(voltage_df.head(20).to_string(index=False))
print()


# ============================================================
# 2. WEAK BUS DETAILS
# ============================================================

print("=" * 110)
print("2. WEAK BUS DETAILS")
print("=" * 110)

if WEAK_BUS in n.buses.index:

    bus = n.buses.loc[WEAK_BUS]

    print(f"Bus ID       : {WEAK_BUS}")
    print(f"Voltage      : {v.loc[WEAK_BUS]:.6f} pu")
    print(f"Nominal kV   : {bus.get('v_nom', np.nan)}")
    print(f"Carrier      : {bus.get('carrier', '')}")
    print(f"Substation   : {bus.get('substation', '')}")
    print(f"Country      : {bus.get('country', '')}")
    print(f"x            : {bus.get('x', np.nan)}")
    print(f"y            : {bus.get('y', np.nan)}")

else:
    print(f"WARNING: Weak bus {WEAK_BUS} not found.")

print()


# ============================================================
# 3. CONNECTED LINES
# ============================================================

print("=" * 110)
print("3. LINES CONNECTED TO WEAK BUS")
print("=" * 110)

connected_lines = n.lines[
    (n.lines.bus0 == WEAK_BUS) |
    (n.lines.bus1 == WEAK_BUS)
].copy()

if len(connected_lines) == 0:
    print("No lines directly connected to weak bus.")
else:

    rows = []

    for line_id, line in connected_lines.iterrows():

        loading = np.nan

        if hasattr(n, "lines_t") and "p0" in n.lines_t:
            p0 = n.lines_t.p0.loc[SNAPSHOT, line_id]
            s_nom = line.s_nom

            if s_nom and s_nom > 0:
                loading = abs(p0) / s_nom * 100

        rows.append({
            "line": line_id,
            "bus0": line.bus0,
            "bus1": line.bus1,
            "s_nom_MW": line.s_nom,
            "r": line.r,
            "x": line.x,
            "b": line.b,
            "length": line.length,
            "loading_pct": loading,
        })

    connected_df = pd.DataFrame(rows)
    connected_df = connected_df.sort_values(
        "loading_pct",
        ascending=False,
        na_position="last",
    )

    print(connected_df.to_string(index=False))

print()


# ============================================================
# 4. TOP LOADED LINES
# ============================================================

print("=" * 110)
print("4. TOP 20 LOADED LINES")
print("=" * 110)

line_rows = []

for line_id, line in n.lines.iterrows():

    try:
        p0 = n.lines_t.p0.loc[SNAPSHOT, line_id]
        q0 = n.lines_t.q0.loc[SNAPSHOT, line_id]

        s = np.sqrt(p0**2 + q0**2)

        if line.s_nom > 0:
            loading = s / line.s_nom * 100
        else:
            loading = np.nan

    except Exception:
        p0 = np.nan
        q0 = np.nan
        loading = np.nan

    line_rows.append({
        "line": line_id,
        "bus0": line.bus0,
        "bus1": line.bus1,
        "s_nom_MW": line.s_nom,
        "p0_MW": p0,
        "q0_Mvar": q0,
        "loading_pct": loading,
    })

line_df = pd.DataFrame(line_rows)

line_df = line_df.sort_values(
    "loading_pct",
    ascending=False,
    na_position="last",
)

print(line_df.head(20).to_string(index=False))
print()


# ============================================================
# 5. TOP LOADED TRANSFORMERS
# ============================================================

print("=" * 110)
print("5. TRANSFORMER LOADING")
print("=" * 110)

transformer_rows = []

for trafo_id, trafo in n.transformers.iterrows():

    try:
        p0 = n.transformers_t.p0.loc[SNAPSHOT, trafo_id]
        q0 = n.transformers_t.q0.loc[SNAPSHOT, trafo_id]

        s = np.sqrt(p0**2 + q0**2)

        if trafo.s_nom > 0:
            loading = s / trafo.s_nom * 100
        else:
            loading = np.nan

    except Exception:
        p0 = np.nan
        q0 = np.nan
        loading = np.nan

    transformer_rows.append({
        "transformer": trafo_id,
        "bus0": trafo.bus0,
        "bus1": trafo.bus1,
        "s_nom_MW": trafo.s_nom,
        "p0_MW": p0,
        "q0_Mvar": q0,
        "loading_pct": loading,
    })

trafo_df = pd.DataFrame(transformer_rows)

trafo_df = trafo_df.sort_values(
    "loading_pct",
    ascending=False,
    na_position="last",
)

print(trafo_df.head(20).to_string(index=False))
print()


# ============================================================
# 6. GENERATORS CONNECTED TO WEAK BUS
# ============================================================

print("=" * 110)
print("6. GENERATION AT / NEAR WEAK BUS")
print("=" * 110)

weak_generators = n.generators[
    n.generators.bus == WEAK_BUS
].copy()

if len(weak_generators):

    gen_rows = []

    for gen_id, gen in weak_generators.iterrows():

        try:
            p = n.generators_t.p.loc[SNAPSHOT, gen_id]
        except Exception:
            p = np.nan

        gen_rows.append({
            "generator": gen_id,
            "bus": gen.bus,
            "carrier": gen.carrier,
            "p_nom_MW": gen.p_nom,
            "dispatch_MW": p,
        })

    print(pd.DataFrame(gen_rows).to_string(index=False))

else:
    print("No generators directly connected to weak bus.")

print()


# ============================================================
# 7. LOAD CONNECTED TO WEAK BUS
# ============================================================

print("=" * 110)
print("7. LOAD AT WEAK BUS")
print("=" * 110)

weak_loads = n.loads[
    n.loads.bus == WEAK_BUS
].copy()

if len(weak_loads):

    load_rows = []

    for load_id, load in weak_loads.iterrows():

        try:
            p = n.loads_t.p_set.loc[SNAPSHOT, load_id]
        except Exception:
            p = np.nan

        try:
            q = n.loads_t.q_set.loc[SNAPSHOT, load_id]
        except Exception:
            q = np.nan

        load_rows.append({
            "load": load_id,
            "bus": load.bus,
            "p_MW": p,
            "q_Mvar": q,
        })

    print(pd.DataFrame(load_rows).to_string(index=False))

else:
    print("No loads directly connected to weak bus.")

print()


# ============================================================
# 8. SUMMARY
# ============================================================

print("=" * 110)
print("S3.6 SUMMARY")
print("=" * 110)

weak_voltage = (
    float(v.loc[WEAK_BUS])
    if WEAK_BUS in v.index
    else np.nan
)

max_line_loading = (
    float(line_df.iloc[0]["loading_pct"])
    if len(line_df)
    else np.nan
)

max_trafo_loading = (
    float(trafo_df.iloc[0]["loading_pct"])
    if len(trafo_df)
    else np.nan
)

print(f"Weak bus                 : {WEAK_BUS}")
print(f"Weak-bus voltage         : {weak_voltage:.6f} pu")
print(f"Maximum line loading     : {max_line_loading:.3f} %")
print(f"Maximum transformer load : {max_trafo_loading:.3f} %")
print()

print("Top 5 overloaded lines:")
print(
    line_df[
        ["line", "bus0", "bus1", "s_nom_MW", "loading_pct"]
    ].head(5).to_string(index=False)
)

print()


# ============================================================
# SAVE RESULTS
# ============================================================

summary_rows = []

for _, row in voltage_df.head(20).iterrows():

    summary_rows.append({
        "analysis": "weak_bus_voltage",
        "element": row["bus"],
        "value_1": row["voltage_pu"],
        "value_2": np.nan,
        "value_3": np.nan,
    })

for _, row in line_df.head(20).iterrows():

    summary_rows.append({
        "analysis": "top_loaded_line",
        "element": row["line"],
        "value_1": row["loading_pct"],
        "value_2": row["s_nom_MW"],
        "value_3": row["p0_MW"],
    })

for _, row in trafo_df.head(20).iterrows():

    summary_rows.append({
        "analysis": "transformer_loading",
        "element": row["transformer"],
        "value_1": row["loading_pct"],
        "value_2": row["s_nom_MW"],
        "value_3": row["p0_MW"],
    })

result_df = pd.DataFrame(summary_rows)

result_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 110)
print("S3.6 COMPLETE")
print("=" * 110)
print(f"Results saved to:")
print(OUTPUT_FILE)