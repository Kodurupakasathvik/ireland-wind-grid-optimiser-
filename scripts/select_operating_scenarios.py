import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/eirgrid_ireland_2026.csv")
OUTPUT = Path("data/processed/selected_operating_scenarios.csv")

print("=" * 70)
print("       IRELAND GRID - OPERATING SCENARIO SELECTION")
print("=" * 70)

print()
print("Loading:")
print(INPUT)

df = pd.read_csv(INPUT, parse_dates=["DateTime"])

print(f"OK: Loaded {len(df):,} records.")

# ------------------------------------------------------------
# Derived variables
# ------------------------------------------------------------

df["Wind_Not_Generated"] = (
    df["IE Wind Availability"] - df["IE Wind Generation"]
).clip(lower=0)

df["Wind_Penetration_Pct"] = (
    df["IE Wind Generation"] / df["IE Demand"] * 100
)

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

def select_closest(data, column, target):
    idx = (data[column] - target).abs().idxmin()
    return data.loc[idx]

# ------------------------------------------------------------
# 1. NORMAL OPERATING CONDITION
# ------------------------------------------------------------

normal_target_demand = df["IE Demand"].median()
normal_target_wind = df["IE Wind Generation"].median()

normal = df.copy()

normal["distance"] = (
    (normal["IE Demand"] - normal_target_demand).abs()
    / df["IE Demand"].std()
    +
    (normal["IE Wind Generation"] - normal_target_wind).abs()
    / df["IE Wind Generation"].std()
)

normal_row = normal.loc[normal["distance"].idxmin()]

# ------------------------------------------------------------
# 2. PEAK DEMAND
# ------------------------------------------------------------

peak_demand_row = df.loc[df["IE Demand"].idxmax()]

# ------------------------------------------------------------
# 3. HIGH WIND
# ------------------------------------------------------------

high_wind_row = df.loc[df["IE Wind Generation"].idxmax()]

# ------------------------------------------------------------
# 4. HIGH WIND + HIGH DEMAND
#
# Must be a genuinely different operating point from S3.
# We therefore exclude the S3 timestamp and select the
# strongest combined demand + wind condition remaining.
# ------------------------------------------------------------

demand_threshold = df["IE Demand"].quantile(0.90)
wind_threshold = df["IE Wind Generation"].quantile(0.90)

high_wind_high_demand = df[
    (df["IE Demand"] >= demand_threshold)
    &
    (df["IE Wind Generation"] >= wind_threshold)
    &
    (df["DateTime"] != high_wind_row["DateTime"])
].copy()

if len(high_wind_high_demand) > 0:

    high_wind_high_demand["stress_score"] = (
        0.5
        * high_wind_high_demand["IE Demand"]
        / df["IE Demand"].max()
        +
        0.5
        * high_wind_high_demand["IE Wind Generation"]
        / df["IE Wind Generation"].max()
    )

    stress_row = high_wind_high_demand.loc[
        high_wind_high_demand["stress_score"].idxmax()
    ]

else:

    raise RuntimeError(
        "No distinct high-wind/high-demand operating point "
        "was found after excluding the S3 timestamp."
    )

# ------------------------------------------------------------
# 5. HIGH AVAILABILITY / LOW GENERATION
# ------------------------------------------------------------

dispatch_down_row = df.loc[df["Wind_Not_Generated"].idxmax()]

# ------------------------------------------------------------
# 6. MAXIMUM COMBINED STRESS
# ------------------------------------------------------------

df["stress_score"] = (
    df["IE Demand"] / df["IE Demand"].max()
    +
    df["IE Wind Availability"] / df["IE Wind Availability"].max()
    +
    df["SNSP"] / df["SNSP"].max()
)

maximum_stress_row = df.loc[df["stress_score"].idxmax()]

# ------------------------------------------------------------
# Build scenario table
# ------------------------------------------------------------

scenarios = [
    ("S1_NORMAL", normal_row),
    ("S2_PEAK_DEMAND", peak_demand_row),
    ("S3_HIGH_WIND", high_wind_row),
    ("S4_HIGH_WIND_HIGH_DEMAND", stress_row),
    ("S5_HIGH_AVAILABILITY_LOW_GENERATION", dispatch_down_row),
    ("S6_MAXIMUM_STRESS", maximum_stress_row),
]

records = []

print()
print("-" * 70)
print("SELECTED OPERATING SCENARIOS")
print("-" * 70)

for scenario_name, row in scenarios:

    record = {
        "Scenario": scenario_name,
        "DateTime": row["DateTime"],
        "IE_Demand_MW": row["IE Demand"],
        "IE_Generation_MW": row["IE Generation"],
        "IE_Wind_Availability_MW": row["IE Wind Availability"],
        "IE_Wind_Generation_MW": row["IE Wind Generation"],
        "Wind_Not_Generated_MW": row["Wind_Not_Generated"],
        "IE_Solar_Generation_MW": row["IE Solar Generation"],
        "IE_Hydro_MW": row["IE Hydro"],
        "EWIC_MW": row["EWIC I/C"],
        "Greenlink_MW": row["Greenlink I/C"],
        "IE_Wind_Penetration": row["IE Wind Penetration"],
        "SNSP": row["SNSP"],
    }

    records.append(record)

    print()
    print(scenario_name)
    print(f"  Time                 : {row['DateTime']}")
    print(f"  Demand               : {row['IE Demand']:,.2f} MW")
    print(f"  Wind availability    : {row['IE Wind Availability']:,.2f} MW")
    print(f"  Wind generation      : {row['IE Wind Generation']:,.2f} MW")
    print(f"  Wind not generated   : {row['Wind_Not_Generated']:,.2f} MW")
    print(f"  Solar                : {row['IE Solar Generation']:,.2f} MW")
    print(f"  Hydro                : {row['IE Hydro']:,.2f} MW")
    print(f"  Wind penetration     : {row['IE Wind Penetration'] * 100:.2f}%")
    print(f"  SNSP                 : {row['SNSP'] * 100:.2f}%")

scenario_df = pd.DataFrame(records)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

scenario_df.to_csv(OUTPUT, index=False)

print()
print("=" * 70)
print("                 SCENARIO SELECTION COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(OUTPUT)

print()
print(f"Scenarios created : {len(scenario_df)}")

print()
print("Next stage:")
print("REAL EIRGRID SCENARIOS -> MAP INTO PYPSA -> POWER FLOW")
print("=" * 70)