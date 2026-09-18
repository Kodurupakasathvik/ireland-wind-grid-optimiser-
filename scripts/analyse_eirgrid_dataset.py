import pandas as pd
from pathlib import Path

FILE = Path("data/processed/eirgrid_ireland_2026.csv")

print("=" * 70)
print("       EIRGRID IRELAND DATASET - STATISTICAL ANALYSIS")
print("=" * 70)

print()
print("Loading:")
print(FILE)

df = pd.read_csv(FILE, parse_dates=["DateTime"])

print("OK: Dataset loaded.")

print()
print("-" * 70)
print("1. DATASET")
print("-" * 70)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Start             : {df['DateTime'].min()}")
print(f"End               : {df['DateTime'].max()}")
print(f"Time span         : {(df['DateTime'].max() - df['DateTime'].min()).days} days")

print()
print("-" * 70)
print("2. IRELAND DEMAND")
print("-" * 70)

demand = df["IE Demand"]

print(f"Minimum demand    : {demand.min():,.2f} MW")
print(f"Average demand    : {demand.mean():,.2f} MW")
print(f"Maximum demand    : {demand.max():,.2f} MW")

max_demand_row = df.loc[demand.idxmax()]

print()
print("Peak demand event:")
print(f"  Time            : {max_demand_row['DateTime']}")
print(f"  Demand          : {max_demand_row['IE Demand']:,.2f} MW")
print(f"  Wind generation : {max_demand_row['IE Wind Generation']:,.2f} MW")
print(f"  Wind available  : {max_demand_row['IE Wind Availability']:,.2f} MW")

print()
print("-" * 70)
print("3. WIND GENERATION")
print("-" * 70)

wind = df["IE Wind Generation"]
wind_avail = df["IE Wind Availability"]

print(f"Minimum wind      : {wind.min():,.2f} MW")
print(f"Average wind      : {wind.mean():,.2f} MW")
print(f"Maximum wind      : {wind.max():,.2f} MW")

max_wind_row = df.loc[wind.idxmax()]

print()
print("Peak wind event:")
print(f"  Time            : {max_wind_row['DateTime']}")
print(f"  Wind generation : {max_wind_row['IE Wind Generation']:,.2f} MW")
print(f"  Wind available  : {max_wind_row['IE Wind Availability']:,.2f} MW")
print(f"  IE demand       : {max_wind_row['IE Demand']:,.2f} MW")

print()
print("-" * 70)
print("4. WIND AVAILABILITY VS GENERATION")
print("-" * 70)

df["Wind_Not_Generated"] = (
    df["IE Wind Availability"] - df["IE Wind Generation"]
)

df["Wind_Not_Generated"] = df["Wind_Not_Generated"].clip(lower=0)

print(
    f"Average available wind       : "
    f"{wind_avail.mean():,.2f} MW"
)

print(
    f"Average wind not generated   : "
    f"{df['Wind_Not_Generated'].mean():,.2f} MW"
)

print(
    f"Maximum wind not generated   : "
    f"{df['Wind_Not_Generated'].max():,.2f} MW"
)

print(
    f"Total wind availability      : "
    f"{wind_avail.sum():,.2f} MW-quarter"
)

print(
    f"Total wind generation        : "
    f"{wind.sum():,.2f} MW-quarter"
)

print()
print("-" * 70)
print("5. WIND PENETRATION")
print("-" * 70)

penetration = df["IE Wind Penetration"]

print(f"Minimum            : {penetration.min()*100:.2f}%")
print(f"Average            : {penetration.mean()*100:.2f}%")
print(f"Maximum            : {penetration.max()*100:.2f}%")

print()
print("-" * 70)
print("6. SOLAR")
print("-" * 70)

solar = df["IE Solar Generation"]

print(f"Minimum solar      : {solar.min():,.2f} MW")
print(f"Average solar      : {solar.mean():,.2f} MW")
print(f"Maximum solar      : {solar.max():,.2f} MW")

print()
print("-" * 70)
print("7. INTERCONNECTORS")
print("-" * 70)

for col in ["EWIC I/C", "Greenlink I/C"]:

    print()
    print(col)

    print(f"  Minimum          : {df[col].min():,.2f} MW")
    print(f"  Average          : {df[col].mean():,.2f} MW")
    print(f"  Maximum          : {df[col].max():,.2f} MW")

print()
print("-" * 70)
print("8. SNSP")
print("-" * 70)

snsp = df["SNSP"]

print(f"Minimum            : {snsp.min()*100:.2f}%")
print(f"Average            : {snsp.mean()*100:.2f}%")
print(f"Maximum            : {snsp.max()*100:.2f}%")

print()
print("-" * 70)
print("9. HIGH WIND PERIODS")
print("-" * 70)

threshold = wind.quantile(0.90)

high_wind = df[wind >= threshold]

print(
    f"90th percentile wind threshold : "
    f"{threshold:,.2f} MW"
)

print(
    f"Periods above threshold       : "
    f"{len(high_wind):,}"
)

print(
    f"Average demand during periods : "
    f"{high_wind['IE Demand'].mean():,.2f} MW"
)

print(
    f"Average wind availability     : "
    f"{high_wind['IE Wind Availability'].mean():,.2f} MW"
)

print(
    f"Average wind generation       : "
    f"{high_wind['IE Wind Generation'].mean():,.2f} MW"
)

print(
    f"Average wind not generated    : "
    f"{high_wind['Wind_Not_Generated'].mean():,.2f} MW"
)

print()
print("-" * 70)
print("10. HIGHEST WIND EVENTS")
print("-" * 70)

top = df.nlargest(10, "IE Wind Generation")[
    [
        "DateTime",
        "IE Demand",
        "IE Wind Availability",
        "IE Wind Generation",
        "IE Wind Penetration",
        "SNSP",
    ]
]

print(top.to_string(index=False))

print()
print("-" * 70)
print("11. HIGH WIND AVAILABILITY BUT LOWER GENERATION")
print("-" * 70)

top_not_generated = df.nlargest(10, "Wind_Not_Generated")[
    [
        "DateTime",
        "IE Demand",
        "IE Wind Availability",
        "IE Wind Generation",
        "Wind_Not_Generated",
        "IE Wind Penetration",
        "SNSP",
    ]
]

print(top_not_generated.to_string(index=False))

print()
print("=" * 70)
print("                 ANALYSIS COMPLETE")
print("=" * 70)

print()
print("The processed EirGrid dataset is ready for")
print("selection of real operating snapshots.")

print()
print("Next project stage:")
print("REAL EIRGRID DATA -> REAL OPERATING SCENARIOS -> PyPSA")
print("=" * 70)