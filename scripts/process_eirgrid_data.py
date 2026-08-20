import os
import pandas as pd

INPUT = "data/raw/System-Data-Qtr-Hourly-2026-V7.xlsx"
OUTPUT = "data/processed/eirgrid_ireland_2026.csv"

print("=" * 70)
print("       EIRGRID DATA - IRELAND DATA PROCESSING")
print("=" * 70)

print("\nLoading:")
print(INPUT)

df = pd.read_excel(INPUT, sheet_name="System Data")

print(f"Original rows    : {len(df):,}")
print(f"Original columns : {len(df.columns)}")

# ------------------------------------------------------------
# 1. Select only variables required for the Ireland model
# ------------------------------------------------------------

columns = [
    "DateTime",
    "GMT Offset",
    "IE Demand",
    "IE Generation",
    "IE Wind Availability",
    "IE Wind Generation",
    "IE Solar Availability",
    "IE Solar Generation",
    "IE Hydro",
    "EWIC I/C",
    "Greenlink I/C",
    "IE Wind Penetration",
    "IE Solar Penetration",
    "AI Demand",
    "AI Generation",
    "AI Wind Generation",
    "AI Wind Availability",
    "AI Oversupply",
    "AI Oversupply Percentage",
    "SNSP",
]

df = df[columns].copy()

print("\nSelected Ireland/system variables:")
for col in df.columns:
    print(f"  {col}")

# ------------------------------------------------------------
# 2. Remove completely empty rows
# ------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=[
        "DateTime",
        "IE Demand",
        "IE Generation",
        "IE Wind Generation",
    ]
)

print("\nRows after removing invalid records:")
print(f"  Before : {before:,}")
print(f"  After  : {len(df):,}")
print(f"  Removed: {before - len(df):,}")

# ------------------------------------------------------------
# 3. Ensure correct datetime type
# ------------------------------------------------------------

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

invalid_dates = df["DateTime"].isna().sum()

print("\nDatetime validation:")
print(f"  Invalid timestamps: {invalid_dates}")

df = df.dropna(subset=["DateTime"])

# ------------------------------------------------------------
# 4. Sort chronologically
# ------------------------------------------------------------

df = df.sort_values("DateTime").reset_index(drop=True)

# ------------------------------------------------------------
# 5. Check duplicate timestamps
# ------------------------------------------------------------

duplicates = df["DateTime"].duplicated().sum()

print("\nTimestamp validation:")
print(f"  Duplicate timestamps: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates(
        subset="DateTime",
        keep="first"
    )

# ------------------------------------------------------------
# 6. Create useful derived variables
# ------------------------------------------------------------

df["IE_Wind_Curtailment_Est"] = (
    df["IE Wind Availability"] -
    df["IE Wind Generation"]
)

# Avoid tiny negative values caused by measurement/rounding
df["IE_Wind_Curtailment_Est"] = (
    df["IE_Wind_Curtailment_Est"].clip(lower=0)
)

df["IE_Wind_Curtailment_Pct"] = 0.0

valid_availability = df["IE Wind Availability"] > 0

df.loc[valid_availability, "IE_Wind_Curtailment_Pct"] = (
    df.loc[valid_availability, "IE_Wind_Curtailment_Est"]
    / df.loc[valid_availability, "IE Wind Availability"]
    * 100
)

# ------------------------------------------------------------
# 7. Create time features
# ------------------------------------------------------------

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["QuarterHour"] = df["DateTime"].dt.minute

# ------------------------------------------------------------
# 8. Basic sanity checks
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SANITY CHECKS")
print("-" * 70)

numeric_columns = [
    "IE Demand",
    "IE Generation",
    "IE Wind Availability",
    "IE Wind Generation",
    "IE Solar Generation",
    "IE Hydro",
    "EWIC I/C",
    "Greenlink I/C",
    "IE Wind Penetration",
    "SNSP",
]

for col in numeric_columns:

    negative = (df[col] < 0).sum()

    print(
        f"{col:30s} "
        f"min={df[col].min():10.3f} "
        f"max={df[col].max():10.3f} "
        f"negative={negative}"
    )

# ------------------------------------------------------------
# 9. Date coverage
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("DATE COVERAGE")
print("-" * 70)

print(f"Start : {df['DateTime'].min()}")
print(f"End   : {df['DateTime'].max()}")
print(f"Rows  : {len(df):,}")

# ------------------------------------------------------------
# 10. Save processed dataset
# ------------------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(f"  {OUTPUT}")

print(f"\nFinal rows    : {len(df):,}")
print(f"Final columns : {len(df.columns)}")

print("\nIMPORTANT:")
print("The original EirGrid workbook was not modified.")
print("This CSV is the cleaned operational dataset for the project.")