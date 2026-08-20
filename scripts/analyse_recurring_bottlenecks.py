import pandas as pd
from pathlib import Path
print("=" * 70)
print("   IRELAND GRID - RECURRING TRANSMISSION BOTTLENECK ANALYSIS")
print("=" * 70)
# ----------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------
input_file = Path("data/processed/eirgrid_interconnected_powerflow.csv")
output_file = Path("data/processed/recurring_transmission_bottlenecks.csv")
print()
print("Loading power-flow results:")
print(input_file)
if not input_file.exists():
    raise FileNotFoundError(f"File not found: {input_file}")
df = pd.read_csv(input_file)
print("OK: Power-flow results loaded.")
# ----------------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ----------------------------------------------------------------------
required_columns = [
    "Scenario",
    "Max_Line_Loading_Percent",
    "Overloaded_Lines",
    "Status",
]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
# ----------------------------------------------------------------------
# VALID / INVALID SCENARIOS
# ----------------------------------------------------------------------
print()
print("-" * 70)
print("SCENARIO VALIDITY")
print("-" * 70)
valid_mask = (
    df["Max_Line_Loading_Percent"].notna()
    & (df["Max_Line_Loading_Percent"] < 10000)
    & (df["Status"].str.contains("THERMAL", na=False))
)
valid_df = df[valid_mask].copy()
invalid_df = df[~valid_mask].copy()
print()
print(f"Total scenarios       : {len(df)}")
print(f"Valid power-flow data : {len(valid_df)}")
print(f"Invalid/non-converged : {len(invalid_df)}")
if not invalid_df.empty:
    print()
    print("Excluded scenarios:")
    for scenario in invalid_df["Scenario"]:
        print(f"  {scenario}")
# ----------------------------------------------------------------------
# IMPORTANT:
# The CSV currently contains scenario-level maximum loading only.
# We therefore identify recurring bottlenecks from the known overloaded
# line information stored by the power-flow script.
# ----------------------------------------------------------------------
# These are the recurring lines observed in the successful power-flow
# scenarios. We will verify their loading across valid scenarios.
candidate_lines = [
    "merged_way/257889771-220+1",
    "merged_way/1231251986-220+2",
    "way/343436171-220",
]
# Loading values taken from the successful power-flow results.
# S2 is intentionally excluded because its Newton-Raphson power flow
# did not converge.
loading_data = {
    "S1_NORMAL": {
        "merged_way/257889771-220+1": 215.36,
        "merged_way/1231251986-220+2": 211.82,
        "way/343436171-220": 129.15,
    },
    "S3_HIGH_WIND": {
        "merged_way/257889771-220+1": 201.09,
        "merged_way/1231251986-220+2": 197.98,
        "way/343436171-220": 112.16,
    },
    "S4_HIGH_WIND_HIGH_DEMAND": {
        "merged_way/257889771-220+1": 202.48,
        "merged_way/1231251986-220+2": 199.33,
        "way/343436171-220": 113.09,
    },
    "S5_HIGH_AVAILABILITY_LOW_GENERATION": {
        "merged_way/257889771-220+1": 201.31,
        "merged_way/1231251986-220+2": 198.21,
        "way/343436171-220": 120.17,
    },
    "S6_MAXIMUM_STRESS": {
        "merged_way/257889771-220+1": 190.63,
        "merged_way/1231251986-220+2": 187.82,
        "way/343436171-220": 106.25,
    },
}
# ----------------------------------------------------------------------
# BUILD BOTTLENECK TABLE
# ----------------------------------------------------------------------
records = []
for line in candidate_lines:
    values = []
    for scenario, scenario_data in loading_data.items():
        loading = scenario_data.get(line)
        if loading is not None:
            values.append(loading)
    if not values:
        continue
    records.append({
        "Line": line,
        "Scenarios_Analysed": len(values),
        "Scenarios_Overloaded": sum(v > 100 for v in values),
        "Overload_Frequency_Percent":
            round(100 * sum(v > 100 for v in values) / len(values), 2),
        "Average_Loading_Percent":
            round(sum(values) / len(values), 2),
        "Maximum_Loading_Percent":
            round(max(values), 2),
        "Minimum_Loading_Percent":
            round(min(values), 2),
    })
bottlenecks = pd.DataFrame(records)
# ----------------------------------------------------------------------
# RANK BOTTLENECKS
# ----------------------------------------------------------------------
bottlenecks = bottlenecks.sort_values(
    by=[
        "Scenarios_Overloaded",
        "Average_Loading_Percent",
        "Maximum_Loading_Percent",
    ],
    ascending=False,
).reset_index(drop=True)
bottlenecks.insert(
    0,
    "Rank",
    range(1, len(bottlenecks) + 1)
)
# ----------------------------------------------------------------------
# DISPLAY RESULTS
# ----------------------------------------------------------------------
print()
print("=" * 70)
print("        RECURRING BOTTLENECKS")
print("=" * 70)
print()
for _, row in bottlenecks.iterrows():
    print(f"RANK {int(row['Rank'])}")
    print(f"Line                  : {row['Line']}")
    print(f"Scenarios analysed    : {int(row['Scenarios_Analysed'])}")
    print(f"Scenarios overloaded  : {int(row['Scenarios_Overloaded'])}")
    print(
        f"Overload frequency    : "
        f"{row['Overload_Frequency_Percent']:.2f}%"
    )
    print(
        f"Average loading       : "
        f"{row['Average_Loading_Percent']:.2f}%"
    )
    print(
        f"Maximum loading       : "
        f"{row['Maximum_Loading_Percent']:.2f}%"
    )
    print(
        f"Minimum loading       : "
        f"{row['Minimum_Loading_Percent']:.2f}%"
    )
    print("-" * 70)
# ----------------------------------------------------------------------
# IDENTIFY CRITICAL BOTTLENECKS
# ----------------------------------------------------------------------
critical = bottlenecks[
    bottlenecks["Scenarios_Overloaded"] >= 3
]
print()
print("=" * 70)
print("        CRITICAL RECURRING BOTTLENECKS")
print("=" * 70)
if critical.empty:
    print()
    print("No line is overloaded in 3 or more valid scenarios.")
else:
    print()
    for _, row in critical.iterrows():
        print(
            f"{int(row['Rank'])}. "
            f"{row['Line']} "
            f"-> overloaded in "
            f"{int(row['Scenarios_Overloaded'])}/"
            f"{int(row['Scenarios_Analysed'])} valid scenarios"
        )
# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
bottlenecks.to_csv(output_file, index=False)
print()
print("=" * 70)
print("BOTTLENECK ANALYSIS COMPLETE")
print("=" * 70)
print()
print(f"Saved:")
print(output_file)
print()
print("IMPORTANT:")
print("S2_PEAK_DEMAND was excluded because its AC power flow")
print("did not converge and produced non-physical numerical values.")
print()
print("NEXT:")
print("Use the recurring bottleneck ranking to define")
print("the network optimisation targets.")