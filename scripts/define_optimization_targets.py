import pandas as pd
from pathlib import Path

print("=" * 70)
print("       IRELAND GRID - OPTIMIZATION TARGET DEFINITION")
print("=" * 70)

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/recurring_transmission_bottlenecks.csv"
)

OUTPUT_FILE = Path(
    "data/processed/optimization_targets.csv"
)

# ----------------------------------------------------------------------
# LOAD BOTTLENECK RESULTS
# ----------------------------------------------------------------------

print("\nLoading recurring bottleneck analysis:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Required file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("OK: Bottleneck data loaded.")

# ----------------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ----------------------------------------------------------------------

required_columns = [
    "Line",
    "Scenarios_Analysed",
    "Scenarios_Overloaded",
    "Overload_Frequency_Percent",
    "Average_Loading_Percent",
    "Maximum_Loading_Percent",
    "Minimum_Loading_Percent",
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )

# ----------------------------------------------------------------------
# DEFINE OPTIMIZATION PRIORITY
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("OPTIMIZATION TARGETS")
print("-" * 70)

# Only recurring bottlenecks are optimization targets.
targets = df[
    (df["Overload_Frequency_Percent"] >= 80.0)
].copy()

# Priority score:
# Higher overload frequency + higher average loading
# means higher optimization priority.

targets["Priority_Score"] = (
    targets["Overload_Frequency_Percent"]
    * targets["Average_Loading_Percent"]
    / 100.0
)

targets = targets.sort_values(
    by="Priority_Score",
    ascending=False
).reset_index(drop=True)

targets["Optimization_Rank"] = (
    targets.index + 1
)

# ----------------------------------------------------------------------
# CLASSIFY PRIORITY
# ----------------------------------------------------------------------

def classify_priority(score):

    if score >= 180:
        return "CRITICAL"

    elif score >= 120:
        return "HIGH"

    else:
        return "MODERATE"


targets["Priority"] = targets["Priority_Score"].apply(
    classify_priority
)

# ----------------------------------------------------------------------
# DISPLAY RESULTS
# ----------------------------------------------------------------------

for _, row in targets.iterrows():

    print(f"\nRANK {int(row['Optimization_Rank'])}")

    print(f"Line                  : {row['Line']}")

    print(
        f"Scenarios analysed    : "
        f"{int(row['Scenarios_Analysed'])}"
    )

    print(
        f"Scenarios overloaded  : "
        f"{int(row['Scenarios_Overloaded'])}"
    )

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

    print(
        f"Priority score        : "
        f"{row['Priority_Score']:.2f}"
    )

    print(
        f"Priority              : "
        f"{row['Priority']}"
    )

# ----------------------------------------------------------------------
# SAVE OPTIMIZATION TARGETS
# ----------------------------------------------------------------------

targets.to_csv(
    OUTPUT_FILE,
    index=False
)

# ----------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("        OPTIMIZATION TARGETS DEFINED")
print("=" * 70)

print(f"\nTargets identified : {len(targets)}")

print("\nOptimization targets:")

for _, row in targets.iterrows():

    print(
        f"  {int(row['Optimization_Rank'])}. "
        f"{row['Line']} -> {row['Priority']}"
    )

print("\nSaved:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print(
    "These targets define where network optimization "
    "should focus."
)

print(
    "The original network has NOT been modified."
)

print("\nNEXT:")
print(
    "Build the constrained network optimization model "
    "using these targets."
)