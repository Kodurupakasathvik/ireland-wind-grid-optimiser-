import os
import pandas as pd

FILE = "data/raw/System-Data-Qtr-Hourly-2026-V7.xlsx"

print("=" * 70)
print("       EIRGRID SYSTEM DATA - DATASET INSPECTION")
print("=" * 70)

print(f"\nFile:")
print(f"  {FILE}")

if not os.path.exists(FILE):
    print("\nERROR: File not found.")
    raise SystemExit(1)

print("\nOK: File found.")

# ------------------------------------------------------------
# 1. WORKBOOK SHEETS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. WORKBOOK SHEETS")
print("-" * 70)

xls = pd.ExcelFile(FILE)

print(f"Number of sheets: {len(xls.sheet_names)}")

for i, sheet in enumerate(xls.sheet_names, 1):
    print(f"  {i}. {sheet}")

# ------------------------------------------------------------
# 2. INSPECT EACH SHEET
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. SHEET STRUCTURE")
print("-" * 70)

for sheet in xls.sheet_names:

    print("\n" + "=" * 70)
    print(f"SHEET: {sheet}")
    print("=" * 70)

    try:
        df = pd.read_excel(FILE, sheet_name=sheet)

        print(f"Rows    : {len(df):,}")
        print(f"Columns : {len(df.columns)}")

        print("\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        print("\nFirst 5 rows:")
        print(df.head().to_string())

        print("\nData types:")
        print(df.dtypes.to_string())

        missing = df.isna().sum()

        print("\nMissing values:")
        missing_found = False

        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count:,}")
                missing_found = True

        if not missing_found:
            print("  None")

    except Exception as e:
        print(f"FAILED to inspect sheet: {e}")

# ------------------------------------------------------------
# 3. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("                 INSPECTION COMPLETE")
print("=" * 70)

print("""
IMPORTANT:
This script only INSPECTS the EirGrid workbook.

It does not:
- modify the original file
- clean the data
- delete rows
- rename columns
- create PyPSA scenarios
- alter the network

The next step will be decided from the actual workbook structure.
""")