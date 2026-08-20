"""
======================================================================
IRELAND GRID - SCENARIO CONSTRUCTION INSPECTOR
======================================================================

Purpose:
    Locate and inspect the code/data responsible for constructing
    scenario dispatch values.

This script:
    - DOES NOT modify any network
    - DOES NOT reinforce anything
    - DOES NOT run reinforcement
    - DOES NOT overwrite scenario data
    - Searches the project for scenario definitions and assignments
    - Identifies where S1-S6 values may be constructed
    - Checks Python source files and processed CSV/JSON files
    - Prints likely scenario-construction locations

Primary issue being investigated:
    S2_PEAK_DEMAND contains NaN/zero generator and load dispatch.

======================================================================
"""

from pathlib import Path
import re
import json
import csv


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


SCENARIOS = [
    "S1_NORMAL",
    "S2_PEAK_DEMAND",
    "S3_HIGH_WIND",
    "S4_HIGH_WIND_HIGH_DEMAND",
    "S5_HIGH_AVAILABILITY_LOW_GENERATION",
    "S6_MAXIMUM_STRESS",
]


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

print("=" * 70)
print("IRELAND GRID - SCENARIO CONSTRUCTION INSPECTOR")
print("=" * 70)

print()
print("INSPECTION MODE")
print()
print("This script:")
print("  - DOES NOT modify networks")
print("  - DOES NOT reinforce lines")
print("  - DOES NOT overwrite scenario data")
print("  - Searches project files for scenario construction")
print("  - Identifies likely S1-S6 scenario definitions")
print("  - Looks for generator/load dispatch assignments")
print()

print(f"Project root : {PROJECT_ROOT}")
print(f"Scripts      : {SCRIPTS_DIR}")
print(f"Processed    : {PROCESSED_DIR}")
print(f"Raw data     : {RAW_DIR}")


# ---------------------------------------------------------------------
# FILE COLLECTION
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("COLLECTING PROJECT FILES")
print("=" * 70)

source_extensions = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".txt",
    ".md",
}

files = []

for directory in [SCRIPTS_DIR, RAW_DIR, PROCESSED_DIR]:

    if not directory.exists():
        print(f"WARNING: Directory does not exist: {directory}")
        continue

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in source_extensions:
            continue

        files.append(path)

files = sorted(set(files))

print(f"Files inspected: {len(files)}")


# ---------------------------------------------------------------------
# SEARCH PATTERNS
# ---------------------------------------------------------------------

patterns = {

    "scenario_names": [
        r"S1_NORMAL",
        r"S2_PEAK_DEMAND",
        r"S3_HIGH_WIND",
        r"S4_HIGH_WIND_HIGH_DEMAND",
        r"S5_HIGH_AVAILABILITY_LOW_GENERATION",
        r"S6_MAXIMUM_STRESS",
    ],

    "snapshot_assignment": [
        r"\.snapshots\s*=",
        r"set_snapshots",
        r"snapshots",
    ],

    "generator_dispatch": [
        r"generators_t",
        r"p_set",
        r"p_max_pu",
        r"p_min_pu",
        r"p_nom",
        r"generat",
    ],

    "load_dispatch": [
        r"loads_t",
        r"loads",
        r"q_set",
        r"load",
    ],

    "scenario_construction": [
        r"scenario",
        r"SCENARIO",
        r"peak",
        r"wind",
        r"demand",
        r"availability",
        r"stress",
        r"dispatch",
    ],

    "nan_handling": [
        r"NaN",
        r"nan",
        r"fillna",
        r"isna",
        r"notna",
        r"dropna",
    ],
}


# ---------------------------------------------------------------------
# SEARCH SOURCE FILES
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("SEARCHING SOURCE/DATA FILES")
print("=" * 70)


matches = []


for path in files:

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        continue

    lines = text.splitlines()

    for category, regexes in patterns.items():

        for regex in regexes:

            try:
                compiled = re.compile(regex, re.IGNORECASE)
            except re.error:
                continue

            for line_number, line in enumerate(lines, start=1):

                if compiled.search(line):

                    matches.append({
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "line": line_number,
                        "category": category,
                        "pattern": regex,
                        "text": line.strip(),
                    })


# ---------------------------------------------------------------------
# GROUP MATCHES
# ---------------------------------------------------------------------

categories = [
    "scenario_names",
    "scenario_construction",
    "generator_dispatch",
    "load_dispatch",
    "snapshot_assignment",
    "nan_handling",
]


for category in categories:

    print()
    print("-" * 70)
    print(category.upper())
    print("-" * 70)

    category_matches = [
        m for m in matches
        if m["category"] == category
    ]

    if not category_matches:
        print("No matches found.")
        continue

    seen = set()

    count = 0

    for match in category_matches:

        key = (
            match["file"],
            match["line"],
            match["text"],
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            f"{match['file']}:{match['line']}"
        )
        print(
            f"    {match['text']}"
        )

        count += 1

        # Avoid producing an enormous console dump.
        if count >= 100:
            print("    ... output truncated ...")
            break


# ---------------------------------------------------------------------
# S2-SPECIFIC SEARCH
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("S2_PEAK_DEMAND SPECIFIC SEARCH")
print("=" * 70)

s2_matches = []

for path in files:

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    lines = text.splitlines()

    for line_number, line in enumerate(lines, start=1):

        if "S2_PEAK_DEMAND" in line:

            s2_matches.append(
                (
                    path.relative_to(PROJECT_ROOT),
                    line_number,
                    line.strip(),
                )
            )


if not s2_matches:

    print("NO DIRECT S2_PEAK_DEMAND REFERENCES FOUND.")

else:

    for file_name, line_number, line in s2_matches:

        print()
        print(f"{file_name}:{line_number}")
        print(f"    {line}")


# ---------------------------------------------------------------------
# SEARCH FOR SCENARIO DICTIONARIES
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("POSSIBLE SCENARIO DICTIONARIES / TABLES")
print("=" * 70)

dictionary_keywords = [
    "SCENARIOS",
    "SCENARIO_DATA",
    "scenario_data",
    "scenario_config",
    "scenario_configuration",
    "SCENARIO_CONFIG",
    "PEAK_DEMAND",
    "HIGH_WIND",
    "MAXIMUM_STRESS",
]


for path in files:

    if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml"}:
        continue

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    found = False

    for keyword in dictionary_keywords:

        if keyword.lower() in text.lower():
            found = True
            break

    if found:

        print()
        print(
            f"LIKELY SCENARIO FILE: "
            f"{path.relative_to(PROJECT_ROOT)}"
        )


# ---------------------------------------------------------------------
# CSV INSPECTION
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("CSV SCENARIO DATA INSPECTION")
print("=" * 70)


csv_files = list(
    PROCESSED_DIR.rglob("*.csv")
) + list(
    RAW_DIR.rglob("*.csv")
)


for path in sorted(set(csv_files)):

    try:

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as f:

            reader = csv.reader(f)

            rows = []

            for i, row in enumerate(reader):

                rows.append(row)

                if i >= 5:
                    break

        if not rows:
            continue

        header = rows[0]

        joined_header = " ".join(
            str(x) for x in header
        ).lower()

        scenario_related = any(
            keyword in joined_header
            for keyword in [
                "scenario",
                "snapshot",
                "s2",
                "demand",
                "wind",
                "generation",
                "dispatch",
                "load",
            ]
        )

        if scenario_related:

            print()
            print(
                f"CSV: {path.relative_to(PROJECT_ROOT)}"
            )

            print(
                "  Columns:"
            )

            print(
                "    " + " | ".join(header)
            )

            for row in rows[1:6]:

                print(
                    "    " + " | ".join(row)
                )

    except Exception as exc:

        print(
            f"WARNING: Could not inspect "
            f"{path.relative_to(PROJECT_ROOT)}: {exc}"
        )


# ---------------------------------------------------------------------
# JSON INSPECTION
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("JSON SCENARIO DATA INSPECTION")
print("=" * 70)


json_files = list(
    PROCESSED_DIR.rglob("*.json")
) + list(
    RAW_DIR.rglob("*.json")
)


for path in sorted(set(json_files)):

    try:

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            data = json.load(f)

    except Exception:

        continue

    serialized = json.dumps(
        data,
        ensure_ascii=False,
    )

    if any(
        scenario.lower() in serialized.lower()
        for scenario in SCENARIOS
    ):

        print()
        print(
            f"JSON containing scenario definitions: "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        if isinstance(data, dict):

            print(
                "  Top-level keys:"
            )

            for key in data.keys():

                print(
                    f"    {key}"
                )

        elif isinstance(data, list):

            print(
                f"  Top-level list length: {len(data)}"
            )


# ---------------------------------------------------------------------
# PYTHON FILE PRIORITIZATION
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("LIKELY FILES TO INSPECT FIRST")
print("=" * 70)


python_files = [
    p for p in files
    if p.suffix.lower() == ".py"
]


priority_scores = []


for path in python_files:

    try:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:

        continue

    lower = text.lower()

    score = 0

    scoring_terms = {

        "s2_peak_demand": 20,
        "scenario": 5,
        "p_set": 5,
        "p_max_pu": 5,
        "generators_t": 5,
        "loads_t": 5,
        "peak_demand": 10,
        "high_wind": 8,
        "maximum_stress": 8,
        "snapshot": 3,
        "dispatch": 5,
        "fillna": 2,
        "nan": 2,
    }

    for term, points in scoring_terms.items():

        score += lower.count(term) * points

    if score > 0:

        priority_scores.append(
            (
                score,
                path.relative_to(PROJECT_ROOT),
            )
        )


priority_scores.sort(
    key=lambda x: x[0],
    reverse=True,
)


for score, path in priority_scores[:20]:

    print(
        f"{score:5d}  {path}"
    )


# ---------------------------------------------------------------------
# FINAL INTERPRETATION
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()
print("The previous diagnostic established:")
print()
print("  S2_PEAK_DEMAND")
print("      generator p_set = NaN")
print("      load p_set      = NaN")
print("      generator p_nom = 0")
print()
print("Therefore the next repair target is the scenario")
print("construction/data-loading layer.")
print()
print("DO NOT:")
print("  - increase line capacities")
print("  - add another reinforcement")
print("  - interpret failed S2 loading values")
print("  - mark S2 overloaded lines as physical bottlenecks")
print()
print("The files printed above should reveal:")
print("  1. Where S1-S6 are defined")
print("  2. Where generator dispatch is assigned")
print("  3. Where load demand is assigned")
print("  4. Whether scenario data is stored in CSV/JSON")
print("  5. Whether NaN values are introduced during import")
print("  6. Which script creates the optimized network")
print()
print("NEXT ACTION:")
print()
print("Inspect the highest-scoring scenario-construction Python file.")
print("Do not modify it yet.")
print()
print("=" * 70)
print("SCENARIO CONSTRUCTION INSPECTION COMPLETE")
print("=" * 70)