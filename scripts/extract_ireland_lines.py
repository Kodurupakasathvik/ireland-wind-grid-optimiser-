import pandas as pd

# Get Irish bus IDs
buses = pd.read_csv(
    "data/processed/ireland_buses.csv",
    usecols=["bus_id"]
)

irish_bus_ids = set(buses["bus_id"])

output_file = "data/processed/ireland_lines.csv"

header = [
    "line_id",
    "bus0",
    "bus1",
    "voltage",
    "i_nom",
    "circuits",
    "s_nom",
    "r",
    "x",
    "b",
    "length",
    "underground",
    "under_construction",
    "type",
    "tags"
]

total_lines = 0
first_write = True

print("Processing lines correctly...")

with open(
    "data/raw/lines.csv",
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    next(f)  # skip header

    for line_number, line in enumerate(f, start=2):

        # Split only at the first 15 commas.
        # Everything after that is geometry.
        fields = line.rstrip("\n").split(",", 15)

        if len(fields) < 15:
            continue

        bus0 = fields[1]
        bus1 = fields[2]

        if bus0 in irish_bus_ids and bus1 in irish_bus_ids:

            row = fields[:15]

            df = pd.DataFrame([row], columns=header)

            df.to_csv(
                output_file,
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False
            total_lines += 1

            if total_lines % 100 == 0:
                print("Irish lines found:", total_lines)

print("Done!")
print("Total Irish lines:", total_lines)