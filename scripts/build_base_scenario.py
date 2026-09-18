import os
import sys
import pypsa
import pandas as pd


NETWORK_FILE = "data/processed/ireland_network.nc"
OUTPUT_FILE = "data/processed/ireland_base_scenario.nc"

# ------------------------------------------------------------
# BASE-CASE ASSUMPTIONS
# ------------------------------------------------------------
# These are NOT official Irish grid values.
# They are temporary engineering assumptions used to create
# a functioning base operating scenario.
#
# They will later be replaced by real EirGrid / ENTSO-E data.

TOTAL_DEMAND_MW = 5000.0
TOTAL_GENERATION_MW = 5000.0

SNAPSHOT = "base"


def main():

    print("=" * 70)
    print("       IRELAND GRID - BASE OPERATING SCENARIO")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    if not os.path.exists(NETWORK_FILE):
        print("\nERROR: Network file not found:")
        print(NETWORK_FILE)
        sys.exit(1)

    print("\nLoading network:")
    print(NETWORK_FILE)

    try:
        n = pypsa.Network(NETWORK_FILE)
    except Exception as e:
        print("\nERROR: Could not load network.")
        print(e)
        sys.exit(1)

    print("OK: Network loaded.")

    # --------------------------------------------------------
    # 2. Remove previous synthetic scenario components
    # --------------------------------------------------------

    old_loads = [
        name for name in n.loads.index
        if str(name).startswith("base_load_")
    ]

    old_generators = [
        name for name in n.generators.index
        if str(name).startswith("base_generation_")
    ]

    for name in old_loads:
        n.remove("Load", name)

    for name in old_generators:
        n.remove("Generator", name)

    # --------------------------------------------------------
    # 3. Select transmission buses
    # --------------------------------------------------------

    if "v_nom" in n.buses.columns:
        transmission_buses = n.buses[
            n.buses["v_nom"] >= 220
        ].index.tolist()
    else:
        transmission_buses = n.buses.index.tolist()

    if len(transmission_buses) == 0:
        print("\nERROR: No transmission buses found.")
        sys.exit(1)

    print("\nTransmission buses:", len(transmission_buses))

    # --------------------------------------------------------
    # 4. Calculate bus connectivity
    # --------------------------------------------------------

    degree = {bus: 0 for bus in transmission_buses}

    for _, line in n.lines.iterrows():

        bus0 = line["bus0"]
        bus1 = line["bus1"]

        if bus0 in degree:
            degree[bus0] += 1

        if bus1 in degree:
            degree[bus1] += 1

    for _, trafo in n.transformers.iterrows():

        bus0 = trafo["bus0"]
        bus1 = trafo["bus1"]

        if bus0 in degree:
            degree[bus0] += 1

        if bus1 in degree:
            degree[bus1] += 1

    degree_series = pd.Series(degree).sort_values(ascending=False)

    print("\nMost connected buses:")
    print(degree_series.head(10).to_string())

    # --------------------------------------------------------
    # 5. Select load buses
    # --------------------------------------------------------
    #
    # We distribute demand across the most connected
    # transmission buses.
    #
    # This is only a temporary synthetic load distribution.

    number_of_load_buses = min(10, len(transmission_buses))

    load_buses = degree_series.head(
        number_of_load_buses
    ).index.tolist()

    # Demand weights
    weights = list(range(number_of_load_buses, 0, -1))

    total_weight = sum(weights)

    print("\nSelected load buses:")

    for bus, weight in zip(load_buses, weights):

        demand = TOTAL_DEMAND_MW * weight / total_weight

        print(
            f"  {bus}: {demand:.2f} MW"
        )

        n.add(
            "Load",
            f"base_load_{bus}",
            bus=bus,
            p_set=demand
        )

    # --------------------------------------------------------
    # 6. Select generation buses
    # --------------------------------------------------------
    #
    # We use several highly connected buses as conventional
    # generation locations.
    #
    # One generator is configured as the slack generator.

    generation_buses = degree_series.head(
        min(5, len(transmission_buses))
    ).index.tolist()

    # Generation distribution
    generation_weights = [5, 4, 3, 2, 1][:len(generation_buses)]

    total_generation_weight = sum(generation_weights)

    print("\nSelected generation buses:")

    for i, (bus, weight) in enumerate(
        zip(generation_buses, generation_weights)
    ):

        generation = (
            TOTAL_GENERATION_MW
            * weight
            / total_generation_weight
        )

        # First generator acts as slack/reference.
        control = "Slack" if i == 0 else "PQ"

        n.add(
            "Generator",
            f"base_generation_{bus}",
            bus=bus,
            p_set=generation,
            control=control,
            p_nom=generation,
            carrier="base_generation"
        )

        print(
            f"  {bus}: {generation:.2f} MW "
            f"({control})"
        )

    # --------------------------------------------------------
    # 7. Set snapshot
    # --------------------------------------------------------

    n.set_snapshots([SNAPSHOT])

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    output_directory = os.path.dirname(OUTPUT_FILE)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    n.export_to_netcdf(OUTPUT_FILE)

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("             BASE SCENARIO CREATED")
    print("=" * 70)

    print(f"\nNetwork:")
    print(f"  Buses        : {len(n.buses)}")
    print(f"  Lines        : {len(n.lines)}")
    print(f"  Transformers : {len(n.transformers)}")

    print("\nOperating scenario:")
    print(f"  Loads        : {len(n.loads)}")
    print(f"  Generators   : {len(n.generators)}")

    print(
        f"\nTotal demand assumed      : "
        f"{TOTAL_DEMAND_MW:.1f} MW"
    )

    print(
        f"Total generation assumed : "
        f"{TOTAL_GENERATION_MW:.1f} MW"
    )

    print(f"\nSaved:")
    print(f"  {OUTPUT_FILE}")

    print("\nIMPORTANT:")
    print("This is a synthetic engineering baseline.")
    print("The demand and generation values are NOT official")
    print("Irish system measurements.")
    print("They will be replaced with real EirGrid/ENTSO-E data")
    print("in the next data phase.")

    print("\nNext step:")
    print("Run the base-case power flow and inspect line loading.")

    print("=" * 70)


if __name__ == "__main__":
    main()