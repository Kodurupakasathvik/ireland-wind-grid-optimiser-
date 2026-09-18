from pathlib import Path
import pypsa
import numpy as np
import copy

# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_PATH = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

SLACK_GENERATOR = "eirgrid_non_wind_generation"

print("=" * 70)
print("S2 AC POWER FLOW - TRANSFORMER ISOLATION TEST")
print("=" * 70)

# ============================================================
# 1. CHECK NETWORK FILE
# ============================================================

print("\nNETWORK FILE")
print("------------")
print(NETWORK_PATH)

if not NETWORK_PATH.exists():
    raise FileNotFoundError(
        f"\nNetwork file not found:\n{NETWORK_PATH}"
    )

# ============================================================
# 2. LOAD NETWORK
# ============================================================

n = pypsa.Network(NETWORK_PATH)

print("\nNETWORK")
print("-------")
print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Links        : {len(n.links)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")

# ============================================================
# 3. SNAPSHOT CHECK
# ============================================================

if SNAPSHOT not in n.snapshots:
    raise ValueError(
        f"\nSnapshot '{SNAPSHOT}' not found.\n"
        f"Available snapshots:\n{list(n.snapshots)}"
    )

# ============================================================
# 4. IN-MEMORY COPY
# ============================================================

print("\nCREATING IN-MEMORY DIAGNOSTIC COPY")
print("----------------------------------")
print("Original network will NOT be modified.")
print("No file will be saved.")

n = copy.deepcopy(n)

# ============================================================
# 5. OPERATING POINT
# ============================================================

print("\nOPERATING POINT")
print("----------------")

# Use p_set because this is the scheduled operating point.
if hasattr(n.generators_t, "p_set"):

    wind_generators = n.generators[
        n.generators["carrier"]
        .astype(str)
        .str.lower()
        .str.contains("wind", na=False)
    ].index.tolist()

    if len(wind_generators) > 0:

        available_wind = [
            g for g in wind_generators
            if g in n.generators_t.p_set.columns
        ]

        wind_generation = float(
            n.generators_t.p_set.loc[
                SNAPSHOT,
                available_wind
            ].sum()
        )

    else:

        wind_generation = 0.0

else:

    wind_generation = 0.0


# Total load
load_columns = [
    x for x in n.loads.index
    if x in n.loads_t.p_set.columns
]

total_load = float(
    n.loads_t.p_set.loc[
        SNAPSHOT,
        load_columns
    ].sum()
)

required_slack = total_load - wind_generation

print(
    f"Wind generation : "
    f"{wind_generation:.6f} MW"
)

print(
    f"Total load      : "
    f"{total_load:.6f} MW"
)

print(
    f"Required slack  : "
    f"{required_slack:.6f} MW"
)

# ============================================================
# 6. SET Q = 0
# ============================================================

print("\nSETTING Q = 0")
print("------------")

if hasattr(n.generators_t, "q_set"):

    if SNAPSHOT in n.generators_t.q_set.index:

        n.generators_t.q_set.loc[
            SNAPSHOT,
            :
        ] = 0.0

if hasattr(n.loads_t, "q_set"):

    if SNAPSHOT in n.loads_t.q_set.index:

        n.loads_t.q_set.loc[
            SNAPSHOT,
            :
        ] = 0.0

print("Generator q_set = 0")
print("Load q_set = 0")

# ============================================================
# 7. DISCONNECT INTERCONNECTORS
# ============================================================

print("\nDISCONNECTING INTERCONNECTORS")
print("-----------------------------")

# Generators
for generator in [
    "ewic_import",
    "greenlink_import"
]:

    if generator in n.generators.index:

        if generator in n.generators_t.p_set.columns:

            n.generators_t.p_set.loc[
                SNAPSHOT,
                generator
            ] = 0.0

        print(
            f"{generator}: 0 MW"
        )

# Links
for link in [
    "EWIC_interface",
    "Greenlink_interface"
]:

    if link in n.links.index:

        if link in n.links_t.p_set.columns:

            n.links_t.p_set.loc[
                SNAPSHOT,
                link
            ] = 0.0

        print(
            f"{link}: 0 MW"
        )

# ============================================================
# 8. ORIGINAL TRANSFORMERS
# ============================================================

print("\nORIGINAL TRANSFORMERS")
print("---------------------")

transformer_names = list(
    n.transformers.index
)

if len(transformer_names) == 0:

    print("No transformers found.")

else:

    for name in transformer_names:

        tr = n.transformers.loc[name]

        print(f"\n{name}")

        print(
            f"  bus0        = {tr.bus0}"
        )

        print(
            f"  bus1        = {tr.bus1}"
        )

        print(
            f"  r           = {tr.r}"
        )

        print(
            f"  x           = {tr.x}"
        )

        print(
            f"  s_nom       = {tr.s_nom}"
        )

        print(
            f"  tap_ratio   = {tr.tap_ratio}"
        )

        print(
            f"  phase_shift = {tr.phase_shift}"
        )

# ============================================================
# 9. REMOVE TRANSFORMERS
# ============================================================

print("\nREMOVING TRANSFORMERS")
print("---------------------")

print(
    f"Original transformers : "
    f"{len(n.transformers)}"
)

# PyPSA uses remove(), not mremove().
for transformer in transformer_names:

    n.remove(
        "Transformer",
        transformer
    )

print(
    f"Remaining transformers: "
    f"{len(n.transformers)}"
)

# ============================================================
# 10. VERIFY REMOVAL
# ============================================================

print("\nTRANSFORMER REMOVAL CHECK")
print("-------------------------")

if len(n.transformers) == 0:

    print(
        "PASS: All transformers removed."
    )

else:

    print(
        "FAIL: Transformers remain."
    )

    print(
        n.transformers.index.tolist()
    )

# ============================================================
# 11. SET SLACK GENERATOR
# ============================================================

print("\nSETTING SLACK GENERATOR")
print("-----------------------")

if SLACK_GENERATOR not in n.generators.index:

    raise ValueError(
        f"Slack generator '{SLACK_GENERATOR}' "
        f"was not found."
    )

slack_bus = n.generators.at[
    SLACK_GENERATOR,
    "bus"
]

print(
    f"Slack generator : "
    f"{SLACK_GENERATOR}"
)

print(
    f"Slack bus       : "
    f"{slack_bus}"
)

# Give enough capacity for diagnostic purposes.
n.generators.at[
    SLACK_GENERATOR,
    "p_nom"
] = 10000.0

# Set scheduled active power.
if SLACK_GENERATOR in n.generators_t.p_set.columns:

    n.generators_t.p_set.loc[
        SNAPSHOT,
        SLACK_GENERATOR
    ] = required_slack

# Set slack control.
n.generators.at[
    SLACK_GENERATOR,
    "control"
] = "Slack"

print(
    f"Slack setpoint  : "
    f"{required_slack:.6f} MW"
)

print(
    f"Slack p_nom     : "
    f"{n.generators.at[SLACK_GENERATOR, 'p_nom']:.6f} MW"
)

# ============================================================
# 12. DETERMINE TOPOLOGY
# ============================================================

print("\nDETERMINING TOPOLOGY")
print("--------------------")

n.determine_network_topology()

print("\nSUBNETWORKS")
print("-----------")

print(
    n.sub_networks[
        ["carrier", "slack_bus"]
    ]
)

print("\nSUBNETWORK COUNT")
print("----------------")

print(
    f"AC subnetworks : "
    f"{len(n.sub_networks)}"
)

# ============================================================
# 13. SUBNETWORK DETAILS
# ============================================================

print("\nSUBNETWORK DETAILS")
print("------------------")

for sub_name, sub in n.sub_networks.iterrows():

    sub_obj = sub["obj"]

    buses = list(
        sub_obj.buses_i()
    )

    generators = n.generators[
        n.generators.bus.isin(buses)
    ]

    loads = n.loads[
        n.loads.bus.isin(buses)
    ]

    lines = n.lines[
        n.lines.bus0.isin(buses)
        &
        n.lines.bus1.isin(buses)
    ]

    transformers = n.transformers[
        n.transformers.bus0.isin(buses)
        |
        n.transformers.bus1.isin(buses)
    ]

    print("\n----------------------------------------")

    print(
        f"Subnetwork : {sub_name}"
    )

    print(
        f"Carrier    : {sub.carrier}"
    )

    print(
        f"Slack bus  : {sub.slack_bus}"
    )

    print(
        f"Buses      : {len(buses)}"
    )

    print(
        f"Lines      : {len(lines)}"
    )

    print(
        f"Transformers: {len(transformers)}"
    )

    print(
        f"Generators : {len(generators)}"
    )

    print(
        f"Loads      : {len(loads)}"
    )

    # --------------------------------------------------------
    # GENERATORS
    # --------------------------------------------------------

    print("\nGenerators:")

    if len(generators) == 0:

        print("  NONE")

    else:

        for generator in generators.index:

            print(
                f"  {generator}"
            )

    # --------------------------------------------------------
    # LOADS
    # --------------------------------------------------------

    print("\nLoads:")

    if len(loads) == 0:

        print("  NONE")

    else:

        for load in loads.index:

            print(
                f"  {load}"
            )

# ============================================================
# 14. IDENTIFY MAIN GRID
# ============================================================

print("\nMAIN GRID IDENTIFICATION")
print("------------------------")

main_subnetwork = None

for sub_name, sub in n.sub_networks.iterrows():

    sub_obj = sub["obj"]

    buses = list(
        sub_obj.buses_i()
    )

    if slack_bus in buses:

        main_subnetwork = sub_name

        break

if main_subnetwork is None:

    print(
        "ERROR: Slack bus not found "
        "in any AC subnetwork."
    )

else:

    print(
        f"Main grid subnetwork : "
        f"{main_subnetwork}"
    )

    main_obj = n.sub_networks.at[
        main_subnetwork,
        "obj"
    ]

    main_buses = list(
        main_obj.buses_i()
    )

    print(
        f"Main grid buses      : "
        f"{len(main_buses)}"
    )

# ============================================================
# 15. ISOLATED SUBNETWORK ANALYSIS
# ============================================================

print("\nISOLATED SUBNETWORK CHECK")
print("-------------------------")

isolated_count = 0

for sub_name, sub in n.sub_networks.iterrows():

    if sub_name == main_subnetwork:

        continue

    isolated_count += 1

    sub_obj = sub["obj"]

    buses = list(
        sub_obj.buses_i()
    )

    generators = n.generators[
        n.generators.bus.isin(buses)
    ]

    loads = n.loads[
        n.loads.bus.isin(buses)
    ]

    print("\n----------------------------------------")

    print(
        f"Subnetwork : {sub_name}"
    )

    print(
        f"Slack bus  : {sub.slack_bus}"
    )

    print(
        f"Buses      : {len(buses)}"
    )

    print(
        f"Generators : {len(generators)}"
    )

    print(
        f"Loads      : {len(loads)}"
    )

    if (
        len(generators) == 0
        and
        len(loads) == 0
    ):

        print(
            "STATUS: Electrically isolated "
            "network with no generation/load."
        )

    elif (
        len(generators) == 0
        and
        len(loads) > 0
    ):

        print(
            "STATUS: ISOLATED LOAD SUBNETWORK "
            "WITHOUT GENERATION."
        )

    elif (
        len(generators) > 0
        and
        len(loads) == 0
    ):

        print(
            "STATUS: ISOLATED GENERATION "
            "SUBNETWORK WITHOUT LOAD."
        )

    else:

        print(
            "STATUS: Independent subnetwork "
            "with generation and load."
        )

if isolated_count == 0:

    print(
        "No isolated subnetworks detected."
    )

# ============================================================
# 16. ACTIVE POWER BALANCE
# ============================================================

print("\nACTIVE POWER BALANCE")
print("--------------------")

generation_columns = [
    x for x in n.generators.index
    if x in n.generators_t.p_set.columns
]

load_columns = [
    x for x in n.loads.index
    if x in n.loads_t.p_set.columns
]

generation = float(
    n.generators_t.p_set.loc[
        SNAPSHOT,
        generation_columns
    ].sum()
)

load = float(
    n.loads_t.p_set.loc[
        SNAPSHOT,
        load_columns
    ].sum()
)

print(
    f"Generation : "
    f"{generation:.6f} MW"
)

print(
    f"Load       : "
    f"{load:.6f} MW"
)

print(
    f"Difference : "
    f"{generation - load:.6f} MW"
)

# ============================================================
# 17. TRANSFORMER SIDE BUS CHECK
# ============================================================

print("\nTRANSFORMER-SIDE BUS CHECK")
print("--------------------------")

# We saved the original transformer names before removing them.
for transformer in transformer_names:

    # The transformer object is gone, but its two buses remain.
    # Determine whether each bus belongs to the main grid.

    # Find original bus names from the original network.
    # We can reconstruct them from the transformer naming
    # using the original network copy loaded above.
    pass

# Reload original network ONLY for reference.
original = pypsa.Network(NETWORK_PATH)

for transformer in transformer_names:

    if transformer not in original.transformers.index:

        continue

    tr = original.transformers.loc[
        transformer
    ]

    bus0 = tr.bus0
    bus1 = tr.bus1

    bus0_sub = None
    bus1_sub = None

    for sub_name, sub in n.sub_networks.iterrows():

        buses = list(
            sub["obj"].buses_i()
        )

        if bus0 in buses:
            bus0_sub = sub_name

        if bus1 in buses:
            bus1_sub = sub_name

    print("\n" + transformer)

    print(
        f"  {bus0}"
    )

    print(
        f"    subnetwork: {bus0_sub}"
    )

    print(
        f"  {bus1}"
    )

    print(
        f"    subnetwork: {bus1_sub}"
    )

    if (
        bus0_sub is not None
        and
        bus1_sub is not None
        and
        bus0_sub != bus1_sub
    ):

        print(
            "  RESULT: Transformer was connecting "
            "two different AC subnetworks."
        )

    else:

        print(
            "  RESULT: Both buses remain in the "
            "same AC subnetwork."
        )

# ============================================================
# 18. DO NOT RUN AC PF
# ============================================================

print("\nAC POWER FLOW DECISION")
print("----------------------")

print(
    "AC PF is intentionally NOT executed."
)

print(
    "Reason: removing all transformers creates "
    "electrically disconnected subnetworks."
)

print(
    "The purpose of this test is to determine "
    "whether the transformers are responsible "
    "for connecting the voltage levels."
)

# ============================================================
# 19. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER ISOLATION TEST COMPLETE")
print("=" * 70)

print("\nFINAL RESULT")
print("------------")

print(
    f"Original transformers : "
    f"{len(transformer_names)}"
)

print(
    f"Remaining transformers: "
    f"{len(n.transformers)}"
)

print(
    f"AC subnetworks        : "
    f"{len(n.sub_networks)}"
)

print(
    f"Main grid subnetwork  : "
    f"{main_subnetwork}"
)

if len(n.transformers) == 0:

    print(
        "\nPASS: All 5 transformers were removed "
        "from the in-memory diagnostic network."
    )

else:

    print(
        "\nFAIL: Transformer removal was incomplete."
    )

print(
    "\nOriginal network file was NOT modified."
)

print("=" * 70)