import pypsa
import pandas as pd
import numpy as np

NETWORK_FILE = "data/processed/ireland_base_scenario.nc"

print("=" * 70)
print("       IRELAND GRID - BASE CASE ANALYSIS")
print("=" * 70)

print("\nLoading:")
print(NETWORK_FILE)

n = pypsa.Network(NETWORK_FILE)

print("OK: Network loaded.")

# ============================================================
# NETWORK
# ============================================================

print("\n" + "-" * 70)
print("NETWORK")
print("-" * 70)

print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")

# ============================================================
# SNAPSHOT
# ============================================================

snapshot = n.snapshots[0]

print("\n" + "-" * 70)
print("SNAPSHOT")
print("-" * 70)

print(f"Using snapshot: {snapshot}")

# ============================================================
# COMPONENT SETPOINTS
# ============================================================

print("\n" + "-" * 70)
print("COMPONENT SETPOINTS")
print("-" * 70)

# Load values
if len(n.loads) > 0:

    load_values = n.loads["p_set"].astype(float)

    total_demand = float(load_values.sum())

else:

    load_values = pd.Series(dtype=float)
    total_demand = 0.0


# Generator values
if len(n.generators) > 0:

    generator_values = n.generators["p_set"].astype(float)

    total_generation = float(generator_values.sum())

else:

    generator_values = pd.Series(dtype=float)
    total_generation = 0.0


print(f"Total demand          : {total_demand:.2f} MW")
print(f"Total generation      : {total_generation:.2f} MW")
print(
    f"Initial balance       : "
    f"{total_generation - total_demand:.2f} MW"
)

# ============================================================
# SHOW LOADS
# ============================================================

print("\nLoads:")

if len(n.loads) > 0:

    for name, value in load_values.items():

        print(
            f"  {name}: {value:.2f} MW"
        )

else:

    print("  NONE")

# ============================================================
# SHOW GENERATORS
# ============================================================

print("\nGenerators:")

if len(n.generators) > 0:

    for name, value in generator_values.items():

        print(
            f"  {name}: {value:.2f} MW"
        )

else:

    print("  NONE")

# ============================================================
# POWER FLOW
# ============================================================

print("\n" + "-" * 70)
print("LINEAR POWER FLOW")
print("-" * 70)

try:

    # PyPSA uses the currently selected snapshot.
    n.lpf()

    print("OK: Linear power flow completed.")

except Exception as e:

    print("FAILED: Linear power flow")
    print(e)

    raise SystemExit(1)

# ============================================================
# LINE LOADING
# ============================================================

print("\n" + "-" * 70)
print("TRANSMISSION LINE LOADING")
print("-" * 70)

if len(n.lines) > 0:

    line_flow = n.lines_t.p0.loc[snapshot]

    line_limit = n.lines["s_nom"]

    loading = (
        line_flow.abs()
        / line_limit
        * 100
    )

    line_results = pd.DataFrame({

        "flow_mw": line_flow,

        "thermal_limit_mw": line_limit,

        "loading_percent": loading

    })

    line_results = line_results.sort_values(
        "loading_percent",
        ascending=False
    )

    print(
        line_results.head(20).to_string()
    )

    max_line_loading = float(
        line_results["loading_percent"].max()
    )

    overloaded_lines = line_results[
        line_results["loading_percent"] > 100
    ]

else:

    line_results = pd.DataFrame()

    max_line_loading = 0.0

    overloaded_lines = pd.DataFrame()


print(
    f"\nMaximum line loading: "
    f"{max_line_loading:.2f}%"
)

print(
    f"Lines above 100%: "
    f"{len(overloaded_lines)}"
)

# ============================================================
# TRANSFORMER LOADING
# ============================================================

print("\n" + "-" * 70)
print("TRANSFORMER LOADING")
print("-" * 70)

if len(n.transformers) > 0:

    trafo_flow = n.transformers_t.p0.loc[snapshot]

    trafo_limit = n.transformers["s_nom"]

    trafo_loading = (
        trafo_flow.abs()
        / trafo_limit
        * 100
    )

    trafo_results = pd.DataFrame({

        "flow_mw": trafo_flow,

        "thermal_limit_mw": trafo_limit,

        "loading_percent": trafo_loading

    })

    trafo_results = trafo_results.sort_values(
        "loading_percent",
        ascending=False
    )

    print(
        trafo_results.to_string()
    )

    max_trafo_loading = float(
        trafo_results["loading_percent"].max()
    )

    overloaded_trafos = trafo_results[
        trafo_results["loading_percent"] > 100
    ]

else:

    trafo_results = pd.DataFrame()

    max_trafo_loading = 0.0

    overloaded_trafos = pd.DataFrame()


print(
    f"\nMaximum transformer loading: "
    f"{max_trafo_loading:.2f}%"
)

print(
    f"Transformers above 100%: "
    f"{len(overloaded_trafos)}"
)

# ============================================================
# BUS ANGLES
# ============================================================

print("\n" + "-" * 70)
print("BUS VOLTAGE ANGLES")
print("-" * 70)

try:

    angles = n.buses_t.v_ang.loc[snapshot]

    print(
        f"Minimum angle : "
        f"{angles.min():.6f}"
    )

    print(
        f"Maximum angle : "
        f"{angles.max():.6f}"
    )

except Exception as e:

    print(
        f"Could not read bus angles: {e}"
    )

# ============================================================
# CONGESTION DETAILS
# ============================================================

print("\n" + "-" * 70)
print("CONGESTION DETAILS")
print("-" * 70)

if len(overloaded_lines) > 0:

    print("\nOverloaded transmission lines:")

    for name, row in overloaded_lines.iterrows():

        print(
            f"\n  {name}"
        )

        print(
            f"    Flow       : "
            f"{row['flow_mw']:.2f} MW"
        )

        print(
            f"    Limit      : "
            f"{row['thermal_limit_mw']:.2f} MW"
        )

        print(
            f"    Loading    : "
            f"{row['loading_percent']:.2f}%"
        )

        if name in n.lines.index:

            bus0 = n.lines.loc[name, "bus0"]

            bus1 = n.lines.loc[name, "bus1"]

            print(
                f"    From bus   : {bus0}"
            )

            print(
                f"    To bus     : {bus1}"
            )

else:

    print("No overloaded transmission lines.")

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("                 BASE CASE RESULT")
print("=" * 70)

print(
    f"Total demand              : "
    f"{total_demand:.2f} MW"
)

print(
    f"Total generation          : "
    f"{total_generation:.2f} MW"
)

print(
    f"Maximum line loading      : "
    f"{max_line_loading:.2f}%"
)

print(
    f"Maximum transformer load  : "
    f"{max_trafo_loading:.2f}%"
)

if (
    len(overloaded_lines) == 0
    and len(overloaded_trafos) == 0
):

    print(
        "\nSTATUS: "
        "NO THERMAL VIOLATIONS"
    )

else:

    print(
        "\nSTATUS: "
        "THERMAL VIOLATIONS PRESENT"
    )

print("\n" + "=" * 70)

print(
    "This remains a synthetic engineering baseline."
)

print(
    "It is NOT official EirGrid operating data."
)

print("=" * 70)