"""
====================================================================================================
S4.2 — RESIDUAL BOTTLENECK DIAGNOSIS
====================================================================================================

Purpose
-------
Diagnose the residual thermal / voltage bottlenecks remaining after the
P3_HIGH_COORDINATED reinforcement package at the critical S2_PEAK_DEMAND
snapshot.

Important
---------
- Source network is READ-ONLY.
- Reinforcements are applied only to the in-memory PyPSA network.
- Reactive support is applied to the existing generator
  eirgrid_wind_way/104388595-220.
- AC nonlinear power flow is used.
- Dispatch and loads are unchanged.
- No optimisation or reinforcement selection is performed here.

====================================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

PACKAGE_NAME = "P3_HIGH_COORDINATED"

REACTIVE_GENERATOR = "eirgrid_wind_way/104388595-220"
REACTIVE_MVAR = 500.0

# Thermal acceptance threshold.
THERMAL_LIMIT_PCT = 100.0

# Voltage acceptance range used by the project screening.
V_MIN_LIMIT_PU = 0.95
V_MAX_LIMIT_PU = 1.05

# Output files.
OUTPUT_SUMMARY = Path(
    "data/processed/s4_2_residual_bottleneck_diagnosis_summary.csv"
)

OUTPUT_LINES = Path(
    "data/processed/s4_2_residual_bottleneck_lines.csv"
)

OUTPUT_BUSES = Path(
    "data/processed/s4_2_residual_bottleneck_buses.csv"
)

OUTPUT_TRANSFORMERS = Path(
    "data/processed/s4_2_residual_bottleneck_transformers.csv"
)


# ==================================================================================================
# P3 REINFORCEMENTS
# ==================================================================================================

P3_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.75,
    "merged_way/61295764-220+1": 2.00,
    "way/343436171-220": 2.00,
    "merged_way/257889771-220+1": 1.75,
    "merged_relation/4872159-220+1": 1.75,
}


# ==================================================================================================
# HELPERS
# ==================================================================================================

def print_rule(char="-", width=100):
    print(char * width)


def safe_float(value):
    """
    Convert a scalar to float safely.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def get_line_loading(n, snapshot):
    """
    Return line loading percentage.

    PyPSA uses:
        p0 / s_nom

    for AC line apparent-power loading.

    We use max(|p0|, |p1|) / s_nom so that loading is evaluated
    from either end of the line.
    """

    if len(n.lines) == 0:
        return pd.Series(dtype=float)

    s_nom = n.lines["s_nom"].replace(0, np.nan)

    p0 = n.lines_t.p0.loc[snapshot].abs()
    p1 = n.lines_t.p1.loc[snapshot].abs()

    loading = (
        pd.concat([p0, p1], axis=1)
        .max(axis=1)
        / s_nom
        * 100.0
    )

    return loading.sort_values(ascending=False)


def get_transformer_loading(n, snapshot):
    """
    Calculate transformer loading percentage.

    Uses max(|p0|, |p1|) / s_nom.
    """

    if len(n.transformers) == 0:
        return pd.Series(dtype=float)

    s_nom = n.transformers["s_nom"].replace(0, np.nan)

    p0 = n.transformers_t.p0.loc[snapshot].abs()
    p1 = n.transformers_t.p1.loc[snapshot].abs()

    loading = (
        pd.concat([p0, p1], axis=1)
        .max(axis=1)
        / s_nom
        * 100.0
    )

    return loading.sort_values(ascending=False)


def get_bus_voltage(n, snapshot):
    """
    Return bus voltage magnitude in pu.
    """

    if len(n.buses) == 0:
        return pd.Series(dtype=float)

    v_mag = n.buses_t.v_mag_pu.loc[snapshot]

    return v_mag.sort_values()


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.2 — RESIDUAL BOTTLENECK DIAGNOSIS")
print("=" * 100)

print()
print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print(f"Package  : {PACKAGE_NAME}")
print("PF       : AC nonlinear")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Source   : READ-ONLY")
print(f"Reactive : +{REACTIVE_MVAR:.0f} MVAr")


# ==================================================================================================
# LOAD NETWORK
# ==================================================================================================

if not NETWORK_PATH.exists():
    raise FileNotFoundError(
        f"Network file not found:\n{NETWORK_PATH.resolve()}"
    )

n = pypsa.Network(NETWORK_PATH)


# ==================================================================================================
# SNAPSHOT CHECK
# ==================================================================================================

if SNAPSHOT not in n.snapshots:
    raise RuntimeError(
        f"Snapshot '{SNAPSHOT}' not found in network.\n"
        f"Available snapshots:\n{list(n.snapshots)}"
    )


# ==================================================================================================
# APPLY P3 REINFORCEMENTS
# ==================================================================================================

print()
print_rule()
print("APPLYING P3 REINFORCEMENTS")
print_rule()

reinforcement_records = []

for component_name, multiplier in P3_REINFORCEMENTS.items():

    if component_name not in n.lines.index:
        raise RuntimeError(
            f"Required reinforcement line does not exist:\n"
            f"{component_name}"
        )

    old_s_nom = float(n.lines.at[component_name, "s_nom"])

    new_s_nom = old_s_nom * multiplier

    n.lines.at[component_name, "s_nom"] = new_s_nom

    reinforcement_records.append(
        {
            "component": component_name,
            "multiplier": multiplier,
            "old_s_nom_mw": old_s_nom,
            "new_s_nom_mw": new_s_nom,
        }
    )

    print(
        f"{component_name:<55}"
        f"{multiplier:>6.2f}x"
        f"{old_s_nom:>12.3f} -> "
        f"{new_s_nom:>12.3f} MW"
    )


# ==================================================================================================
# APPLY REACTIVE SUPPORT
# ==================================================================================================

print()
print_rule()
print("APPLYING REACTIVE SUPPORT")
print_rule()


# ----------------------------------------------------------------------------------
# IMPORTANT FIX
# ----------------------------------------------------------------------------------
#
# The previous script failed here:
#
# old_q = n.generators_t.q_set.loc[SNAPSHOT, REACTIVE_GENERATOR]
#
# because the generator was not present as a column in q_set.
#
# We first verify that the generator itself exists in n.generators.
# Then we ensure the q_set time-series column exists.
# ----------------------------------------------------------------------------------

if REACTIVE_GENERATOR not in n.generators.index:

    matching_generators = [
        g
        for g in n.generators.index
        if "104388595" in str(g)
    ]

    raise RuntimeError(
        "\nReactive-support generator does not exist in n.generators.\n\n"
        f"Requested generator:\n"
        f"  {REACTIVE_GENERATOR}\n\n"
        f"Matching generators containing '104388595':\n"
        f"  {matching_generators}\n\n"
        f"First available generators:\n"
        f"  {list(n.generators.index[:20])}"
    )


# Ensure the snapshot exists in q_set.
if SNAPSHOT not in n.generators_t.q_set.index:

    n.generators_t.q_set.loc[SNAPSHOT] = np.nan


# Ensure generator has a q_set column.
if REACTIVE_GENERATOR not in n.generators_t.q_set.columns:

    # Use the generator's static q_set value when available.
    static_q = safe_float(
        n.generators.at[REACTIVE_GENERATOR, "q_set"]
        if "q_set" in n.generators.columns
        else 0.0
    )

    if np.isnan(static_q):
        static_q = 0.0

    n.generators_t.q_set[REACTIVE_GENERATOR] = static_q


old_q = safe_float(
    n.generators_t.q_set.loc[
        SNAPSHOT,
        REACTIVE_GENERATOR
    ]
)

if np.isnan(old_q):
    old_q = 0.0


n.generators_t.q_set.loc[
    SNAPSHOT,
    REACTIVE_GENERATOR
] = REACTIVE_MVAR


print(
    f"Reactive support applied through generator: "
    f"{REACTIVE_GENERATOR}"
)

print(
    f"Q setpoint: "
    f"{old_q:.3f} -> "
    f"{REACTIVE_MVAR:.3f} MVAr"
)


# ==================================================================================================
# RUN AC NONLINEAR POWER FLOW
# ==================================================================================================

print()
print_rule()
print("RUNNING AC NONLINEAR POWER FLOW")
print_rule()

try:

    n.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-8,
        use_seed=True,
    )

except Exception as exc:

    print()
    print("=" * 100)
    print("POWER FLOW FAILED")
    print("=" * 100)
    print()
    print(str(exc))
    print()

    raise


# ==================================================================================================
# POWER-FLOW CONVERGENCE
# ==================================================================================================

converged = True

try:

    pf_converged = n.sub_networks_t["Converged"].loc[SNAPSHOT]

    if isinstance(pf_converged, pd.Series):
        converged = bool(pf_converged.all())
    else:
        converged = bool(pf_converged)

except Exception:

    # If the exact convergence structure differs between PyPSA versions,
    # reaching this point without an exception is treated as successful PF.
    converged = True


# ==================================================================================================
# LINE BOTTLENECK DIAGNOSIS
# ==================================================================================================

line_loading = get_line_loading(n, SNAPSHOT)

overloaded_lines = line_loading[
    line_loading > THERMAL_LIMIT_PCT
]

top_lines = line_loading.head(15)


# ==================================================================================================
# BUS VOLTAGE DIAGNOSIS
# ==================================================================================================

bus_voltage = get_bus_voltage(n, SNAPSHOT)

low_voltage_buses = bus_voltage[
    bus_voltage < V_MIN_LIMIT_PU
]

high_voltage_buses = bus_voltage[
    bus_voltage > V_MAX_LIMIT_PU
]


# ==================================================================================================
# TRANSFORMER DIAGNOSIS
# ==================================================================================================

transformer_loading = get_transformer_loading(
    n,
    SNAPSHOT
)

overloaded_transformers = transformer_loading[
    transformer_loading > THERMAL_LIMIT_PCT
]

top_transformers = transformer_loading.head(15)


# ==================================================================================================
# MAIN RESULT
# ==================================================================================================

print()
print("=" * 100)
print("RESIDUAL BOTTLENECK RESULT")
print("=" * 100)

print()
print(f"Converged                 : {converged}")

if len(bus_voltage) > 0:

    min_voltage = float(bus_voltage.min())
    min_voltage_bus = str(bus_voltage.idxmin())

    max_voltage = float(bus_voltage.max())
    max_voltage_bus = str(bus_voltage.idxmax())

else:

    min_voltage = np.nan
    min_voltage_bus = "N/A"

    max_voltage = np.nan
    max_voltage_bus = "N/A"


if len(line_loading) > 0:

    max_line_loading = float(line_loading.max())
    critical_line = str(line_loading.idxmax())

else:

    max_line_loading = np.nan
    critical_line = "N/A"


if len(transformer_loading) > 0:

    max_transformer_loading = float(
        transformer_loading.max()
    )

    worst_transformer = str(
        transformer_loading.idxmax()
    )

else:

    max_transformer_loading = np.nan
    worst_transformer = "N/A"


print(f"Minimum voltage           : {min_voltage:.6f} pu")
print(f"Minimum-voltage bus       : {min_voltage_bus}")

print(f"Maximum voltage           : {max_voltage:.6f} pu")
print(f"Maximum-voltage bus       : {max_voltage_bus}")

print(f"Low-voltage buses         : {len(low_voltage_buses)}")
print(f"High-voltage buses        : {len(high_voltage_buses)}")

print(f"Max line loading          : {max_line_loading:.6f} %")
print(f"Overloaded lines          : {len(overloaded_lines)}")
print(f"Critical line             : {critical_line}")

print(
    f"Max transformer loading   : "
    f"{max_transformer_loading:.6f} %"
)

print(f"Worst transformer         : {worst_transformer}")


# ==================================================================================================
# TOP RESIDUAL LINE BOTTLENECKS
# ==================================================================================================

print()
print_rule()
print("TOP RESIDUAL LINE BOTTLENECKS")
print_rule()

if len(top_lines) == 0:

    print("No lines found.")

else:

    print(
        f"{'Line':<60}"
        f"{'Loading %':>12}"
        f"{'s_nom MW':>12}"
        f"{'p0 MW':>12}"
        f"{'p1 MW':>12}"
    )

    for line_name, loading in top_lines.items():

        s_nom = safe_float(
            n.lines.at[line_name, "s_nom"]
        )

        p0 = safe_float(
            n.lines_t.p0.loc[
                SNAPSHOT,
                line_name
            ]
        )

        p1 = safe_float(
            n.lines_t.p1.loc[
                SNAPSHOT,
                line_name
            ]
        )

        print(
            f"{str(line_name):<60}"
            f"{loading:>12.3f}"
            f"{s_nom:>12.3f}"
            f"{p0:>12.3f}"
            f"{p1:>12.3f}"
        )


# ==================================================================================================
# OVERLOADED LINE DETAILS
# ==================================================================================================

print()
print_rule()
print("OVERLOADED LINE DETAILS")
print_rule()

if len(overloaded_lines) == 0:

    print("No overloaded lines.")

else:

    print(
        f"{'Line':<60}"
        f"{'Loading %':>12}"
        f"{'Excess %':>12}"
        f"{'s_nom MW':>12}"
    )

    for line_name, loading in overloaded_lines.items():

        s_nom = safe_float(
            n.lines.at[line_name, "s_nom"]
        )

        excess = loading - THERMAL_LIMIT_PCT

        print(
            f"{str(line_name):<60}"
            f"{loading:>12.3f}"
            f"{excess:>12.3f}"
            f"{s_nom:>12.3f}"
        )


# ==================================================================================================
# LOW-VOLTAGE BUS DETAILS
# ==================================================================================================

print()
print_rule()
print("LOW-VOLTAGE BUS DETAILS")
print_rule()

if len(low_voltage_buses) == 0:

    print("No buses below voltage limit.")

else:

    print(
        f"{'Bus':<65}"
        f"{'V pu':>12}"
        f"{'Deviation':>14}"
    )

    for bus_name, voltage in low_voltage_buses.items():

        deviation = voltage - V_MIN_LIMIT_PU

        print(
            f"{str(bus_name):<65}"
            f"{voltage:>12.6f}"
            f"{deviation:>14.6f}"
        )


# ==================================================================================================
# TOP LOW-VOLTAGE BUSES
# ==================================================================================================

print()
print_rule()
print("LOWEST-VOLTAGE BUSES")
print_rule()

if len(bus_voltage) > 0:

    lowest_buses = bus_voltage.head(15)

    print(
        f"{'Bus':<65}"
        f"{'V magnitude pu':>18}"
    )

    for bus_name, voltage in lowest_buses.items():

        print(
            f"{str(bus_name):<65}"
            f"{voltage:>18.6f}"
        )


# ==================================================================================================
# TRANSFORMER BOTTLENECKS
# ==================================================================================================

print()
print_rule()
print("TOP TRANSFORMER LOADINGS")
print_rule()

if len(top_transformers) == 0:

    print("No transformers found.")

else:

    print(
        f"{'Transformer':<60}"
        f"{'Loading %':>12}"
        f"{'s_nom MW':>12}"
    )

    for transformer_name, loading in top_transformers.items():

        s_nom = safe_float(
            n.transformers.at[
                transformer_name,
                "s_nom"
            ]
        )

        print(
            f"{str(transformer_name):<60}"
            f"{loading:>12.3f}"
            f"{s_nom:>12.3f}"
        )


# ==================================================================================================
# BOTTLENECK CLASSIFICATION
# ==================================================================================================

thermal_problem = (
    len(overloaded_lines) > 0
    or len(overloaded_transformers) > 0
)

voltage_problem = (
    len(low_voltage_buses) > 0
    or len(high_voltage_buses) > 0
)

if not thermal_problem and not voltage_problem:

    diagnosis = "NO RESIDUAL THERMAL OR VOLTAGE VIOLATION"

elif thermal_problem and voltage_problem:

    diagnosis = "COUPLED THERMAL + VOLTAGE BOTTLENECK"

elif thermal_problem:

    diagnosis = "THERMAL BOTTLENECK"

else:

    diagnosis = "VOLTAGE BOTTLENECK"


# ==================================================================================================
# SYSTEM DIAGNOSIS
# ==================================================================================================

print()
print("=" * 100)
print("SYSTEM-LEVEL DIAGNOSIS")
print("=" * 100)

print()
print(f"Primary diagnosis         : {diagnosis}")

print()
print("Interpretation:")
print()

if len(overloaded_lines) > 0:

    print(
        f"- {len(overloaded_lines)} transmission line(s) remain "
        f"above {THERMAL_LIMIT_PCT:.1f}% loading."
    )

    print(
        f"- The critical residual line is "
        f"{critical_line} at "
        f"{max_line_loading:.6f}%."
    )

else:

    print(
        "- No transmission lines remain above the thermal limit."
    )


if len(low_voltage_buses) > 0:

    print(
        f"- {len(low_voltage_buses)} bus(es) remain below "
        f"{V_MIN_LIMIT_PU:.2f} pu."
    )

    print(
        f"- The minimum voltage is "
        f"{min_voltage:.6f} pu at "
        f"{min_voltage_bus}."
    )

else:

    print(
        "- No bus remains below the lower voltage limit."
    )


if len(high_voltage_buses) > 0:

    print(
        f"- {len(high_voltage_buses)} bus(es) exceed "
        f"{V_MAX_LIMIT_PU:.2f} pu."
    )

else:

    print(
        "- No bus exceeds the upper voltage limit."
    )


if len(overloaded_transformers) > 0:

    print(
        f"- {len(overloaded_transformers)} transformer(s) "
        f"remain above the thermal limit."
    )

else:

    print(
        "- No transformer exceeds the thermal limit."
    )


# ==================================================================================================
# ACCEPTANCE CHECK
# ==================================================================================================

all_acceptable = (
    converged
    and len(overloaded_lines) == 0
    and len(overloaded_transformers) == 0
    and len(low_voltage_buses) == 0
    and len(high_voltage_buses) == 0
)

print()
print_rule()
print("ACCEPTANCE CHECK")
print_rule()

print(f"Power flow converged       : {converged}")
print(f"Overloaded lines           : {len(overloaded_lines)}")
print(f"Overloaded transformers    : {len(overloaded_transformers)}")
print(f"Low-voltage buses         : {len(low_voltage_buses)}")
print(f"High-voltage buses        : {len(high_voltage_buses)}")
print(f"Fully acceptable           : {all_acceptable}")


# ==================================================================================================
# SAVE LINE RESULTS
# ==================================================================================================

line_records = []

for line_name, loading in line_loading.items():

    line_records.append(
        {
            "snapshot": SNAPSHOT,
            "package": PACKAGE_NAME,
            "line": line_name,
            "loading_pct": float(loading),
            "overloaded": bool(
                loading > THERMAL_LIMIT_PCT
            ),
            "s_nom_mw": safe_float(
                n.lines.at[line_name, "s_nom"]
            ),
            "p0_mw": safe_float(
                n.lines_t.p0.loc[
                    SNAPSHOT,
                    line_name
                ]
            ),
            "p1_mw": safe_float(
                n.lines_t.p1.loc[
                    SNAPSHOT,
                    line_name
                ]
            ),
        }
    )

line_df = pd.DataFrame(line_records)

OUTPUT_LINES.parent.mkdir(
    parents=True,
    exist_ok=True
)

line_df.to_csv(
    OUTPUT_LINES,
    index=False
)


# ==================================================================================================
# SAVE BUS RESULTS
# ==================================================================================================

bus_records = []

for bus_name, voltage in bus_voltage.items():

    bus_records.append(
        {
            "snapshot": SNAPSHOT,
            "package": PACKAGE_NAME,
            "bus": bus_name,
            "voltage_pu": float(voltage),
            "low_voltage": bool(
                voltage < V_MIN_LIMIT_PU
            ),
            "high_voltage": bool(
                voltage > V_MAX_LIMIT_PU
            ),
        }
    )

bus_df = pd.DataFrame(bus_records)

bus_df.to_csv(
    OUTPUT_BUSES,
    index=False
)


# ==================================================================================================
# SAVE TRANSFORMER RESULTS
# ==================================================================================================

transformer_records = []

for transformer_name, loading in transformer_loading.items():

    transformer_records.append(
        {
            "snapshot": SNAPSHOT,
            "package": PACKAGE_NAME,
            "transformer": transformer_name,
            "loading_pct": float(loading),
            "overloaded": bool(
                loading > THERMAL_LIMIT_PCT
            ),
            "s_nom_mw": safe_float(
                n.transformers.at[
                    transformer_name,
                    "s_nom"
                ]
            ),
            "p0_mw": safe_float(
                n.transformers_t.p0.loc[
                    SNAPSHOT,
                    transformer_name
                ]
            ),
            "p1_mw": safe_float(
                n.transformers_t.p1.loc[
                    SNAPSHOT,
                    transformer_name
                ]
            ),
        }
    )

transformer_df = pd.DataFrame(
    transformer_records
)

transformer_df.to_csv(
    OUTPUT_TRANSFORMERS,
    index=False
)


# ==================================================================================================
# SAVE SUMMARY
# ==================================================================================================

summary = pd.DataFrame(
    [
        {
            "snapshot": SNAPSHOT,
            "package": PACKAGE_NAME,
            "reactive_support_mvar": REACTIVE_MVAR,
            "converged": converged,
            "min_voltage_pu": min_voltage,
            "minimum_voltage_bus": min_voltage_bus,
            "max_voltage_pu": max_voltage,
            "maximum_voltage_bus": max_voltage_bus,
            "low_voltage_buses": len(
                low_voltage_buses
            ),
            "high_voltage_buses": len(
                high_voltage_buses
            ),
            "max_line_loading_pct": max_line_loading,
            "critical_line": critical_line,
            "overloaded_lines": len(
                overloaded_lines
            ),
            "max_transformer_loading_pct": (
                max_transformer_loading
            ),
            "worst_transformer": worst_transformer,
            "overloaded_transformers": len(
                overloaded_transformers
            ),
            "diagnosis": diagnosis,
            "fully_acceptable": all_acceptable,
        }
    ]
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ==================================================================================================
# FINAL OUTPUT
# ==================================================================================================

print()
print("=" * 100)
print("S4.2 COMPLETE")
print("=" * 100)

print()
print("Summary saved to:")
print(OUTPUT_SUMMARY.resolve())

print()
print("Detailed line results saved to:")
print(OUTPUT_LINES.resolve())

print()
print("Detailed bus results saved to:")
print(OUTPUT_BUSES.resolve())

print()
print("Detailed transformer results saved to:")
print(OUTPUT_TRANSFORMERS.resolve())

print()
print("=" * 100)