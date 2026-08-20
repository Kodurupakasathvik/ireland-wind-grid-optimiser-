# ==================================================================================================
#
# S4.5 — VOLTAGE BOTTLENECK DIAGNOSTIC
#
# Purpose
# -------
# Diagnose the residual voltage problem after:
#
#   P3 reinforcements
#   + ALL FOUR residual lines at 1.25x
#   + NO additional reactive support
#
# IMPORTANT
# ---------
# Source network is READ-ONLY.
# No .nc network file is modified.
#
# Snapshot:
#   S2_PEAK_DEMAND
#
# Power flow:
#   AC nonlinear
#
# Dispatch:
#   unchanged
#
# Loads:
#   unchanged
#
# Reactive support:
#   NONE
#
# IMPORTANT NUMERICAL RULE
# ------------------------
# If AC power flow does not converge, DO NOT extract or interpret
# voltage/loading values. PyPSA may contain numerically divergent
# values after a failed Newton-Raphson iteration.
#
# ==================================================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


warnings.filterwarnings("ignore")


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT = Path(
    "data/processed/s4_5_voltage_bottleneck_diagnostic.csv"
)

V_MIN_LIMIT = 0.95

# Numerical sanity limits.
# These are NOT engineering acceptance criteria.
# They are only used to detect obviously corrupted/divergent
# power-flow outputs.
V_SANITY_MIN = 0.50
V_SANITY_MAX = 1.50


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
# RESIDUAL LINES
# ==================================================================================================

RESIDUAL_LINES = [
    "way/235559472-220",
    "way/713396116-220",
    "way/42838773-220",
    "merged_way/516651706-220+2",
]

TARGET_MULTIPLIER = 1.25


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.5 — VOLTAGE BOTTLENECK DIAGNOSTIC")
print("=" * 100)

print()
print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print("PF       : AC nonlinear")
print("Dispatch : unchanged")
print("Loads    : unchanged")
print("Source   : READ-ONLY")

print()
print("Test package:")
print("  P3 reinforcements")
print("  + ALL FOUR residual lines at 1.25x")
print("  + NO additional reactive support")

print()
print("Voltage criterion:")
print(f"  Minimum acceptable voltage : {V_MIN_LIMIT:.2f} pu")


# ==================================================================================================
# LOAD NETWORK
# ==================================================================================================

print()
print("=" * 100)
print("LOADING SOURCE NETWORK")
print("=" * 100)

network = pypsa.Network(
    str(NETWORK_PATH)
)

if SNAPSHOT not in network.snapshots:
    raise ValueError(
        f"Snapshot '{SNAPSHOT}' not found."
    )

network.set_snapshots(
    network.snapshots
)

print(f"Buses        : {len(network.buses)}")
print(f"Lines        : {len(network.lines)}")
print(f"Transformers : {len(network.transformers)}")
print(f"Generators   : {len(network.generators)}")
print(f"Loads        : {len(network.loads)}")


# ==================================================================================================
# APPLY P3 REINFORCEMENTS
# ==================================================================================================

print()
print("=" * 100)
print("APPLYING P3 REINFORCEMENTS")
print("=" * 100)

for line, multiplier in P3_REINFORCEMENTS.items():

    if line not in network.lines.index:
        print()
        print(f"WARNING: P3 line not found: {line}")
        continue

    old_s_nom = float(
        network.lines.at[
            line,
            "s_nom",
        ]
    )

    new_s_nom = (
        old_s_nom
        * multiplier
    )

    network.lines.at[
        line,
        "s_nom",
    ] = new_s_nom

    print(
        f"{line:<65}"
        f"{multiplier:>7.2f}x"
        f"{old_s_nom:>12.3f}"
        f" -> "
        f"{new_s_nom:>12.3f} MVA"
    )


# ==================================================================================================
# APPLY RESIDUAL LINE REINFORCEMENTS
# ==================================================================================================

print()
print("=" * 100)
print("APPLYING RESIDUAL LINE REINFORCEMENTS")
print("=" * 100)

for line in RESIDUAL_LINES:

    if line not in network.lines.index:
        print()
        print(f"WARNING: residual line not found: {line}")
        continue

    old_s_nom = float(
        network.lines.at[
            line,
            "s_nom",
        ]
    )

    new_s_nom = (
        old_s_nom
        * TARGET_MULTIPLIER
    )

    network.lines.at[
        line,
        "s_nom",
    ] = new_s_nom

    print(
        f"{line:<65}"
        f"{TARGET_MULTIPLIER:>7.2f}x"
        f"{old_s_nom:>12.3f}"
        f" -> "
        f"{new_s_nom:>12.3f} MVA"
    )


# ==================================================================================================
# NO REACTIVE SUPPORT
# ==================================================================================================

print()
print("=" * 100)
print("REACTIVE SUPPORT")
print("=" * 100)

print(
    "No additional reactive support applied."
)

print(
    "The diagnostic intentionally preserves the original dispatch "
    "and reactive operating point."
)


# ==================================================================================================
# RUN AC NONLINEAR POWER FLOW
# ==================================================================================================

print()
print("=" * 100)
print("RUNNING AC NONLINEAR POWER FLOW")
print("=" * 100)

converged = False

try:

    pf_result = network.pf()

    # ------------------------------------------------------------------
    # PyPSA versions may return different structures.
    #
    # The safest validation is to inspect network.iterate_components()
    # result tables after PF and check for finite values.
    # ------------------------------------------------------------------

    print()
    print("AC nonlinear power flow returned.")

except Exception as exc:

    print()
    print("ERROR: AC nonlinear power flow raised an exception.")
    print(f"Exception: {exc}")

    pf_result = None


# ==================================================================================================
# CHECK POWER-FLOW CONVERGENCE
# ==================================================================================================

print()
print("=" * 100)
print("VALIDATING POWER-FLOW SOLUTION")
print("=" * 100)


# ------------------------------------------------------------------
# Check bus voltage values.
# ------------------------------------------------------------------

try:

    voltage_series = network.buses_t.v_mag_pu.loc[
        SNAPSHOT
    ].astype(float)

except Exception as exc:

    voltage_series = pd.Series(
        dtype=float
    )

    print(
        f"WARNING: Could not extract bus voltages: {exc}"
    )


# ------------------------------------------------------------------
# Basic finite-value check.
# ------------------------------------------------------------------

finite_voltage = (
    voltage_series.notna()
    &
    np.isfinite(voltage_series)
)


finite_count = int(
    finite_voltage.sum()
)

total_buses = int(
    len(voltage_series)
)


# ------------------------------------------------------------------
# Detect obviously divergent values.
# ------------------------------------------------------------------

if finite_count == 0:

    converged = False

    print(
        "RESULT: NO VALID VOLTAGE SOLUTION."
    )

elif finite_count < total_buses:

    converged = False

    print(
        "RESULT: INVALID POWER-FLOW SOLUTION."
    )

    print(
        f"Finite bus voltages : {finite_count}/{total_buses}"
    )

else:

    valid_voltage = voltage_series[
        finite_voltage
    ]

    min_v_check = float(
        valid_voltage.min()
    )

    max_v_check = float(
        valid_voltage.max()
    )

    if (
        min_v_check < V_SANITY_MIN
        or
        max_v_check > V_SANITY_MAX
    ):

        converged = False

        print(
            "RESULT: POWER-FLOW SOLUTION IS NUMERICALLY INVALID."
        )

        print(
            f"Voltage range observed : "
            f"{min_v_check:.6f} -> {max_v_check:.6f} pu"
        )

        print()
        print(
            "The values are outside the diagnostic sanity range."
        )

        print(
            "This indicates numerical divergence rather than a "
            "physical operating point."
        )

    else:

        converged = True

        print(
            "RESULT: VALID NUMERICAL VOLTAGE SOLUTION."
        )

        print(
            f"Voltage range : "
            f"{min_v_check:.6f} -> {max_v_check:.6f} pu"
        )


# ==================================================================================================
# STOP IF POWER FLOW IS INVALID
# ==================================================================================================

if not converged:

    print()
    print("=" * 100)
    print("S4.5 STOPPED — INVALID AC POWER-FLOW SOLUTION")
    print("=" * 100)

    print()
    print(
        "The AC nonlinear power flow did not produce a valid "
        "physical solution."
    )

    print()
    print("IMPORTANT:")
    print(
        "The voltage and line-loading values from this failed "
        "iteration MUST NOT be interpreted."
    )

    print()
    print("Likely issue:")
    print(
        "  Newton-Raphson AC power flow divergence under the "
        "P3 + ALL-FOUR 1.25x configuration."
    )

    print()
    print("Next diagnostic step:")
    print(
        "  Investigate the AC power-flow feasibility/convergence "
        "before performing voltage-bottleneck analysis."
    )

    print()
    print(
        "No network file was modified."
    )

    print("=" * 100)

    # Save a machine-readable failure record.

    failure_result = pd.DataFrame(
        [
            {
                "candidate": "P3_PLUS_ALL4_1.25X",
                "snapshot": SNAPSHOT,
                "power_flow": "AC_NONLINEAR",
                "converged": False,
                "valid_solution": False,
                "min_voltage_pu": np.nan,
                "max_voltage_pu": np.nan,
                "low_voltage_buses": np.nan,
                "max_line_loading_pct": np.nan,
                "overloaded_lines": np.nan,
                "max_transformer_loading_pct": np.nan,
                "status": "AC_POWER_FLOW_NONCONVERGED",
            }
        ]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    failure_result.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("Diagnostic status saved to:")
    print(f"  {OUTPUT}")

    print("=" * 100)

    raise SystemExit(0)


# ==================================================================================================
# EXTRACT VALID VOLTAGE RESULTS
# ==================================================================================================

print()
print("=" * 100)
print("EXTRACTING VALID VOLTAGE RESULTS")
print("=" * 100)

valid_voltage = (
    voltage_series[
        finite_voltage
    ]
    .sort_values()
)


# ==================================================================================================
# SYSTEM VOLTAGE SUMMARY
# ==================================================================================================

min_voltage = float(
    valid_voltage.min()
)

max_voltage = float(
    valid_voltage.max()
)

min_voltage_bus = str(
    valid_voltage.idxmin()
)

max_voltage_bus = str(
    valid_voltage.idxmax()
)

low_voltage_buses = int(
    (
        valid_voltage
        < V_MIN_LIMIT
    ).sum()
)

print()
print("=" * 100)
print("SYSTEM VOLTAGE SUMMARY")
print("=" * 100)

print()
print(
    f"Minimum voltage           : {min_voltage:.6f} pu"
)

print(
    f"Minimum-voltage bus       : {min_voltage_bus}"
)

print(
    f"Maximum voltage           : {max_voltage:.6f} pu"
)

print(
    f"Maximum-voltage bus       : {max_voltage_bus}"
)

print(
    f"Voltage limit             : {V_MIN_LIMIT:.6f} pu"
)

print(
    f"Low-voltage buses         : {low_voltage_buses}"
)


# ==================================================================================================
# LOW-VOLTAGE BUS DIAGNOSTIC
# ==================================================================================================

print()
print("=" * 100)
print("LOW-VOLTAGE BUS DIAGNOSTIC")
print("=" * 100)

low_voltage = (
    valid_voltage[
        valid_voltage < V_MIN_LIMIT
    ]
)

if len(low_voltage) == 0:

    print()
    print("No buses below the 0.95 pu voltage criterion.")

else:

    print()
    print(
        f"{'BUS':<65}"
        f"{'V (pu)':>12}"
        f"{'DEFICIT (pu)':>16}"
    )

    print("-" * 95)

    for bus, voltage in low_voltage.items():

        deficit = (
            V_MIN_LIMIT
            - float(voltage)
        )

        print(
            f"{str(bus):<65}"
            f"{float(voltage):>12.6f}"
            f"{deficit:>16.6f}"
        )


# ==================================================================================================
# VOLTAGE BAND DISTRIBUTION
# ==================================================================================================

v_lt_090 = int(
    (
        valid_voltage < 0.90
    ).sum()
)

v_090_095 = int(
    (
        (valid_voltage >= 0.90)
        &
        (valid_voltage < 0.95)
    ).sum()
)

v_095_105 = int(
    (
        (valid_voltage >= 0.95)
        &
        (valid_voltage <= 1.05)
    ).sum()
)

v_105_110 = int(
    (
        (valid_voltage > 1.05)
        &
        (valid_voltage <= 1.10)
    ).sum()
)

v_gt_110 = int(
    (
        valid_voltage > 1.10
    ).sum()
)


print()
print("=" * 100)
print("VOLTAGE BAND DISTRIBUTION")
print("=" * 100)

print()
print(
    f"V < 0.90 pu          : {v_lt_090}"
)

print(
    f"0.90 <= V < 0.95 pu  : {v_090_095}"
)

print(
    f"0.95 <= V <= 1.05 pu : {v_095_105}"
)

print(
    f"1.05 < V <= 1.10 pu  : {v_105_110}"
)

print(
    f"V > 1.10 pu          : {v_gt_110}"
)


# ==================================================================================================
# LINE LOADING DIAGNOSTIC
# ==================================================================================================

print()
print("=" * 100)
print("LINE LOADING DIAGNOSTIC")
print("=" * 100)


try:

    line_loading = (
        network.lines_t
        .get("loading")
        .loc[SNAPSHOT]
        .astype(float)
    )

except Exception:

    try:

        # PyPSA normally provides p0/p1 rather than a permanent
        # loading table. Calculate apparent power from p and q.

        p0 = (
            network.lines_t.p0
            .loc[SNAPSHOT]
            .astype(float)
        )

        q0 = (
            network.lines_t.q0
            .loc[SNAPSHOT]
            .astype(float)
        )

        apparent_power = np.sqrt(
            p0**2 + q0**2
        )

        s_nom = (
            network.lines.s_nom
            .astype(float)
        )

        line_loading = (
            apparent_power
            / s_nom
            * 100.0
        )

    except Exception as exc:

        print()
        print(
            f"WARNING: Could not calculate line loading: {exc}"
        )

        line_loading = pd.Series(
            dtype=float
        )


finite_line_loading = (
    line_loading.notna()
    &
    np.isfinite(line_loading)
)

line_loading = line_loading[
    finite_line_loading
]

if len(line_loading) == 0:

    max_line_loading = np.nan
    overloaded_lines = 0

else:

    max_line_loading = float(
        line_loading.max()
    )

    overloaded_lines = int(
        (
            line_loading > 100.0
        ).sum()
    )


print()
print(
    f"Maximum line loading : "
    f"{max_line_loading:.6f} %"
)

print(
    f"Overloaded lines      : "
    f"{overloaded_lines}"
)


if overloaded_lines > 0:

    print()
    print("OVERLOADED LINES")
    print("-" * 95)

    overloaded = (
        line_loading[
            line_loading > 100.0
        ]
        .sort_values(
            ascending=False
        )
    )

    for line, loading in overloaded.items():

        print(
            f"{str(line):<65}"
            f"{float(loading):>12.6f} %"
        )


# ==================================================================================================
# TRANSFORMER LOADING DIAGNOSTIC
# ==================================================================================================

print()
print("=" * 100)
print("TRANSFORMER LOADING DIAGNOSTIC")
print("=" * 100)


try:

    transformer_p0 = (
        network.transformers_t.p0
        .loc[SNAPSHOT]
        .astype(float)
    )

    transformer_q0 = (
        network.transformers_t.q0
        .loc[SNAPSHOT]
        .astype(float)
    )

    transformer_s = np.sqrt(
        transformer_p0**2
        +
        transformer_q0**2
    )

    transformer_s_nom = (
        network.transformers.s_nom
        .astype(float)
    )

    transformer_loading = (
        transformer_s
        /
        transformer_s_nom
        *
        100.0
    )

    transformer_loading = (
        transformer_loading[
            transformer_loading.notna()
            &
            np.isfinite(
                transformer_loading
            )
        ]
    )

except Exception as exc:

    print()
    print(
        f"WARNING: Could not calculate transformer loading: {exc}"
    )

    transformer_loading = pd.Series(
        dtype=float
    )


if len(transformer_loading) == 0:

    max_transformer_loading = np.nan
    worst_transformer = "N/A"
    overloaded_transformers = 0

else:

    max_transformer_loading = float(
        transformer_loading.max()
    )

    worst_transformer = str(
        transformer_loading.idxmax()
    )

    overloaded_transformers = int(
        (
            transformer_loading > 100.0
        ).sum()
    )


print()
print(
    f"Maximum transformer loading : "
    f"{max_transformer_loading:.6f} %"
)

print(
    f"Overloaded transformers      : "
    f"{overloaded_transformers}"
)

print(
    f"Worst transformer            : "
    f"{worst_transformer}"
)


# ==================================================================================================
# SAVE DIAGNOSTIC RESULTS
# ==================================================================================================

print()
print("=" * 100)
print("SAVING DIAGNOSTIC RESULTS")
print("=" * 100)


result = pd.DataFrame(
    [
        {
            "candidate": "P3_PLUS_ALL4_1.25X",
            "category": "VOLTAGE_BOTTLENECK_DIAGNOSTIC",
            "snapshot": SNAPSHOT,
            "power_flow": "AC_NONLINEAR",
            "converged": True,
            "valid_solution": True,

            "target_multiplier": TARGET_MULTIPLIER,

            "min_voltage_pu": min_voltage,
            "min_voltage_bus": min_voltage_bus,
            "max_voltage_pu": max_voltage,
            "max_voltage_bus": max_voltage_bus,

            "voltage_limit_pu": V_MIN_LIMIT,
            "low_voltage_buses": low_voltage_buses,

            "v_lt_090": v_lt_090,
            "v_090_095": v_090_095,
            "v_095_105": v_095_105,
            "v_105_110": v_105_110,
            "v_gt_110": v_gt_110,

            "max_line_loading_pct": max_line_loading,
            "overloaded_lines": overloaded_lines,

            "max_transformer_loading_pct":
                max_transformer_loading,

            "overloaded_transformers":
                overloaded_transformers,

            "worst_transformer":
                worst_transformer,

            "status":
                "VALID_AC_SOLUTION",
        }
    ]
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    "Results saved to:"
)

print(
    f"  {OUTPUT}"
)


# ==================================================================================================
# INTERPRETATION
# ==================================================================================================

print()
print("=" * 100)
print("S4.5 DIAGNOSTIC INTERPRETATION")
print("=" * 100)

print()

if low_voltage_buses == 0:

    print(
        "No residual undervoltage buses were detected."
    )

else:

    print(
        f"Residual undervoltage buses : "
        f"{low_voltage_buses}"
    )

    print(
        f"Minimum voltage             : "
        f"{min_voltage:.6f} pu"
    )

    print(
        f"Critical voltage bus        : "
        f"{min_voltage_bus}"
    )

    print(
        f"Voltage deficit             : "
        f"{V_MIN_LIMIT - min_voltage:.6f} pu"
    )


print()

if overloaded_lines == 0:

    print(
        "No overloaded lines detected."
    )

else:

    print(
        f"Maximum line loading        : "
        f"{max_line_loading:.6f} %"
    )

    print(
        f"Overloaded lines             : "
        f"{overloaded_lines}"
    )


print()

print(
    f"Maximum transformer loading : "
    f"{max_transformer_loading:.6f} %"
)

print()
print(
    "S4.5 diagnostic complete."
)

print()
print(
    "IMPORTANT:"
)

print(
    "No network file was modified."
)

print(
    "This stage performs diagnosis only."
)

print("=" * 100)