# ==================================================================================================
#
# S4.5B — AC POWER-FLOW CONVERGENCE DIAGNOSTIC
#
# Purpose
# -------
# Diagnose AC nonlinear power-flow convergence after:
#
#   P3 reinforcements
#   + ALL FOUR residual lines at 1.25x
#
# IMPORTANT
# ---------
# Source network is READ-ONLY.
# No .nc network file is modified.
#
# This stage:
#   1. Loads the original network.
#   2. Explicitly isolates S2_PEAK_DEMAND.
#   3. Applies P3 reinforcements.
#   4. Applies all four residual-line reinforcements.
#   5. Preserves generator dispatch and loads.
#   6. Runs AC nonlinear power flow.
#   7. Checks numerical validity.
#   8. Diagnoses generator, bus, transformer and line data.
#
# NO voltage-bottleneck interpretation is performed if PF fails.
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
    "data/processed/s4_5b_ac_convergence_diagnostic.csv"
)

V_MIN_LIMIT = 0.80
V_MAX_LIMIT = 1.20

P3_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.75,
    "merged_way/61295764-220+1": 2.00,
    "way/343436171-220": 2.00,
    "merged_way/257889771-220+1": 1.75,
    "merged_relation/4872159-220+1": 1.75,
}

RESIDUAL_LINES = [
    "way/235559472-220",
    "way/713396116-220",
    "way/42838773-220",
    "merged_way/516651706-220+2",
]

RESIDUAL_MULTIPLIER = 1.25


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.5B — AC POWER-FLOW CONVERGENCE DIAGNOSTIC")
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

print(f"Buses        : {len(network.buses)}")
print(f"Lines        : {len(network.lines)}")
print(f"Transformers : {len(network.transformers)}")
print(f"Generators   : {len(network.generators)}")
print(f"Loads        : {len(network.loads)}")

if SNAPSHOT not in network.snapshots:
    raise ValueError(
        f"Snapshot '{SNAPSHOT}' not found."
    )


# ==================================================================================================
# EXPLICIT SNAPSHOT ISOLATION
# ==================================================================================================

print()
print("=" * 100)
print("ISOLATING TARGET SNAPSHOT")
print("=" * 100)

print("Available snapshots:")
for snapshot in network.snapshots:
    print(f"  {snapshot}")

print()
print(f"Selecting ONLY: {SNAPSHOT}")

network.set_snapshots(
    [SNAPSHOT]
)

print()
print("Active snapshots after isolation:")
for snapshot in network.snapshots:
    print(f"  {snapshot}")

if len(network.snapshots) != 1 or network.snapshots[0] != SNAPSHOT:
    raise RuntimeError(
        "Snapshot isolation failed."
    )


# ==================================================================================================
# APPLY P3 REINFORCEMENTS
# ==================================================================================================

print()
print("=" * 100)
print("APPLYING P3 REINFORCEMENTS")
print("=" * 100)

for line, multiplier in P3_REINFORCEMENTS.items():

    if line not in network.lines.index:
        raise KeyError(
            f"P3 line not found: {line}"
        )

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
# APPLY RESIDUAL REINFORCEMENTS
# ==================================================================================================

print()
print("=" * 100)
print("APPLYING RESIDUAL LINE REINFORCEMENTS")
print("=" * 100)

for line in RESIDUAL_LINES:

    if line not in network.lines.index:
        raise KeyError(
            f"Residual line not found: {line}"
        )

    old_s_nom = float(
        network.lines.at[
            line,
            "s_nom",
        ]
    )

    new_s_nom = (
        old_s_nom
        * RESIDUAL_MULTIPLIER
    )

    network.lines.at[
        line,
        "s_nom",
    ] = new_s_nom

    print(
        f"{line:<65}"
        f"{RESIDUAL_MULTIPLIER:>7.2f}x"
        f"{old_s_nom:>12.3f}"
        f" -> "
        f"{new_s_nom:>12.3f} MVA"
    )


# ==================================================================================================
# GENERATOR DIAGNOSTIC — BEFORE PF
# ==================================================================================================

print()
print("=" * 100)
print("GENERATOR OPERATING-POINT DIAGNOSTIC")
print("=" * 100)

generator_rows = []

for generator in network.generators.index:

    bus = network.generators.at[
        generator,
        "bus",
    ]

    p_nom = network.generators.at[
        generator,
        "p_nom",
    ]

    control = network.generators.at[
        generator,
        "control",
    ]

    try:
        p_set = float(
            network.generators_t.p_set.at[
                SNAPSHOT,
                generator,
            ]
        )
    except Exception:
        p_set = np.nan

    try:
        q_set = float(
            network.generators_t.q_set.at[
                SNAPSHOT,
                generator,
            ]
        )
    except Exception:
        q_set = np.nan

    try:
        v_set = float(
            network.generators_t.control.at[
                SNAPSHOT,
                generator,
            ]
        )
    except Exception:
        try:
            v_set = float(
                network.generators.at[
                    generator,
                    "v_set",
                ]
            )
        except Exception:
            v_set = np.nan

    generator_rows.append(
        {
            "generator": generator,
            "bus": bus,
            "p_nom_mw": p_nom,
            "p_set_mw": p_set,
            "q_set_mvar": q_set,
            "control": control,
            "v_set_pu": v_set,
        }
    )

generator_df = pd.DataFrame(
    generator_rows
)

print(
    generator_df.to_string(
        index=False
    )
)


# ==================================================================================================
# LOAD DIAGNOSTIC
# ==================================================================================================

print()
print("=" * 100)
print("LOAD OPERATING-POINT DIAGNOSTIC")
print("=" * 100)

load_rows = []

for load in network.loads.index:

    bus = network.loads.at[
        load,
        "bus",
    ]

    try:
        p_set = float(
            network.loads_t.p_set.at[
                SNAPSHOT,
                load,
            ]
        )
    except Exception:
        p_set = np.nan

    try:
        q_set = float(
            network.loads_t.q_set.at[
                SNAPSHOT,
                load,
            ]
        )
    except Exception:
        q_set = np.nan

    load_rows.append(
        {
            "load": load,
            "bus": bus,
            "p_set_mw": p_set,
            "q_set_mvar": q_set,
        }
    )

load_df = pd.DataFrame(
    load_rows
)

print(
    load_df.to_string(
        index=False
    )
)


# ==================================================================================================
# BUS DATA DIAGNOSTIC
# ==================================================================================================

print()
print("=" * 100)
print("BUS DATA SANITY CHECK")
print("=" * 100)

bus_rows = []

for bus in network.buses.index:

    v_nom = network.buses.at[
        bus,
        "v_nom",
    ]

    carrier = network.buses.at[
        bus,
        "carrier",
    ]

    bus_rows.append(
        {
            "bus": bus,
            "v_nom_kv": v_nom,
            "carrier": carrier,
        }
    )

bus_df = pd.DataFrame(
    bus_rows
)

print(
    bus_df.to_string(
        index=False
    )
)


# ==================================================================================================
# TRANSFORMER DIAGNOSTIC
# ==================================================================================================

print()
print("=" * 100)
print("TRANSFORMER SANITY CHECK")
print("=" * 100)

if len(network.transformers) == 0:

    print("No transformers found.")

else:

    transformer_rows = []

    for transformer in network.transformers.index:

        transformer_rows.append(
            {
                "transformer": transformer,
                "bus0": network.transformers.at[
                    transformer,
                    "bus0",
                ],
                "bus1": network.transformers.at[
                    transformer,
                    "bus1",
                ],
                "s_nom_mva": network.transformers.at[
                    transformer,
                    "s_nom",
                ],
                "x_pu": network.transformers.at[
                    transformer,
                    "x",
                ],
                "r_pu": network.transformers.at[
                    transformer,
                    "r",
                ],
                "tap_ratio": network.transformers.at[
                    transformer,
                    "tap_ratio",
                ],
                "phase_shift": network.transformers.at[
                    transformer,
                    "phase_shift",
                ],
            }
        )

    transformer_df = pd.DataFrame(
        transformer_rows
    )

    print(
        transformer_df.to_string(
            index=False
        )
    )


# ==================================================================================================
# LINE IMPEDANCE SANITY CHECK
# ==================================================================================================

print()
print("=" * 100)
print("LINE IMPEDANCE SANITY CHECK")
print("=" * 100)

bad_lines = []

for line in network.lines.index:

    r = network.lines.at[
        line,
        "r",
    ]

    x = network.lines.at[
        line,
        "x",
    ]

    b = network.lines.at[
        line,
        "b",
    ]

    s_nom = network.lines.at[
        line,
        "s_nom",
    ]

    if not np.isfinite(r):
        bad_lines.append(
            (line, "non-finite r")
        )

    if not np.isfinite(x):
        bad_lines.append(
            (line, "non-finite x")
        )

    if not np.isfinite(b):
        bad_lines.append(
            (line, "non-finite b")
        )

    if not np.isfinite(s_nom):
        bad_lines.append(
            (line, "non-finite s_nom")
        )

    if s_nom <= 0:
        bad_lines.append(
            (line, "non-positive s_nom")
        )

    if abs(x) < 1e-12:
        bad_lines.append(
            (line, "near-zero x")
        )

if bad_lines:

    print("Potentially problematic lines:")

    for line, reason in bad_lines:
        print(
            f"  {line:<65} {reason}"
        )

else:

    print(
        "No obvious invalid line parameters detected."
    )


# ==================================================================================================
# NETWORK POWER BALANCE CHECK
# ==================================================================================================

print()
print("=" * 100)
print("NETWORK POWER BALANCE CHECK")
print("=" * 100)

try:

    total_generation = float(
        network.generators_t.p_set.loc[
            SNAPSHOT
        ].sum()
    )

except Exception:

    total_generation = np.nan

try:

    total_load = float(
        network.loads_t.p_set.loc[
            SNAPSHOT
        ].sum()
    )

except Exception:

    total_load = np.nan

print(
    f"Total generator P set : {total_generation:.6f} MW"
)

print(
    f"Total load P set      : {total_load:.6f} MW"
)

if np.isfinite(total_generation) and np.isfinite(total_load):

    print(
        f"Generation - Load    : "
        f"{total_generation - total_load:.6f} MW"
    )


# ==================================================================================================
# AC POWER FLOW
# ==================================================================================================

print()
print("=" * 100)
print("RUNNING AC NONLINEAR POWER FLOW")
print("=" * 100)

pf_exception = None

try:

    pf_result = network.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-8,
        use_seed=True,
    )

    print()
    print("AC nonlinear power flow returned.")

except Exception as exc:

    pf_result = None

    pf_exception = repr(exc)

    print()
    print("AC nonlinear power flow raised an exception:")
    print(pf_exception)


# ==================================================================================================
# CONVERGENCE CHECK
# ==================================================================================================

print()
print("=" * 100)
print("VALIDATING POWER-FLOW SOLUTION")
print("=" * 100)

converged = False

if pf_result is not None:

    try:

        if isinstance(
            pf_result,
            tuple
        ):

            # PyPSA commonly returns:
            # (converged, error)

            converged_data = pf_result[0]

        else:

            converged_data = pf_result

        if isinstance(
            converged_data,
            pd.DataFrame
        ):

            if SNAPSHOT in converged_data.index:

                converged = bool(
                    converged_data.loc[
                        SNAPSHOT
                    ].all()
                )

            elif SNAPSHOT in converged_data.columns:

                converged = bool(
                    converged_data[
                        SNAPSHOT
                    ].all()
                )

        elif np.isscalar(
            converged_data
        ):

            converged = bool(
                converged_data
            )

    except Exception as exc:

        print(
            "Could not directly determine convergence flag:"
        )

        print(
            f"  {repr(exc)}"
        )


# ==================================================================================================
# EXTRACT VOLTAGES ONLY IF NUMERICALLY VALID
# ==================================================================================================

valid_solution = False

voltage_series = None

if pf_result is not None:

    try:

        voltage_series = (
            network.buses_t.v_mag_pu.loc[
                SNAPSHOT
            ]
        )

        voltage_values = (
            voltage_series
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(voltage_values) > 0:

            finite = np.all(
                np.isfinite(
                    voltage_values
                )
            )

            physical = (
                voltage_values.min()
                >= V_MIN_LIMIT
                and
                voltage_values.max()
                <= V_MAX_LIMIT
            )

            valid_solution = (
                converged
                and
                finite
                and
                physical
            )

    except Exception as exc:

        print(
            "Voltage extraction failed:"
        )

        print(
            f"  {repr(exc)}"
        )


# ==================================================================================================
# RESULT
# ==================================================================================================

print()
print("=" * 100)
print("S4.5B RESULT")
print("=" * 100)

print(
    f"Convergence flag : {converged}"
)

print(
    f"Numerically valid physical solution : "
    f"{valid_solution}"
)

if pf_exception is not None:

    print()
    print("PF exception:")
    print(
        pf_exception
    )


# ==================================================================================================
# IF INVALID — DIAGNOSTIC ONLY
# ==================================================================================================

if not valid_solution:

    print()
    print("=" * 100)
    print("AC POWER-FLOW SOLUTION IS INVALID")
    print("=" * 100)

    print()
    print(
        "The current configuration cannot be used for "
        "voltage-bottleneck analysis."
    )

    print()
    print("Diagnostic focus:")
    print("  1. Snapshot isolation")
    print("  2. Generator operating points")
    print("  3. Load operating points")
    print("  4. Bus voltage bases")
    print("  5. Transformer parameters")
    print("  6. Line impedance sanity")
    print("  7. Network P balance")
    print("  8. AC nonlinear convergence")

else:

    print()
    print("=" * 100)
    print("VALID AC SOLUTION CONFIRMED")
    print("=" * 100)

    min_voltage = float(
        voltage_values.min()
    )

    max_voltage = float(
        voltage_values.max()
    )

    min_bus = (
        voltage_series.idxmin()
    )

    max_bus = (
        voltage_series.idxmax()
    )

    print(
        f"Minimum voltage : {min_voltage:.6f} pu"
    )

    print(
        f"Minimum bus     : {min_bus}"
    )

    print(
        f"Maximum voltage : {max_voltage:.6f} pu"
    )

    print(
        f"Maximum bus     : {max_bus}"
    )


# ==================================================================================================
# BUILD SUMMARY
# ==================================================================================================

summary = {
    "stage": "S4.5B",
    "snapshot": SNAPSHOT,
    "converged": converged,
    "valid_physical_solution": valid_solution,
    "pf_exception": pf_exception,
    "total_generation_mw": total_generation,
    "total_load_mw": total_load,
    "generation_minus_load_mw": (
        total_generation - total_load
        if np.isfinite(total_generation)
        and np.isfinite(total_load)
        else np.nan
    ),
    "network_buses": len(network.buses),
    "network_lines": len(network.lines),
    "network_transformers": len(network.transformers),
    "network_generators": len(network.generators),
    "network_loads": len(network.loads),
    "bad_line_parameters": len(bad_lines),
}


# ==================================================================================================
# SAVE
# ==================================================================================================

print()
print("=" * 100)
print("SAVING DIAGNOSTIC RESULTS")
print("=" * 100)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

pd.DataFrame(
    [summary]
).to_csv(
    OUTPUT,
    index=False
)

print()
print("Results saved to:")
print(
    f"  {OUTPUT}"
)


# ==================================================================================================
# FINAL STATUS
# ==================================================================================================

print()
print("=" * 100)

if valid_solution:

    print(
        "S4.5B COMPLETE — VALID AC SOLUTION CONFIRMED."
    )

else:

    print(
        "S4.5B COMPLETE — AC CONVERGENCE PROBLEM CONFIRMED."
    )

print("=" * 100)

print()
print("IMPORTANT:")
print("No network file was modified.")
print("No invalid voltage/loading values were interpreted.")
print("Only S2_PEAK_DEMAND was evaluated.")
print("=" * 100)