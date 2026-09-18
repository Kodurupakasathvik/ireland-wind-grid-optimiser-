# ==================================================================================================
# S3.14 — FINAL TARGETED THERMAL BOTTLENECK REINFORCEMENT TEST
# ==================================================================================================

import os
import pandas as pd
import pypsa


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = (
    r"C:\Users\Dell\ireland-wind-grid-optimiser"
    r"\data\processed\eirgrid_second_reinforced_network.nc"
)

OUTPUT_PATH = (
    r"C:\Users\Dell\ireland-wind-grid-optimiser"
    r"\data\processed\s3_14_final_targeted_thermal_reinforcement_results.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"

LAMBDA = 0.953125

WEAK_BUS = "way/104388595-220"

Q_SUPPORT = 500.0

# --------------------------------------------------------------------------------
# Fixed reinforcements from S3.10–S3.13
# --------------------------------------------------------------------------------

FIXED_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.50,
    "merged_way/61295764-220+1": 1.50,
    "way/343436171-220": 1.50,
    "merged_way/257889771-220+1": 1.25,
}

# --------------------------------------------------------------------------------
# New target identified by S3.13
# --------------------------------------------------------------------------------

TARGET_LINE = "merged_relation/4872159-220+1"

TEST_MULTIPLIERS = [1.25, 1.50, 1.75, 2.00]

# Generator located at the weak bus.
#
# NOTE:
# The generator exists in n.generators, but the original network has
# an EMPTY generators_t.q_set table. The script therefore creates the
# required q_set table automatically.
GENERATOR_NAME = "eirgrid_wind_way/104388595-220"


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 110)
print("S3.14 — FINAL TARGETED THERMAL BOTTLENECK REINFORCEMENT TEST")
print("=" * 110)

print(f"Network       : {NETWORK_PATH}")
print(f"Snapshot      : {SNAPSHOT}")
print(f"Lambda        : {LAMBDA}")
print(f"Weak bus      : {WEAK_BUS}")
print(f"Q support     : +{Q_SUPPORT:.0f} MVAr")

print()
print("FIXED REINFORCEMENTS")
print("-" * 110)

for line_name, multiplier in FIXED_REINFORCEMENTS.items():
    print(f"{line_name:<55} : {multiplier:.2f}x")

print()
print("NEW TARGET")
print("-" * 110)
print(f"{TARGET_LINE}")
print(f"Test multipliers: {TEST_MULTIPLIERS}")

print()
print("=" * 110)
print("TESTING FINAL TARGETED THERMAL BOTTLENECK")
print("=" * 110)


# ==================================================================================================
# HELPER — ENSURE Q SET TABLE EXISTS
# ==================================================================================================

def ensure_q_set_table(network):
    """
    Ensure generators_t.q_set exists with all generator columns.

    The supplied network currently has:
        n.generators_t.q_set.columns == []

    Therefore we create a zero-valued table covering all snapshots
    and all generators.
    """

    # If q_set is completely empty, construct it.
    if network.generators_t.q_set.empty:
        network.generators_t.q_set = pd.DataFrame(
            0.0,
            index=network.snapshots,
            columns=network.generators.index,
        )

    else:
        # Add missing generator columns.
        for generator in network.generators.index:
            if generator not in network.generators_t.q_set.columns:
                network.generators_t.q_set[generator] = 0.0

        # Ensure all snapshots exist.
        for snapshot in network.snapshots:
            if snapshot not in network.generators_t.q_set.index:
                network.generators_t.q_set.loc[snapshot] = 0.0

        # Keep rows ordered like network snapshots.
        network.generators_t.q_set = network.generators_t.q_set.reindex(
            index=network.snapshots,
            columns=network.generators.index,
            fill_value=0.0,
        )


# ==================================================================================================
# LOAD ORIGINAL NETWORK ONCE FOR REFERENCE
# ==================================================================================================

if not os.path.exists(NETWORK_PATH):
    raise FileNotFoundError(
        f"\nNetwork file not found:\n{NETWORK_PATH}"
    )

base_network = pypsa.Network(NETWORK_PATH)


# ==================================================================================================
# VALIDATE SNAPSHOT
# ==================================================================================================

if SNAPSHOT not in base_network.snapshots:
    raise ValueError(
        f"\nSnapshot '{SNAPSHOT}' not found.\n"
        f"Available snapshots:\n{list(base_network.snapshots)}"
    )


# ==================================================================================================
# VALIDATE FIXED REINFORCEMENTS
# ==================================================================================================

for line_name in FIXED_REINFORCEMENTS:

    if line_name not in base_network.lines.index:
        raise ValueError(
            f"\nFixed reinforcement line not found:\n{line_name}"
        )


# ==================================================================================================
# VALIDATE TARGET
# ==================================================================================================

if TARGET_LINE not in base_network.lines.index:
    raise ValueError(
        f"\nTarget line not found:\n{TARGET_LINE}"
    )


# ==================================================================================================
# VALIDATE GENERATOR
# ==================================================================================================

if GENERATOR_NAME not in base_network.generators.index:

    print()
    print("ERROR — requested Q-support generator was not found.")
    print()
    print("Available generators:")
    for generator in base_network.generators.index:
        print(f"  {generator}")

    raise ValueError(
        f"\nGenerator not found:\n{GENERATOR_NAME}"
    )


# ==================================================================================================
# NETWORK SUMMARY
# ==================================================================================================

print()
print("NETWORK")
print("-" * 110)

print(f"Buses        : {len(base_network.buses)}")
print(f"Lines        : {len(base_network.lines)}")
print(f"Transformers : {len(base_network.transformers)}")
print(f"Generators   : {len(base_network.generators)}")
print(f"Loads        : {len(base_network.loads)}")
print(f"Snapshots    : {list(base_network.snapshots)}")


# ==================================================================================================
# RESULTS STORAGE
# ==================================================================================================

results = []


# ==================================================================================================
# RUN TESTS
# ==================================================================================================

for multiplier in TEST_MULTIPLIERS:

    print()
    print("-" * 110)

    print(
        f"TEST — fixed reinforcements + "
        f"{TARGET_LINE} — {multiplier:.2f}x — "
        f"+{Q_SUPPORT:.0f} MVAr"
    )

    print("-" * 110)

    # ----------------------------------------------------------------------------------------------
    # Fresh network for each test
    # ----------------------------------------------------------------------------------------------

    n = pypsa.Network(NETWORK_PATH)

    # ----------------------------------------------------------------------------------------------
    # Apply fixed reinforcements
    # ----------------------------------------------------------------------------------------------

    for line_name, fixed_multiplier in FIXED_REINFORCEMENTS.items():

        original_s_nom = float(n.lines.at[line_name, "s_nom"])

        new_s_nom = original_s_nom * fixed_multiplier

        n.lines.at[line_name, "s_nom"] = new_s_nom

        print(
            f"{line_name:<55} "
            f"{original_s_nom:.6f} -> {new_s_nom:.6f} MW"
        )

    # ----------------------------------------------------------------------------------------------
    # Apply new target reinforcement
    # ----------------------------------------------------------------------------------------------

    original_target_s_nom = float(
        n.lines.at[TARGET_LINE, "s_nom"]
    )

    new_target_s_nom = (
        original_target_s_nom * multiplier
    )

    n.lines.at[TARGET_LINE, "s_nom"] = new_target_s_nom

    print(
        f"{TARGET_LINE:<55} "
        f"{original_target_s_nom:.6f} -> "
        f"{new_target_s_nom:.6f} MW"
    )

    # ----------------------------------------------------------------------------------------------
    # Apply reactive support
    # ----------------------------------------------------------------------------------------------

    ensure_q_set_table(n)

    old_q = float(
        n.generators_t.q_set.loc[
            SNAPSHOT,
            GENERATOR_NAME
        ]
    )

    n.generators_t.q_set.loc[
        SNAPSHOT,
        GENERATOR_NAME
    ] = Q_SUPPORT

    print()
    print(
        f"Reactive support applied through generator: "
        f"{GENERATOR_NAME}"
    )

    print(
        f"Q setpoint: {old_q:.3f} -> "
        f"{Q_SUPPORT:.3f} MVAr"
    )

    # ----------------------------------------------------------------------------------------------
    # Run AC nonlinear power flow
    # ----------------------------------------------------------------------------------------------

    try:

        n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
        )

        converged = True

    except Exception as exc:

        converged = False

        print()
        print("POWER FLOW FAILED")
        print("-" * 110)
        print(str(exc))

        results.append(
            {
                "target_multiplier": multiplier,
                "q_support_mvar": Q_SUPPORT,
                "converged": False,
                "min_voltage_pu": float("nan"),
                "weak_bus_voltage_pu": float("nan"),
                "max_line_loading_pct": float("nan"),
                "overloaded_lines": float("nan"),
                "max_loaded_line": "",
                "target_line_loading_pct": float("nan"),
                "max_transformer_loading_pct": float("nan"),
                "weak_voltage_ok": False,
                "minimum_voltage_ok": False,
                "thermal_loading_ok": False,
                "overload_count_ok": False,
                "fully_acceptable": False,
            }
        )

        continue

    # ----------------------------------------------------------------------------------------------
    # Voltage results
    # ----------------------------------------------------------------------------------------------

    voltage_snapshot = n.buses_t.v_mag_pu.loc[SNAPSHOT]

    min_voltage = float(
        voltage_snapshot.min()
    )

    weak_bus_voltage = float(
        voltage_snapshot.loc[WEAK_BUS]
    )

    # ----------------------------------------------------------------------------------------------
    # Line loading
    #
    # PyPSA AC PF:
    # p0 / s_nom gives loading approximately in percent.
    # Use apparent power where available.
    # ----------------------------------------------------------------------------------------------

    if (
        hasattr(n, "lines_t")
        and hasattr(n.lines_t, "p0")
        and hasattr(n.lines_t, "q0")
    ):

        p0 = n.lines_t.p0.loc[SNAPSHOT]

        q0 = n.lines_t.q0.loc[SNAPSHOT]

        apparent_power = (
            p0.pow(2) + q0.pow(2)
        ).pow(0.5)

        line_loading = (
            apparent_power
            / n.lines["s_nom"]
            * 100.0
        )

    else:

        # Fallback to active-power loading if q0 is unavailable.
        line_loading = (
            n.lines_t.p0.loc[SNAPSHOT]
            .abs()
            / n.lines["s_nom"]
            * 100.0
        )

    line_loading = line_loading.replace(
        [float("inf"), -float("inf")],
        pd.NA,
    ).dropna()

    max_line_loading = float(
        line_loading.max()
    )

    max_loaded_line = str(
        line_loading.idxmax()
    )

    overloaded_mask = (
        line_loading > 100.0
    )

    overloaded_lines = int(
        overloaded_mask.sum()
    )

    # ----------------------------------------------------------------------------------------------
    # Target line loading
    # ----------------------------------------------------------------------------------------------

    target_line_loading = float(
        line_loading.loc[TARGET_LINE]
    )

    # ----------------------------------------------------------------------------------------------
    # Transformer loading
    # ----------------------------------------------------------------------------------------------

    max_transformer_loading = 0.0

    if len(n.transformers) > 0:

        if (
            hasattr(n, "transformers_t")
            and hasattr(n.transformers_t, "p0")
            and hasattr(n.transformers_t, "q0")
        ):

            transformer_p0 = (
                n.transformers_t.p0.loc[SNAPSHOT]
            )

            transformer_q0 = (
                n.transformers_t.q0.loc[SNAPSHOT]
            )

            transformer_s = (
                transformer_p0.pow(2)
                + transformer_q0.pow(2)
            ).pow(0.5)

            transformer_loading = (
                transformer_s
                / n.transformers["s_nom"]
                * 100.0
            )

            transformer_loading = (
                transformer_loading
                .replace(
                    [float("inf"), -float("inf")],
                    pd.NA,
                )
                .dropna()
            )

            if len(transformer_loading) > 0:

                max_transformer_loading = float(
                    transformer_loading.max()
                )

        elif hasattr(n.transformers_t, "p0"):

            transformer_loading = (
                n.transformers_t.p0.loc[SNAPSHOT]
                .abs()
                / n.transformers["s_nom"]
                * 100.0
            )

            transformer_loading = (
                transformer_loading
                .replace(
                    [float("inf"), -float("inf")],
                    pd.NA,
                )
                .dropna()
            )

            if len(transformer_loading) > 0:

                max_transformer_loading = float(
                    transformer_loading.max()
                )

    # ----------------------------------------------------------------------------------------------
    # Acceptance criteria
    # ----------------------------------------------------------------------------------------------

    weak_voltage_ok = (
        weak_bus_voltage >= 1.00
    )

    minimum_voltage_ok = (
        min_voltage >= 0.95
    )

    thermal_loading_ok = (
        max_line_loading <= 100.0
    )

    overload_count_ok = (
        overloaded_lines == 0
    )

    fully_acceptable = (
        converged
        and weak_voltage_ok
        and minimum_voltage_ok
        and thermal_loading_ok
        and overload_count_ok
    )

    # ----------------------------------------------------------------------------------------------
    # RESULT OUTPUT
    # ----------------------------------------------------------------------------------------------

    print()
    print("RESULT")
    print("-" * 100)

    print(
        f"Converged                 : {converged}"
    )

    print(
        f"Min V magnitude           : "
        f"{min_voltage:.6f} pu"
    )

    print(
        f"Weak bus voltage          : "
        f"{weak_bus_voltage:.6f} pu"
    )

    print(
        f"Max line loading          : "
        f"{max_line_loading:.6f} %"
    )

    print(
        f"Overloaded lines          : "
        f"{overloaded_lines}"
    )

    print(
        f"Critical/max loaded line  : "
        f"{max_loaded_line}"
    )

    print(
        f"Target line loading       : "
        f"{target_line_loading:.6f} %"
    )

    print(
        f"Max transformer loading   : "
        f"{max_transformer_loading:.6f} %"
    )

    print(
        f"Fully acceptable          : "
        f"{fully_acceptable}"
    )

    # ----------------------------------------------------------------------------------------------
    # Save result
    # ----------------------------------------------------------------------------------------------

    results.append(
        {
            "target_multiplier": multiplier,
            "q_support_mvar": Q_SUPPORT,
            "converged": converged,
            "min_voltage_pu": min_voltage,
            "weak_bus_voltage_pu": weak_bus_voltage,
            "max_line_loading_pct": max_line_loading,
            "overloaded_lines": overloaded_lines,
            "max_loaded_line": max_loaded_line,
            "target_line_loading_pct": target_line_loading,
            "max_transformer_loading_pct": max_transformer_loading,
            "weak_voltage_ok": weak_voltage_ok,
            "minimum_voltage_ok": minimum_voltage_ok,
            "thermal_loading_ok": thermal_loading_ok,
            "overload_count_ok": overload_count_ok,
            "fully_acceptable": fully_acceptable,
        }
    )


# ==================================================================================================
# RESULTS DATAFRAME
# ==================================================================================================

results_df = pd.DataFrame(results)


# ==================================================================================================
# SUMMARY
# ==================================================================================================

print()
print("=" * 110)
print("S3.14 SUMMARY")
print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# ==================================================================================================
# FULLY ACCEPTABLE CASES
# ==================================================================================================

print()
print("-" * 110)
print("FULLY ACCEPTABLE CASES")
print("-" * 110)

acceptable = results_df[
    results_df["fully_acceptable"] == True
]

if len(acceptable) == 0:

    print(
        "NO CASE SATISFIES ALL FIVE ACCEPTANCE CRITERIA."
    )

else:

    print(
        acceptable.to_string(
            index=False
        )
    )


# ==================================================================================================
# BEST THERMAL CASE
# ==================================================================================================

print()
print("-" * 110)
print("BEST THERMAL CASE")
print("-" * 110)

if len(results_df) > 0:

    best_thermal = results_df.loc[
        results_df["max_line_loading_pct"].idxmin()
    ]

    print(
        f"Target reinforcement : "
        f"{best_thermal['target_multiplier']:.2f}x"
    )

    print(
        f"Max loading          : "
        f"{best_thermal['max_line_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded lines     : "
        f"{int(best_thermal['overloaded_lines'])}"
    )

    print(
        f"Critical line        : "
        f"{best_thermal['max_loaded_line']}"
    )

    print(
        f"Target line loading  : "
        f"{best_thermal['target_line_loading_pct']:.6f} %"
    )


# ==================================================================================================
# BEST TARGET LINE CASE
# ==================================================================================================

print()
print("-" * 110)
print("BEST TARGET-LINE CASE")
print("-" * 110)

if len(results_df) > 0:

    best_target = results_df.loc[
        results_df["target_line_loading_pct"].idxmin()
    ]

    print(
        f"Target reinforcement : "
        f"{best_target['target_multiplier']:.2f}x"
    )

    print(
        f"Target line loading  : "
        f"{best_target['target_line_loading_pct']:.6f} %"
    )

    print(
        f"System max loading   : "
        f"{best_target['max_line_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded lines     : "
        f"{int(best_target['overloaded_lines'])}"
    )


# ==================================================================================================
# SAVE CSV
# ==================================================================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ==================================================================================================
# COMPLETE
# ==================================================================================================

print()
print("=" * 110)
print("S3.14 COMPLETE")
print("=" * 110)

print(
    "Results saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 110)