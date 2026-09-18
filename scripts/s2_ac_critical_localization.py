import copy
import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# S2.5 AC CRITICAL-SCALE LOCALIZATION
# =============================================================================

NETWORK_PATH = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

# Fine continuation interval from the previous diagnostic
SCALES = [
    0.9525,
    0.9550,
    0.9575,
    0.9600,
    0.9625,
    0.9650,
    0.9675,
    0.9700,
    0.9725,
    0.9750,
]


# =============================================================================
# LOAD NETWORK
# =============================================================================

print("=" * 78)
print("S2.5 AC CRITICAL-SCALE LOCALIZATION")
print("=" * 78)

print()
print(f"Network : {NETWORK_PATH}")
print(f"Snapshot: {SNAPSHOT}")

n_base = pypsa.Network(NETWORK_PATH)

print()
print("NETWORK")
print("-" * 78)
print(f"Buses        : {len(n_base.buses)}")
print(f"Lines        : {len(n_base.lines)}")
print(f"Transformers : {len(n_base.transformers)}")
print(f"Generators   : {len(n_base.generators)}")
print(f"Loads        : {len(n_base.loads)}")
print(f"Links        : {len(n_base.links)}")


# =============================================================================
# ORIGINAL BALANCE
# =============================================================================

snapshot = SNAPSHOT

original_generation = n_base.generators_t.p_set.loc[snapshot].sum()
total_load = n_base.loads_t.p_set.loc[snapshot].sum()

print()
print("=" * 78)
print("ORIGINAL BALANCE")
print("=" * 78)

print(f"Generation : {original_generation:.6f} MW")
print(f"Load       : {total_load:.6f} MW")
print(f"Mismatch   : {original_generation - total_load:.6f} MW")


# =============================================================================
# BALANCE CORRECTION
# =============================================================================
#
# Preserve the wind generation and interconnector generation.
# Correct the non-wind generation so total generation equals total load.
#
# This follows the successful continuation diagnostic you already ran.
# =============================================================================

wind_mask = n_base.generators.carrier.astype(str).str.lower().eq("wind")

wind_generation = (
    n_base.generators_t.p_set.loc[snapshot, wind_mask].sum()
)

other_generation = original_generation - wind_generation

required_other_generation = total_load - wind_generation

print()
print("=" * 78)
print("BALANCE CORRECTION")
print("=" * 78)

print(f"Wind generation           : {wind_generation:.6f} MW")
print(f"Other generation          : {other_generation:.6f} MW")
print(
    f"Required other generation : "
    f"{required_other_generation:.6f} MW"
)


# =============================================================================
# CREATE BALANCED BASE NETWORK
# =============================================================================

n_balanced = copy.deepcopy(n_base)

non_wind_mask = ~wind_mask

current_other_generation = (
    n_balanced.generators_t.p_set.loc[
        snapshot, non_wind_mask
    ].sum()
)

if current_other_generation <= 0:
    raise RuntimeError(
        "Non-wind generation is zero or negative; "
        "cannot apply proportional balance correction."
    )

balance_factor = (
    required_other_generation / current_other_generation
)

n_balanced.generators_t.p_set.loc[
    snapshot, non_wind_mask
] *= balance_factor


# Verify balance

balanced_generation = (
    n_balanced.generators_t.p_set.loc[snapshot].sum()
)

balanced_load = (
    n_balanced.loads_t.p_set.loc[snapshot].sum()
)

print()
print(f"Balanced generation : {balanced_generation:.12f} MW")
print(f"Balanced load       : {balanced_load:.12f} MW")
print(
    f"Balanced mismatch   : "
    f"{balanced_generation - balanced_load:.12f} MW"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def reset_to_scale(n, scale):
    """
    Scale all generators and loads proportionally from the
    balanced S2 operating point.
    """

    n.generators_t.p_set.loc[snapshot] = (
        n_balanced.generators_t.p_set.loc[snapshot] * scale
    )

    n.loads_t.p_set.loc[snapshot] = (
        n_balanced.loads_t.p_set.loc[snapshot] * scale
    )


def get_line_loading(n):
    """
    Calculate maximum active-power loading for each AC line.
    """

    if len(n.lines) == 0:
        return pd.Series(dtype=float)

    p0 = n.lines_t.p0.loc[snapshot].abs()
    p1 = n.lines_t.p1.loc[snapshot].abs()

    max_flow = pd.concat(
        [p0, p1],
        axis=1
    ).max(axis=1)

    loading = (
        max_flow / n.lines.s_nom * 100.0
    )

    return loading.sort_values(ascending=False)


def get_transformer_loading(n):
    """
    Calculate maximum active-power loading for each transformer.
    """

    if len(n.transformers) == 0:
        return pd.Series(dtype=float)

    p0 = n.transformers_t.p0.loc[snapshot].abs()
    p1 = n.transformers_t.p1.loc[snapshot].abs()

    max_flow = pd.concat(
        [p0, p1],
        axis=1
    ).max(axis=1)

    loading = (
        max_flow / n.transformers.s_nom * 100.0
    )

    return loading.sort_values(ascending=False)


def run_ac(n):
    """
    Run nonlinear AC power flow.
    """

    try:
        result = n.pf(
            snapshots=[snapshot],
            x_tol=1e-8,
            use_seed=True,
            distribute_slack=True,
        )

        converged = bool(
            result["converged"].loc[snapshot].iloc[0]
        )

        iterations = int(
            result["n_iter"].loc[snapshot].iloc[0]
        )

        error = float(
            result["error"].loc[snapshot].iloc[0]
        )

        return (
            converged,
            iterations,
            error,
            None,
        )

    except Exception as exc:

        return (
            False,
            None,
            np.nan,
            repr(exc),
        )


# =============================================================================
# CONTINUATION
# =============================================================================

print()
print("=" * 78)
print("CRITICAL-SCALE CONTINUATION")
print("=" * 78)


results = []

# Keep the previous converged solution for warm-starting.
previous_solution = None

highest_converged_network = None
highest_converged_scale = None


for scale in SCALES:

    print()
    print("-" * 78)
    print(f"RUNNING AC TEST — SCALE {scale:.4f}")
    print("-" * 78)

    n = copy.deepcopy(n_balanced)

    reset_to_scale(n, scale)

    # -------------------------------------------------------------------------
    # Warm start
    # -------------------------------------------------------------------------

    if previous_solution is not None:

        print(
            "Initialization: PREVIOUS CONVERGED SOLUTION"
        )

        previous_v = previous_solution.buses_t.v_mag_pu.loc[
            snapshot
        ]

        previous_angle = previous_solution.buses_t.v_ang.loc[
            snapshot
        ]

        common_buses = n.buses.index.intersection(
            previous_v.index
        )

        n.buses_t.v_mag_pu.loc[
            snapshot, common_buses
        ] = previous_v.loc[common_buses]

        n.buses_t.v_ang.loc[
            snapshot, common_buses
        ] = previous_angle.loc[common_buses]

    else:

        print(
            "Initialization: NETWORK INITIAL STATE"
        )

    generation = (
        n.generators_t.p_set.loc[snapshot].sum()
    )

    load = (
        n.loads_t.p_set.loc[snapshot].sum()
    )

    print(f"Generation : {generation:.6f} MW")
    print(f"Load       : {load:.6f} MW")
    print(f"Mismatch   : {generation - load:.12f} MW")

    # -------------------------------------------------------------------------
    # AC power flow
    # -------------------------------------------------------------------------

    converged, iterations, error, exception = run_ac(n)

    if converged:

        # -------------------------------------------------------------
        # Voltage
        # -------------------------------------------------------------

        voltage = n.buses_t.v_mag_pu.loc[snapshot]

        min_v = float(voltage.min())
        max_v = float(voltage.max())

        min_v_bus = voltage.idxmin()
        max_v_bus = voltage.idxmax()

        # -------------------------------------------------------------
        # Line loading
        # -------------------------------------------------------------

        line_loading = get_line_loading(n)

        if len(line_loading) > 0:

            max_line_loading = float(
                line_loading.iloc[0]
            )

            max_line = line_loading.index[0]

        else:

            max_line_loading = np.nan
            max_line = None

        # -------------------------------------------------------------
        # Transformer loading
        # -------------------------------------------------------------

        transformer_loading = (
            get_transformer_loading(n)
        )

        if len(transformer_loading) > 0:

            max_transformer_loading = float(
                transformer_loading.iloc[0]
            )

            max_transformer = (
                transformer_loading.index[0]
            )

        else:

            max_transformer_loading = np.nan
            max_transformer = None

        print()
        print("RESULT")
        print("-" * 78)

        print("Converged       : TRUE")
        print(f"Iterations      : {iterations}")
        print(f"Final error     : {error:.6e}")
        print(f"Min V magnitude : {min_v:.6f} pu")
        print(f"Min-V bus       : {min_v_bus}")
        print(f"Max V magnitude : {max_v:.6f} pu")
        print(f"Max-V bus       : {max_v_bus}")
        print(
            f"Max line loading: "
            f"{max_line_loading:.6f} %"
        )
        print(f"Max-loaded line : {max_line}")

        if max_transformer is not None:
            print(
                f"Max transformer loading: "
                f"{max_transformer_loading:.6f} %"
            )
            print(
                f"Max-loaded transformer : "
                f"{max_transformer}"
            )

        # -------------------------------------------------------------
        # Store successful solution for warm start
        # -------------------------------------------------------------

        previous_solution = copy.deepcopy(n)

        highest_converged_network = copy.deepcopy(n)
        highest_converged_scale = scale

    else:

        min_v = np.nan
        max_v = np.nan
        min_v_bus = None
        max_v_bus = None
        max_line_loading = np.nan
        max_line = None
        max_transformer_loading = np.nan
        max_transformer = None

        print()
        print("RESULT")
        print("-" * 78)

        print("Converged       : FALSE")
        print(f"Iterations      : {iterations}")
        print(f"Final error     : {error}")

        if exception is not None:
            print(f"Exception       : {exception}")

        print(
            "Voltage state   : INVALID / DIVERGED"
        )

    results.append(
        {
            "scale": scale,
            "converged": converged,
            "iterations": iterations,
            "error": error,
            "min_v_pu": min_v,
            "min_v_bus": min_v_bus,
            "max_v_pu": max_v,
            "max_v_bus": max_v_bus,
            "max_line_loading_pct": max_line_loading,
            "max_line": max_line,
            "max_transformer_loading_pct":
                max_transformer_loading,
            "max_transformer":
                max_transformer,
            "exception": exception,
        }
    )

    # -------------------------------------------------------------------------
    # Stop after first failure
    # -------------------------------------------------------------------------

    if not converged:

        print()
        print(
            "First AC failure encountered. "
            "Stopping continuation."
        )

        break


# =============================================================================
# SUMMARY
# =============================================================================

results_df = pd.DataFrame(results)

print()
print("=" * 78)
print("FINAL CRITICAL-SCALE SUMMARY")
print("=" * 78)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# CRITICAL SCALE
# =============================================================================

successful = results_df[
    results_df["converged"] == True
]

failed = results_df[
    results_df["converged"] == False
]


print()
print("=" * 78)
print("CRITICAL-SCALE BRACKET")
print("=" * 78)

if len(successful) > 0:

    highest_scale = successful["scale"].max()

    print(
        f"Highest converged scale : "
        f"{highest_scale:.4f}"
    )

else:

    highest_scale = None
    print("No converged scale found.")


if len(failed) > 0:

    first_failed = failed["scale"].iloc[0]

    print(
        f"First failed scale      : "
        f"{first_failed:.4f}"
    )

    if highest_scale is not None:

        print(
            f"Critical region         : "
            f"{highest_scale:.4f} < "
            f"lambda_crit < "
            f"{first_failed:.4f}"
        )

else:

    print(
        "No failure encountered "
        "within tested range."
    )


# =============================================================================
# CRITICAL BUS LOCALIZATION
# =============================================================================

if highest_converged_network is not None:

    n = highest_converged_network

    print()
    print("=" * 78)
    print(
        f"CRITICAL BUS LOCALIZATION "
        f"AT λ = {highest_converged_scale:.4f}"
    )
    print("=" * 78)

    voltage = (
        n.buses_t.v_mag_pu.loc[snapshot]
        .sort_values()
    )

    print()
    print("TOP 10 LOWEST-VOLTAGE BUSES")
    print("-" * 78)

    for bus, value in voltage.head(10).items():

        print(
            f"{bus:<55} "
            f"{value:.6f} pu"
        )


    # -------------------------------------------------------------------------
    # Bus voltage deviation from 1 pu
    # -------------------------------------------------------------------------

    deviation = (
        (voltage - 1.0).abs()
        .sort_values(ascending=False)
    )

    print()
    print("TOP 10 LARGEST VOLTAGE DEVIATIONS")
    print("-" * 78)

    for bus, value in deviation.head(10).items():

        actual_v = voltage.loc[bus]

        print(
            f"{bus:<55} "
            f"V={actual_v:.6f} pu  "
            f"|ΔV|={value:.6f}"
        )


    # =========================================================================
    # CRITICAL LINE LOCALIZATION
    # =========================================================================

    line_loading = get_line_loading(n)

    print()
    print("=" * 78)
    print("TOP 10 MOST-LOADED AC LINES")
    print("=" * 78)

    if len(line_loading) > 0:

        for line, loading in line_loading.head(10).items():

            p0 = abs(
                n.lines_t.p0.loc[snapshot, line]
            )

            p1 = abs(
                n.lines_t.p1.loc[snapshot, line]
            )

            s_nom = n.lines.loc[line, "s_nom"]

            print(
                f"{line:<45} "
                f"loading={loading:8.3f}%  "
                f"flow={max(p0, p1):9.3f} MW  "
                f"s_nom={s_nom:9.3f} MW"
            )


    # =========================================================================
    # CRITICAL TRANSFORMER LOCALIZATION
    # =========================================================================

    transformer_loading = (
        get_transformer_loading(n)
    )

    print()
    print("=" * 78)
    print("TRANSFORMER LOADING")
    print("=" * 78)

    if len(transformer_loading) > 0:

        for transformer, loading in (
            transformer_loading.head(10).items()
        ):

            p0 = abs(
                n.transformers_t.p0.loc[
                    snapshot, transformer
                ]
            )

            p1 = abs(
                n.transformers_t.p1.loc[
                    snapshot, transformer
                ]
            )

            s_nom = n.transformers.loc[
                transformer, "s_nom"
            ]

            print(
                f"{transformer:<45} "
                f"loading={loading:8.3f}%  "
                f"flow={max(p0, p1):9.3f} MW  "
                f"s_nom={s_nom:9.3f} MW"
            )

    else:

        print("No transformers found.")


# =============================================================================
# FINAL SAFETY CHECK
# =============================================================================

print()
print("=" * 78)
print("NETWORK INTEGRITY")
print("=" * 78)

print("Network parameters modified : NO")
print("Reinforcement applied       : NO")
print("Network saved               : NO")
print("Original input preserved    : YES")

print()
print("=" * 78)
print("S2.5 DIAGNOSTIC COMPLETE")
print("=" * 78)