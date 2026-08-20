import copy
import numpy as np
import pandas as pd
import pypsa

NETWORK = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

# Fine continuation range
SCALES = [
    0.75,
    0.80,
    0.825,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95,
    0.975,
    1.00,
]

print("=" * 78)
print("S2 AC FINE CONTINUATION / CRITICAL-SCALE SEARCH")
print("=" * 78)

print(f"\nNetwork : {NETWORK}")
print(f"Snapshot: {SNAPSHOT}")

# -------------------------------------------------------------------------
# LOAD READ-ONLY SOURCE NETWORK
# -------------------------------------------------------------------------

n_source = pypsa.Network(NETWORK)

print("\nNETWORK")
print("-" * 78)
print(f"Buses        : {len(n_source.buses)}")
print(f"Lines        : {len(n_source.lines)}")
print(f"Transformers : {len(n_source.transformers)}")
print(f"Generators   : {len(n_source.generators)}")
print(f"Loads        : {len(n_source.loads)}")
print(f"Links        : {len(n_source.links)}")

# -------------------------------------------------------------------------
# ORIGINAL DISPATCH
# -------------------------------------------------------------------------

snap = SNAPSHOT

original_gen = n_source.generators_t.p_set.loc[snap].copy()
original_load = n_source.loads_t.p_set.loc[snap].copy()

total_generation = original_gen.sum()
total_load = original_load.sum()

print("\nORIGINAL BALANCE")
print("-" * 78)
print(f"Generation : {total_generation:.6f} MW")
print(f"Load       : {total_load:.6f} MW")
print(f"Mismatch   : {total_generation - total_load:.6f} MW")

# -------------------------------------------------------------------------
# BALANCE GENERATION
#
# We reproduce the previous diagnostic:
# non-wind generation is adjusted so that total generation = total load.
# -------------------------------------------------------------------------

wind_mask = n_source.generators.carrier.astype(str).str.lower().eq("wind")

wind_generation = original_gen[wind_mask].sum()

other_generation = original_gen[~wind_mask].sum()

required_other_generation = total_load - wind_generation

print("\nBALANCE CORRECTION")
print("-" * 78)
print(f"Wind generation          : {wind_generation:.6f} MW")
print(f"Other generation         : {other_generation:.6f} MW")
print(f"Required other generation: {required_other_generation:.6f} MW")

# -------------------------------------------------------------------------
# BUILD BALANCED SOURCE NETWORK
# -------------------------------------------------------------------------

n_balanced = copy.deepcopy(n_source)

# Scale all non-wind generation proportionally.
if other_generation <= 0:
    raise RuntimeError("No non-wind generation available for balancing.")

balance_factor = required_other_generation / other_generation

for g in n_balanced.generators.index:

    if not wind_mask.loc[g]:

        old_value = n_balanced.generators_t.p_set.at[snap, g]

        n_balanced.generators_t.p_set.at[snap, g] = (
            old_value * balance_factor
        )

corrected_generation = n_balanced.generators_t.p_set.loc[snap].sum()

print(f"\nBalanced generation : {corrected_generation:.6f} MW")
print(f"Balanced load       : {total_load:.6f} MW")
print(
    f"Balanced mismatch   : "
    f"{corrected_generation - total_load:.12f} MW"
)

# -------------------------------------------------------------------------
# HELPER
# -------------------------------------------------------------------------

def get_voltage_stats(n):

    v = n.buses_t.v_mag_pu.loc[snap]

    finite_v = v[np.isfinite(v)]

    if len(finite_v) == 0:
        return np.nan, np.nan

    return finite_v.min(), finite_v.max()


def get_line_loading(n):

    if len(n.lines) == 0:
        return np.nan, pd.DataFrame()

    p0 = n.lines_t.p0.loc[snap].abs()
    p1 = n.lines_t.p1.loc[snap].abs()

    flow = pd.concat([p0, p1], axis=1).max(axis=1)

    loading = 100.0 * flow / n.lines.s_nom

    idx = loading.idxmax()

    return (
        float(loading.loc[idx]),
        pd.DataFrame(
            {
                "line": loading.index,
                "loading_pct": loading.values,
                "flow_MW": flow.values,
            }
        ).sort_values("loading_pct", ascending=False),
    )


# -------------------------------------------------------------------------
# CONTINUATION
# -------------------------------------------------------------------------

results = []

# Previous converged voltage state.
previous_v_mag = None
previous_v_ang = None

for scale in SCALES:

    print("\n" + "-" * 78)
    print(f"RUNNING AC TEST — SCALE {scale:.3f}")
    print("-" * 78)

    n = copy.deepcopy(n_balanced)

    # -------------------------------------------------------------
    # Scale generation and load simultaneously.
    # This preserves the zero-net-balance condition at every scale.
    # -------------------------------------------------------------

    n.generators_t.p_set.loc[snap] = (
        n_balanced.generators_t.p_set.loc[snap] * scale
    )

    n.loads_t.p_set.loc[snap] = (
        n_balanced.loads_t.p_set.loc[snap] * scale
    )

    generation = n.generators_t.p_set.loc[snap].sum()
    load = n.loads_t.p_set.loc[snap].sum()

    print(f"Generation : {generation:.6f} MW")
    print(f"Load       : {load:.6f} MW")
    print(f"Mismatch   : {generation - load:.12f} MW")

    # -------------------------------------------------------------
    # WARM START
    #
    # Use previous converged voltage solution.
    # -------------------------------------------------------------

    if previous_v_mag is not None:

        common_buses = n.buses.index.intersection(previous_v_mag.index)

        n.buses_t.v_mag_pu.loc[snap, common_buses] = (
            previous_v_mag.loc[common_buses]
        )

        n.buses_t.v_ang.loc[snap, common_buses] = (
            previous_v_ang.loc[common_buses]
        )

        print("Initialization: PREVIOUS CONVERGED SOLUTION")

    else:

        print("Initialization: NETWORK INITIAL STATE")

    # -------------------------------------------------------------
    # AC POWER FLOW
    # -------------------------------------------------------------

    try:

        pf = n.pf(
            snapshots=[snap],
            x_tol=1e-8,
            use_seed=True,
        )

        converged = bool(pf["converged"].loc[snap].all())

        iterations = int(pf["n_iter"].loc[snap].max())

        error = float(pf["error"].loc[snap].max())

        exception = ""

    except Exception as e:

        print("\nPF EXCEPTION")
        print(type(e).__name__, str(e))

        converged = False
        iterations = np.nan
        error = np.nan
        exception = f"{type(e).__name__}: {e}"

    # -------------------------------------------------------------
    # ONLY TRUST PHYSICAL STATE IF CONVERGED
    # -------------------------------------------------------------

    if converged:

        min_v, max_v = get_voltage_stats(n)

        max_loading, loading_table = get_line_loading(n)

        max_flow = (
            loading_table.iloc[0]["flow_MW"]
            if len(loading_table)
            else np.nan
        )

        print("\nRESULT")
        print("-" * 78)
        print("Converged       : TRUE")
        print(f"Iterations      : {iterations}")
        print(f"Final error     : {error:.6e}")
        print(f"Min V magnitude : {min_v:.6f} pu")
        print(f"Max V magnitude : {max_v:.6f} pu")
        print(f"Max line flow   : {max_flow:.6f} MW")
        print(f"Max line loading: {max_loading:.6f} %")

        # Save this solution for the next continuation step.
        previous_v_mag = n.buses_t.v_mag_pu.loc[snap].copy()
        previous_v_ang = n.buses_t.v_ang.loc[snap].copy()

    else:

        min_v = np.nan
        max_v = np.nan
        max_loading = np.nan
        max_flow = np.nan

        print("\nRESULT")
        print("-" * 78)
        print("Converged       : FALSE")
        print(f"Iterations      : {iterations}")
        print(f"Final error     : {error}")
        print("Voltage state   : INVALID / DIVERGED")

        # IMPORTANT:
        # Do NOT use a failed solution as the next warm start.
        # Keep the previous converged solution.

    results.append(
        {
            "scale": scale,
            "converged": converged,
            "iterations": iterations,
            "error": error,
            "min_v_pu": min_v,
            "max_v_pu": max_v,
            "max_line_flow_mw": max_flow,
            "max_loading_pct": max_loading,
            "exception": exception,
        }
    )


# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 78)
print("FINAL FINE CONTINUATION SUMMARY")
print("=" * 78)

print(results_df.to_string(index=False))

# -------------------------------------------------------------------------
# CRITICAL SCALE
# -------------------------------------------------------------------------

successful = results_df[results_df["converged"] == True]
failed = results_df[results_df["converged"] == False]

print("\n")
print("=" * 78)
print("CRITICAL-SCALE SUMMARY")
print("=" * 78)

if len(successful):

    highest_pass = successful["scale"].max()

    print(f"Highest converged scale : {highest_pass:.3f}")

else:

    highest_pass = np.nan
    print("Highest converged scale : NONE")

if len(failed):

    first_fail = failed["scale"].min()

    print(f"First failed scale      : {first_fail:.3f}")

else:

    first_fail = np.nan
    print("First failed scale      : NONE")

if np.isfinite(highest_pass) and np.isfinite(first_fail):

    print(
        f"\nCritical region bracket: "
        f"{highest_pass:.3f} < λcrit < {first_fail:.3f}"
    )

print("\n")
print("=" * 78)
print("NO NETWORK PARAMETERS WERE MODIFIED.")
print("NO REINFORCEMENT WAS APPLIED.")
print("NO OUTPUT NETWORK WAS SAVED.")
print("=" * 78)