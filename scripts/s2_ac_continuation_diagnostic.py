import copy
import numpy as np
import pandas as pd
import pypsa


NETWORK = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

print("=" * 78)
print("S2 AC CONTINUATION / SOLVABILITY DIAGNOSTIC")
print("=" * 78)
print()
print(f"Network : {NETWORK}")
print(f"Snapshot: {SNAPSHOT}")
print()


# -------------------------------------------------------------------------
# 1. LOAD NETWORK
# -------------------------------------------------------------------------

n = pypsa.Network(NETWORK)

print("NETWORK")
print("-" * 78)
print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")
print(f"Links        : {len(n.links)}")
print()


# -------------------------------------------------------------------------
# 2. CHECK ORIGINAL BALANCE
# -------------------------------------------------------------------------

print("=" * 78)
print("1. ORIGINAL POWER BALANCE")
print("=" * 78)

gen_original = n.generators_t.p_set.loc[SNAPSHOT].sum()
load_original = n.loads_t.p_set.loc[SNAPSHOT].sum()

print(f"Generation : {gen_original:.6f} MW")
print(f"Load       : {load_original:.6f} MW")
print(f"Mismatch   : {gen_original - load_original:.6f} MW")
print()


# -------------------------------------------------------------------------
# 3. CORRECT THE NON-WIND GENERATOR
# -------------------------------------------------------------------------

NON_WIND = "eirgrid_non_wind_generation"

if NON_WIND not in n.generators.index:
    raise RuntimeError(
        f"Generator '{NON_WIND}' was not found."
    )

fixed_other_generation = (
    n.generators_t.p_set.loc[
        SNAPSHOT,
        n.generators.index != NON_WIND
    ].sum()
)

target_load = load_original

required_non_wind = target_load - fixed_other_generation

print("=" * 78)
print("2. BALANCE CORRECTION")
print("=" * 78)

print(f"Other generation : {fixed_other_generation:.6f} MW")
print(f"Total load       : {target_load:.6f} MW")
print(f"Required non-wind: {required_non_wind:.6f} MW")
print()

n.generators_t.p_set.loc[SNAPSHOT, NON_WIND] = required_non_wind

gen_corrected = n.generators_t.p_set.loc[SNAPSHOT].sum()
load_corrected = n.loads_t.p_set.loc[SNAPSHOT].sum()

print(f"Corrected generation : {gen_corrected:.6f} MW")
print(f"Corrected load       : {load_corrected:.6f} MW")
print(f"Corrected mismatch   : {gen_corrected - load_corrected:.12f} MW")
print()


# -------------------------------------------------------------------------
# 4. STORE ORIGINAL INJECTIONS
# -------------------------------------------------------------------------

base_generator_p = n.generators_t.p_set.loc[SNAPSHOT].copy()
base_load_p = n.loads_t.p_set.loc[SNAPSHOT].copy()


# -------------------------------------------------------------------------
# 5. FLAT VOLTAGE INITIALIZATION
# -------------------------------------------------------------------------

def flat_start(network):
    """
    Set a controlled flat starting point for AC buses.
    """
    for bus in network.buses.index:
        network.buses_t.v_mag_pu.loc[SNAPSHOT, bus] = 1.0
        network.buses_t.v_ang.loc[SNAPSHOT, bus] = 0.0


# -------------------------------------------------------------------------
# 6. RUN ONE AC TEST
# -------------------------------------------------------------------------

def run_ac_test(scale):

    test = copy.deepcopy(n)

    # Scale active-power injections and loads.
    test.generators_t.p_set.loc[SNAPSHOT] = (
        base_generator_p * scale
    )

    test.loads_t.p_set.loc[SNAPSHOT] = (
        base_load_p * scale
    )

    # Rebalance the non-wind generator exactly at this scale.
    other_generation = (
        test.generators_t.p_set.loc[
            SNAPSHOT,
            test.generators.index != NON_WIND
        ].sum()
    )

    scaled_load = test.loads_t.p_set.loc[SNAPSHOT].sum()

    test.generators_t.p_set.loc[
        SNAPSHOT,
        NON_WIND
    ] = scaled_load - other_generation

    # Flat AC initialization.
    flat_start(test)

    print()
    print("-" * 78)
    print(f"RUNNING AC TEST — SCALE {scale:.2f}")
    print("-" * 78)

    generation = test.generators_t.p_set.loc[SNAPSHOT].sum()
    load = test.loads_t.p_set.loc[SNAPSHOT].sum()

    print(f"Generation : {generation:.6f} MW")
    print(f"Load       : {load:.6f} MW")
    print(f"Mismatch   : {generation - load:.12f} MW")

    try:
        result = test.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True
        )

        converged = bool(
            result["converged"].loc[SNAPSHOT].iloc[0]
        )

        error = float(
            result["error"].loc[SNAPSHOT].iloc[0]
        )

        n_iter = int(
            result["n_iter"].loc[SNAPSHOT].iloc[0]
        )

        v = test.buses_t.v_mag_pu.loc[SNAPSHOT]

        finite_v = v[np.isfinite(v)]

        if len(finite_v):
            min_v = float(finite_v.min())
            max_v = float(finite_v.max())
        else:
            min_v = np.nan
            max_v = np.nan

        # Line flows.
        try:
            p0 = test.lines_t.p0.loc[SNAPSHOT].abs()
            p1 = test.lines_t.p1.loc[SNAPSHOT].abs()

            max_flow = float(
                pd.concat([p0, p1]).max()
            )
        except Exception:
            max_flow = np.nan

        print()
        print(f"Converged       : {converged}")
        print(f"Iterations      : {n_iter}")
        print(f"Final error     : {error:.6e}")
        print(f"Min V magnitude : {min_v:.6f} pu")
        print(f"Max V magnitude : {max_v:.6f} pu")
        print(f"Max line flow   : {max_flow:.6f} MW")

        return {
            "scale": scale,
            "converged": converged,
            "iterations": n_iter,
            "error": error,
            "min_v_pu": min_v,
            "max_v_pu": max_v,
            "max_line_flow_mw": max_flow,
            "exception": ""
        }

    except Exception as exc:

        print()
        print("AC PF EXCEPTION")
        print(type(exc).__name__, str(exc))

        return {
            "scale": scale,
            "converged": False,
            "iterations": np.nan,
            "error": np.nan,
            "min_v_pu": np.nan,
            "max_v_pu": np.nan,
            "max_line_flow_mw": np.nan,
            "exception": f"{type(exc).__name__}: {exc}"
        }


# -------------------------------------------------------------------------
# 7. CONTINUATION TEST
# -------------------------------------------------------------------------

print("=" * 78)
print("3. AC CONTINUATION")
print("=" * 78)

scales = [
    0.25,
    0.50,
    0.75,
    1.00,
]

results = []

for scale in scales:
    results.append(
        run_ac_test(scale)
    )


# -------------------------------------------------------------------------
# 8. SUMMARY
# -------------------------------------------------------------------------

summary = pd.DataFrame(results)

print()
print("=" * 78)
print("FINAL CONTINUATION SUMMARY")
print("=" * 78)

print(summary.to_string(index=False))

print()
print("=" * 78)
print("INTERPRETATION")
print("=" * 78)

successful = summary[
    summary["converged"] == True
]

failed = summary[
    summary["converged"] == False
]

if len(successful) == 0:

    print(
        "NO AC TEST CONVERGED."
    )
    print()
    print(
        "This strongly suggests that the problem is not simply"
    )
    print(
        "the severity of the S2 operating point."
    )
    print()
    print(
        "Next investigation should focus on AC network formulation,"
    )
    print(
        "generator/load electrical modelling, transformer model,"
    )
    print(
        "or PyPSA initialization/convention."
    )

elif len(failed) == 0:

    print(
        "ALL AC CONTINUATION TESTS CONVERGED."
    )
    print()
    print(
        "The network formulation appears numerically solvable."
    )
    print(
        "The original S2 failure is therefore likely associated"
    )
    print(
        "with the full-stress operating point or initialization."
    )

else:

    highest_success = successful["scale"].max()
    first_failure = failed["scale"].min()

    print(
        f"Highest converged scale : {highest_success:.2f}"
    )
    print(
        f"First failed scale      : {first_failure:.2f}"
    )
    print()

    if first_failure >= 0.75:
        print(
            "AC convergence degrades near the full S2 operating point."
        )
        print(
            "This is consistent with severe system stress /"
        )
        print(
            "proximity to an AC solvability boundary."
        )
    else:
        print(
            "AC failure occurs well below the full S2 operating point."
        )
        print(
            "This points more strongly toward a network"
        )
        print(
            "formulation or electrical-parameter problem."
        )

print()
print("=" * 78)
print("NO NETWORK PARAMETERS WERE MODIFIED.")
print("NO REINFORCEMENT WAS APPLIED.")
print("NO OUTPUT NETWORK WAS SAVED.")
print("=" * 78)