import pypsa
import pandas as pd
from pathlib import Path

# ==================================================================================================
# S3.13 — REMAINING THERMAL BOTTLENECK / OVERLOAD RANKING DIAGNOSIS
# ==================================================================================================

NETWORK_PATH = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"
LAMBDA = 0.953125
WEAK_BUS = "way/104388595-220"

# Best configuration established through S3.9–S3.12
REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.50,
    "merged_way/61295764-220+1": 1.50,
    "way/343436171-220": 1.50,
    "merged_way/257889771-220+1": 1.25,
}

Q_SUPPORT_MVAR = 500.0

OUTPUT = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed"
    r"\s3_13_remaining_thermal_bottleneck_diagnosis.csv"
)

print("=" * 110)
print("S3.13 — REMAINING THERMAL BOTTLENECK / OVERLOAD RANKING DIAGNOSIS")
print("=" * 110)

print(f"Network       : {NETWORK_PATH}")
print(f"Snapshot      : {SNAPSHOT}")
print(f"Lambda        : {LAMBDA}")
print(f"Weak bus      : {WEAK_BUS}")
print(f"Q support     : +{Q_SUPPORT_MVAR:.0f} MVAr")
print()

print("FIXED REINFORCEMENTS")
print("-" * 110)

for line, multiplier in REINFORCEMENTS.items():
    print(f"{line:<45} : {multiplier:.2f}x")

print()
print("=" * 110)
print("LOADING NETWORK")
print("=" * 110)

n = pypsa.Network(NETWORK_PATH)

# ----------------------------------------------------------------------------------
# Apply the established reinforcement configuration
# ----------------------------------------------------------------------------------

for line_id, multiplier in REINFORCEMENTS.items():

    if line_id not in n.lines.index:
        raise KeyError(f"Line not found in network: {line_id}")

    original = float(n.lines.at[line_id, "s_nom"])
    new_rating = original * multiplier

    n.lines.at[line_id, "s_nom"] = new_rating

    print(
        f"{line_id:<45} "
        f"{original:10.6f} -> {new_rating:10.6f} MW"
    )

# ----------------------------------------------------------------------------------
# Add local reactive support at the weak bus
# ----------------------------------------------------------------------------------
#
# IMPORTANT:
# This follows the same local-Q support methodology used in S3.9–S3.12.
# If your previous scripts used a particular component naming convention,
# this creates a fresh controllable shunt at the weak bus.
# ----------------------------------------------------------------------------------

q_name = "S3_13_local_Q_support"

if q_name in n.shunt_impedances.index:
    n.shunt_impedances.drop(q_name, inplace=True)

# PyPSA shunt sign convention:
# q > 0 represents inductive consumption in some formulations.
# To avoid silently changing the established methodology, we instead use
# a Generator with q_set / reactive capability if supported by the network.
#
# We inspect existing generators first.

weak_generators = n.generators.index[
    n.generators.bus == WEAK_BUS
].tolist()

if weak_generators:
    q_gen = weak_generators[0]

    original_q_set = (
        float(n.generators.at[q_gen, "q_set"])
        if "q_set" in n.generators.columns
        else 0.0
    )

    if "q_set" in n.generators.columns:
        n.generators.at[q_gen, "q_set"] = (
            original_q_set + Q_SUPPORT_MVAR
        )

        print()
        print(
            f"Reactive support applied through generator: {q_gen}"
        )
        print(
            f"Q setpoint: {original_q_set:.3f} -> "
            f"{original_q_set + Q_SUPPORT_MVAR:.3f} MVAr"
        )
    else:
        raise RuntimeError(
            "Generator exists at weak bus but q_set is unavailable."
        )

else:
    raise RuntimeError(
        f"No generator found at weak bus {WEAK_BUS}. "
        "Use the exact Q-support implementation from S3.12."
    )

# ----------------------------------------------------------------------------------
# Run AC nonlinear power flow
# ----------------------------------------------------------------------------------

print()
print("=" * 110)
print("RUNNING AC NONLINEAR POWER FLOW")
print("=" * 110)

try:
    n.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-8,
        use_seed=True,
    )

    converged = True

except Exception as e:
    print()
    print("POWER FLOW FAILED")
    print(str(e))

    converged = False

# ----------------------------------------------------------------------------------
# Stop if PF failed
# ----------------------------------------------------------------------------------

if not converged:

    result = pd.DataFrame([{
        "snapshot": SNAPSHOT,
        "converged": False,
    }])

    result.to_csv(OUTPUT, index=False)

    print()
    print(f"Results saved to: {OUTPUT}")
    print("=" * 110)

else:

    # ==============================================================================================
    # BUS VOLTAGES
    # ==============================================================================================

    v_mag = n.buses_t.v_mag_pu.loc[SNAPSHOT]

    min_voltage = float(v_mag.min())

    if WEAK_BUS in v_mag.index:
        weak_voltage = float(v_mag.loc[WEAK_BUS])
    else:
        weak_voltage = float("nan")

    # ==============================================================================================
    # LINE LOADINGS
    # ==============================================================================================

    loading = (
        n.lines_t.p0.loc[SNAPSHOT].abs()
        / n.lines.s_nom.replace(0, float("nan"))
        * 100.0
    )

    loading = loading.dropna().sort_values(ascending=False)

    # ==============================================================================================
    # OVERLOADS
    # ==============================================================================================

    overloaded = loading[loading > 100.0]

    max_loading = float(loading.max())
    overload_count = int(len(overloaded))

    # ==============================================================================================
    # TRANSFORMER LOADINGS
    # ==============================================================================================

    transformer_loading = pd.Series(dtype=float)

    if len(n.transformers) > 0:

        transformer_loading = (
            n.transformers_t.p0.loc[SNAPSHOT].abs()
            / n.transformers.s_nom.replace(0, float("nan"))
            * 100.0
        )

        transformer_loading = transformer_loading.dropna()

    if len(transformer_loading) > 0:
        max_transformer_loading = float(transformer_loading.max())
        max_transformer = transformer_loading.idxmax()
    else:
        max_transformer_loading = float("nan")
        max_transformer = None

    # ==============================================================================================
    # OUTPUT
    # ==============================================================================================

    print()
    print("=" * 110)
    print("SYSTEM SUMMARY")
    print("=" * 110)

    print(f"Converged               : {converged}")
    print(f"Minimum V magnitude     : {min_voltage:.6f} pu")
    print(f"Weak bus voltage        : {weak_voltage:.6f} pu")
    print(f"Max line loading        : {max_loading:.6f} %")
    print(f"Overloaded lines        : {overload_count}")
    print(
        f"Max transformer loading : "
        f"{max_transformer_loading:.6f} %"
    )

    # ==============================================================================================
    # FULL OVERLOAD RANKING
    # ==============================================================================================

    print()
    print("=" * 110)
    print("REMAINING OVERLOADED LINES — RANKED")
    print("=" * 110)

    if overload_count == 0:

        print("NO OVERLOADED LINES.")

    else:

        rows = []

        for rank, (line_id, load_pct) in enumerate(
            overloaded.items(), start=1
        ):

            line = n.lines.loc[line_id]

            rows.append({
                "rank": rank,
                "line": line_id,
                "loading_pct": float(load_pct),
                "s_nom_mw": float(line.s_nom),
                "bus0": line.bus0,
                "bus1": line.bus1,
                "length_km": (
                    float(line.length)
                    if "length" in n.lines.columns
                    and pd.notna(line.length)
                    else float("nan")
                ),
                "r": float(line.r),
                "x": float(line.x),
            })

            print(
                f"{rank:>2}. "
                f"{line_id:<45} "
                f"{load_pct:>10.3f}%   "
                f"s_nom={line.s_nom:>10.3f} MW   "
                f"{line.bus0} -> {line.bus1}"
            )

        diagnosis = pd.DataFrame(rows)

        # ------------------------------------------------------------------------------------------
        # Save diagnosis
        # ------------------------------------------------------------------------------------------

        diagnosis.to_csv(OUTPUT, index=False)

    # ==============================================================================================
    # TOP 10 LINES — INCLUDING NON-OVERLOADED
    # ==============================================================================================

    print()
    print("=" * 110)
    print("TOP 10 MOST LOADED LINES")
    print("=" * 110)

    top10 = loading.head(10)

    for rank, (line_id, load_pct) in enumerate(
        top10.items(), start=1
    ):

        line = n.lines.loc[line_id]

        print(
            f"{rank:>2}. "
            f"{line_id:<45} "
            f"{load_pct:>10.3f}%   "
            f"s_nom={line.s_nom:>10.3f} MW"
        )

    # ==============================================================================================
    # CRITICAL BOTTLENECK
    # ==============================================================================================

    print()
    print("=" * 110)
    print("CURRENT THERMAL BOTTLENECK")
    print("=" * 110)

    if len(loading) > 0:

        critical_line = loading.index[0]

        print(f"Line    : {critical_line}")
        print(f"Loading : {loading.iloc[0]:.6f} %")
        print(
            f"s_nom   : "
            f"{n.lines.at[critical_line, 's_nom']:.6f} MW"
        )
        print(
            f"From    : "
            f"{n.lines.at[critical_line, 'bus0']}"
        )
        print(
            f"To      : "
            f"{n.lines.at[critical_line, 'bus1']}"
        )

    # ==============================================================================================
    # BASIC ACCEPTANCE FLAGS
    # ==============================================================================================

    weak_voltage_ok = weak_voltage >= 1.00
    minimum_voltage_ok = min_voltage >= 0.95
    thermal_loading_ok = max_loading <= 100.0
    overload_count_ok = overload_count == 0

    fully_acceptable = (
        converged
        and weak_voltage_ok
        and minimum_voltage_ok
        and thermal_loading_ok
        and overload_count_ok
    )

    print()
    print("=" * 110)
    print("ACCEPTANCE CHECK")
    print("=" * 110)

    print(
        f"Weak voltage >= 1.00 pu : "
        f"{weak_voltage_ok}"
    )

    print(
        f"Minimum voltage >= 0.95 pu : "
        f"{minimum_voltage_ok}"
    )

    print(
        f"Maximum loading <= 100% : "
        f"{thermal_loading_ok}"
    )

    print(
        f"Zero overloads : "
        f"{overload_count_ok}"
    )

    print(
        f"FULLY ACCEPTABLE : "
        f"{fully_acceptable}"
    )

    # ==============================================================================================
    # SAVE SUMMARY
    # ==============================================================================================

    summary = pd.DataFrame([{
        "snapshot": SNAPSHOT,
        "converged": converged,
        "q_support_mvar": Q_SUPPORT_MVAR,
        "min_voltage_pu": min_voltage,
        "weak_bus_voltage_pu": weak_voltage,
        "max_line_loading_pct": max_loading,
        "overloaded_lines": overload_count,
        "max_transformer_loading_pct": max_transformer_loading,
        "max_loaded_transformer": max_transformer,
        "critical_line": (
            loading.index[0]
            if len(loading) > 0
            else None
        ),
        "weak_voltage_ok": weak_voltage_ok,
        "minimum_voltage_ok": minimum_voltage_ok,
        "thermal_loading_ok": thermal_loading_ok,
        "overload_count_ok": overload_count_ok,
        "fully_acceptable": fully_acceptable,
    }])

    summary.to_csv(OUTPUT, index=False)

    print()
    print("=" * 110)
    print("S3.13 COMPLETE")
    print("=" * 110)
    print(f"Results saved to:")
    print(OUTPUT)
    print("=" * 110)