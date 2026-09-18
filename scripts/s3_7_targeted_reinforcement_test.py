from pathlib import Path
import numpy as np
import pandas as pd
import pypsa

# ============================================================
# S3.7 — TARGETED REINFORCEMENT IMPACT TEST
# ============================================================

NETWORK = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed\eirgrid_second_reinforced_network.nc"
)

OUTPUT = Path(
    r"C:\Users\Dell\ireland-wind-grid-optimiser\data\processed"
    r"s3_7_targeted_reinforcement_results.csv"
)

SNAPSHOT = "S2_PEAK_DEMAND"
LAMBDA = 0.953125

# Top congestion candidates identified in S3.6
CANDIDATES = [
    "merged_way/61295764-220+1",
    "way/343436171-220",
    "merged_way/257889771-220+1",
    "merged_way/1231251986-220+2",
    "merged_relation/4872159-220+1",
]

MULTIPLIERS = [1.25, 1.50]

WEAK_BUS = "way/104388595-220"

# ============================================================
# LOAD NETWORK
# ============================================================

print("=" * 110)
print("S3.7 — TARGETED REINFORCEMENT IMPACT TEST")
print("=" * 110)

print(f"Network  : {NETWORK}")
print(f"Snapshot : {SNAPSHOT}")
print(f"Lambda   : {LAMBDA}")
print(f"Weak bus : {WEAK_BUS}")

base = pypsa.Network(NETWORK)

# ============================================================
# SCALE OPERATING POINT
# ============================================================

base.loads_t.p_set.loc[SNAPSHOT] *= LAMBDA
base.generators_t.p_set.loc[SNAPSHOT] *= LAMBDA

# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 110)
print("BASELINE AT CRITICAL OPERATING POINT")
print("=" * 110)

try:
    base.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-8,
        use_seed=True,
    )
except Exception as e:
    print("Baseline PF exception:", e)

v = base.buses_t.v_mag_pu.loc[SNAPSHOT]

if not np.isfinite(v).all() or not (v.abs() < 2.0).all():
    raise RuntimeError(
        "Baseline critical-point solution is invalid."
    )

baseline_min_v = float(v.min())
baseline_weak_v = float(v.loc[WEAK_BUS])

line_loading = (
    np.sqrt(
        base.lines_t.p0.loc[SNAPSHOT] ** 2
        + base.lines_t.q0.loc[SNAPSHOT] ** 2
    )
    / base.lines.s_nom
    * 100
)

baseline_max_loading = float(line_loading.max())
baseline_overloaded = int((line_loading > 100).sum())

print(f"Minimum voltage : {baseline_min_v:.6f} pu")
print(f"Weak bus voltage : {baseline_weak_v:.6f} pu")
print(f"Max line load    : {baseline_max_loading:.6f} %")
print(f"Overloaded lines : {baseline_overloaded}")

# ============================================================
# TEST EACH CANDIDATE
# ============================================================

results = []

for candidate in CANDIDATES:

    if candidate not in base.lines.index:
        print(f"\nWARNING: {candidate} not found. Skipping.")
        continue

    original_s_nom = float(base.lines.at[candidate, "s_nom"])

    for multiplier in MULTIPLIERS:

        print("\n" + "=" * 110)
        print(f"TEST — {candidate} — {multiplier:.2f}x")
        print("=" * 110)

        n = pypsa.Network(NETWORK)

        n.loads_t.p_set.loc[SNAPSHOT] *= LAMBDA
        n.generators_t.p_set.loc[SNAPSHOT] *= LAMBDA

        original = float(n.lines.at[candidate, "s_nom"])
        n.lines.at[candidate, "s_nom"] = original * multiplier

        print(f"Original s_nom : {original:.6f} MW")
        print(
            f"New s_nom      : "
            f"{n.lines.at[candidate, 's_nom']:.6f} MW"
        )

        try:
            n.pf(
                snapshots=[SNAPSHOT],
                x_tol=1e-8,
                use_seed=True,
            )

            v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

            valid = (
                np.isfinite(v).all()
                and (v.abs() < 2.0).all()
            )

            if not valid:
                print("INVALID AC STATE")
                results.append({
                    "candidate": candidate,
                    "multiplier": multiplier,
                    "converged": False,
                    "min_voltage_pu": np.nan,
                    "weak_bus_voltage_pu": np.nan,
                    "max_line_loading_pct": np.nan,
                    "overloaded_lines": np.nan,
                    "reinforced_line_loading_pct": np.nan,
                })
                continue

            line_loading = (
                np.sqrt(
                    n.lines_t.p0.loc[SNAPSHOT] ** 2
                    + n.lines_t.q0.loc[SNAPSHOT] ** 2
                )
                / n.lines.s_nom
                * 100
            )

            min_v = float(v.min())
            weak_v = float(v.loc[WEAK_BUS])
            max_loading = float(line_loading.max())
            overloaded = int((line_loading > 100).sum())
            reinforced_loading = float(
                line_loading.loc[candidate]
            )

            print("\nRESULT")
            print("-" * 110)
            print("Converged          : TRUE")
            print(f"Min V magnitude    : {min_v:.6f} pu")
            print(f"Weak bus voltage   : {weak_v:.6f} pu")
            print(f"Max line loading   : {max_loading:.6f} %")
            print(f"Overloaded lines   : {overloaded}")
            print(
                f"Reinforced loading : "
                f"{reinforced_loading:.6f} %"
            )

            results.append({
                "candidate": candidate,
                "multiplier": multiplier,
                "converged": True,
                "min_voltage_pu": min_v,
                "weak_bus_voltage_pu": weak_v,
                "max_line_loading_pct": max_loading,
                "overloaded_lines": overloaded,
                "reinforced_line_loading_pct":
                    reinforced_loading,
                "weak_voltage_change_pu":
                    weak_v - baseline_weak_v,
                "max_loading_change_pct":
                    max_loading - baseline_max_loading,
            })

        except Exception as e:

            print("POWER FLOW FAILED:")
            print(e)

            results.append({
                "candidate": candidate,
                "multiplier": multiplier,
                "converged": False,
                "min_voltage_pu": np.nan,
                "weak_bus_voltage_pu": np.nan,
                "max_line_loading_pct": np.nan,
                "overloaded_lines": np.nan,
                "reinforced_line_loading_pct": np.nan,
                "weak_voltage_change_pu": np.nan,
                "max_loading_change_pct": np.nan,
            })

# ============================================================
# SUMMARY
# ============================================================

df = pd.DataFrame(results)

print("\n" + "=" * 110)
print("S3.7 SUMMARY")
print("=" * 110)

if len(df):

    print(
        df[
            [
                "candidate",
                "multiplier",
                "converged",
                "min_voltage_pu",
                "weak_bus_voltage_pu",
                "max_line_loading_pct",
                "overloaded_lines",
                "reinforced_line_loading_pct",
            ]
        ].to_string(index=False)
    )

    valid = df[df["converged"] == True].copy()

    if len(valid):

        best_voltage = valid.loc[
            valid["weak_bus_voltage_pu"].idxmax()
        ]

        best_congestion = valid.loc[
            valid["max_line_loading_pct"].idxmin()
        ]

        print("\nBEST FOR WEAK-BUS VOLTAGE")
        print("-" * 110)
        print(
            f"{best_voltage['candidate']} "
            f"@ {best_voltage['multiplier']}x"
        )
        print(
            f"Weak bus voltage : "
            f"{best_voltage['weak_bus_voltage_pu']:.6f} pu"
        )

        print("\nBEST FOR CONGESTION")
        print("-" * 110)
        print(
            f"{best_congestion['candidate']} "
            f"@ {best_congestion['multiplier']}x"
        )
        print(
            f"Max line loading : "
            f"{best_congestion['max_line_loading_pct']:.6f} %"
        )

df.to_csv(OUTPUT, index=False)

print("\nResults saved to:")
print(OUTPUT)

print("\n" + "=" * 110)
print("S3.7 COMPLETE")
print("=" * 110)