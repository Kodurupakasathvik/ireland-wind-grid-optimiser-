from pathlib import Path

import pandas as pd
import pypsa


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

NETWORKS = {
    "OPTIMIZED": DATA / "eirgrid_optimized_network.nc",
    "SECOND_REINFORCED": DATA / "eirgrid_second_reinforced_network.nc",
}

SNAPSHOT = "S2_PEAK_DEMAND"


def audit_network(label, path):
    print()
    print("=" * 70)
    print(f"{label}")
    print(f"Network: {path.name}")
    print("=" * 70)

    n = pypsa.Network(path)

    print(f"Buses        : {len(n.buses)}")
    print(f"Lines        : {len(n.lines)}")
    print(f"Transformers : {len(n.transformers)}")
    print(f"Generators   : {len(n.generators)}")
    print(f"Loads        : {len(n.loads)}")

    if SNAPSHOT not in n.snapshots:
        raise ValueError(
            f"{SNAPSHOT} not found in {path.name}. "
            f"Available snapshots: {list(n.snapshots)}"
        )

    print(f"Snapshot     : {SNAPSHOT}")

    # ------------------------------------------------------------
    # Controlled AC power flow
    # ------------------------------------------------------------
    print()
    print("CONTROLLED AC PF")
    print("-" * 70)

    pf_result = n.pf(
        snapshots=[SNAPSHOT],
        distribute_slack=True,
    )

    converged = pf_result["converged"].loc[SNAPSHOT]
    error = pf_result["error"].loc[SNAPSHOT]
    iterations = pf_result["n_iter"].loc[SNAPSHOT]

    print("Converged by subnetwork:")
    print(converged.to_string())

    print()
    print("Maximum PF error :", float(error.max()))
    print("Maximum iterations:", int(iterations.max()))
    print("All converged    :", bool(converged.all()))

    # ------------------------------------------------------------
    # Bus voltage audit
    # ------------------------------------------------------------
    v = n.buses_t.v_mag_pu.loc[SNAPSHOT].copy()

    voltage_audit = pd.DataFrame(
        {
            "v_pu": v,
            "deviation_from_1pu": (1.0 - v).abs(),
        }
    ).sort_values("v_pu")

    print()
    print("BUS VOLTAGE AUDIT")
    print("-" * 70)

    print("Lowest 15 voltages:")
    print(voltage_audit.head(15).to_string())

    print()
    print("Minimum voltage:")
    print(f"  {v.min():.6f} pu")
    print(f"  Bus: {v.idxmin()}")

    # ------------------------------------------------------------
    # Line loading audit
    # ------------------------------------------------------------
    p0 = n.lines_t.p0.loc[SNAPSHOT].abs()
    p1 = n.lines_t.p1.loc[SNAPSHOT].abs()

    line_loading = pd.DataFrame(
        {
            "loading_p0_pct": p0 / n.lines.s_nom * 100.0,
            "loading_p1_pct": p1 / n.lines.s_nom * 100.0,
        }
    )

    line_loading["max_loading_pct"] = line_loading.max(axis=1)

    line_loading = line_loading.sort_values(
        "max_loading_pct",
        ascending=False,
    )

    print()
    print("LINE LOADING AUDIT")
    print("-" * 70)

    print("Top 15 line loadings:")
    print(line_loading.head(15).to_string())

    print()
    print("Loading summary:")
    print(f"  Maximum loading : {line_loading.max_loading_pct.max():.6f}%")
    print(f"  Worst line      : {line_loading.max_loading_pct.idxmax()}")
    print(f"  Lines >100%     : {(line_loading.max_loading_pct > 100).sum()}")
    print(f"  Lines >110%     : {(line_loading.max_loading_pct > 110).sum()}")
    print(f"  Lines >120%     : {(line_loading.max_loading_pct > 120).sum()}")

    # ------------------------------------------------------------
    # Target reinforced corridors
    # ------------------------------------------------------------
    target_lines = [
        "merged_way/1231251986-220+2",
        "merged_way/257889771-220+1",
        "way/343436171-220",
    ]

    print()
    print("REINFORCED CORRIDOR AUDIT")
    print("-" * 70)

    target = line_loading.loc[
        line_loading.index.intersection(target_lines)
    ].copy()

    target["s_nom_MVA"] = n.lines.loc[
        target.index, "s_nom"
    ]

    print(
        target[
            [
                "s_nom_MVA",
                "loading_p0_pct",
                "loading_p1_pct",
                "max_loading_pct",
            ]
        ].to_string()
    )

    return {
        "network": label,
        "min_voltage_pu": float(v.min()),
        "min_voltage_bus": v.idxmin(),
        "max_line_loading_pct": float(
            line_loading.max_loading_pct.max()
        ),
        "max_line": line_loading.max_loading_pct.idxmax(),
        "lines_over_100": int(
            (line_loading.max_loading_pct > 100).sum()
        ),
        "lines_over_110": int(
            (line_loading.max_loading_pct > 110).sum()
        ),
        "lines_over_120": int(
            (line_loading.max_loading_pct > 120).sum()
        ),
        "pf_converged": bool(converged.all()),
        "pf_max_error": float(error.max()),
        "pf_max_iterations": int(iterations.max()),
    }


def main():
    print("=" * 70)
    print("IRELAND GRID - S2 REINFORCEMENT EFFECT AUDIT")
    print("=" * 70)
    print()
    print("Operating point : S2_PEAK_DEMAND")
    print("PF method       : AC nonlinear")
    print("Slack handling  : distributed slack")
    print("Dispatch        : unchanged")
    print("Loads           : unchanged")
    print("Network         : read-only audit")
    print()

    results = []

    for label, path in NETWORKS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Network file not found: {path}"
            )

        results.append(
            audit_network(label, path)
        )

    # ------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------
    comparison = pd.DataFrame(results).set_index("network")

    print()
    print("=" * 70)
    print("S2 REINFORCEMENT EFFECT COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "pf_converged",
                "pf_max_error",
                "pf_max_iterations",
                "min_voltage_pu",
                "min_voltage_bus",
                "max_line_loading_pct",
                "max_line",
                "lines_over_100",
                "lines_over_110",
                "lines_over_120",
            ]
        ].to_string()
    )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()