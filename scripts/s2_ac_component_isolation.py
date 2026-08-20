import copy
import warnings

import numpy as np
import pandas as pd
import pypsa


# ================================================================
# S2 AC COMPONENT ISOLATION DIAGNOSTIC
# ================================================================

NETWORK_PATH = "data\\processed\\eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

VOLTAGE_MIN = 0.90
VOLTAGE_MAX = 1.10

warnings.filterwarnings("ignore")


# ================================================================
# HELPERS
# ================================================================

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def network_summary(n):
    print(f"Buses        : {len(n.buses)}")
    print(f"Lines        : {len(n.lines)}")
    print(f"Transformers : {len(n.transformers)}")
    print(f"Links        : {len(n.links)}")
    print(f"Generators   : {len(n.generators)}")
    print(f"Loads        : {len(n.loads)}")


def set_q_zero(n):
    """
    Set all generator and load reactive powers to zero.

    Only modify tables that actually exist and contain the required
    snapshot column.
    """

    if len(n.generators) > 0 and hasattr(n.generators_t, "q_set"):
        if SNAPSHOT in n.generators_t.q_set.index:
            n.generators_t.q_set.loc[SNAPSHOT, :] = 0.0

    if len(n.loads) > 0 and hasattr(n.loads_t, "q_set"):
        if SNAPSHOT in n.loads_t.q_set.index:
            n.loads_t.q_set.loc[SNAPSHOT, :] = 0.0


def balance_active_power(n):
    """
    Make the main AC system exactly balanced by changing only the
    non-wind generation unit.

    This is diagnostic only. It does NOT modify the source network.
    """

    if SNAPSHOT not in n.snapshots:
        raise ValueError(f"Snapshot {SNAPSHOT} not found.")

    generation = n.generators_t.p_set.loc[SNAPSHOT].sum()
    load = n.loads_t.p_set.loc[SNAPSHOT].sum()

    difference = load - generation

    candidates = [
        "eirgrid_non_wind_generation"
    ]

    slack_name = None

    for name in candidates:
        if name in n.generators.index:
            slack_name = name
            break

    if slack_name is None:
        raise RuntimeError(
            "Could not find eirgrid_non_wind_generation."
        )

    current = n.generators_t.p_set.loc[SNAPSHOT, slack_name]

    n.generators_t.p_set.loc[
        SNAPSHOT, slack_name
    ] = current + difference

    return generation, load, difference, slack_name


def configure_slack(n):
    """
    Configure one generator as the AC slack.
    """

    if len(n.generators) == 0:
        raise RuntimeError("No generators available.")

    n.generators["control"] = "PQ"

    slack_name = "eirgrid_non_wind_generation"

    if slack_name not in n.generators.index:
        raise RuntimeError(
            "eirgrid_non_wind_generation not found."
        )

    n.generators.at[slack_name, "control"] = "Slack"

    # Ensure adequate nominal capacity for the diagnostic.
    if "p_nom" in n.generators.columns:
        n.generators.at[slack_name, "p_nom"] = max(
            safe_float(n.generators.at[slack_name, "p_nom"]),
            10000.0
        )

    return slack_name


def determine_topology(n):
    try:
        n.determine_network_topology()
        return True
    except Exception as exc:
        print(f"Topology determination failed: {type(exc).__name__}: {exc}")
        return False


def run_pf(n, label):
    """
    Run AC PF and extract robust diagnostics.
    """

    print("\n" + "-" * 70)
    print(f"RUNNING {label}")
    print("-" * 70)

    try:
        configure_slack(n)

        topology_ok = determine_topology(n)

        if not topology_ok:
            print("RESULT: FAIL — topology determination failed.")
            return {
                "label": label,
                "converged": False,
                "reason": "topology failure",
                "min_v": np.nan,
                "max_v": np.nan,
                "max_line_flow": np.nan,
            }

        pf = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True,
        )

        print("\nPF RESULT")
        print(pf)

        converged = False

        try:
            conv = pf["converged"]

            if isinstance(conv, pd.DataFrame):
                converged = bool(conv.loc[SNAPSHOT].all())
            elif isinstance(conv, pd.Series):
                converged = bool(conv.loc[SNAPSHOT])
            else:
                converged = bool(conv)
        except Exception:
            converged = False

        # --------------------------------------------------------
        # VOLTAGES
        # --------------------------------------------------------

        if SNAPSHOT in n.buses_t.v_mag_pu.index:
            v = n.buses_t.v_mag_pu.loc[SNAPSHOT].astype(float)
            finite = v[np.isfinite(v)]

            if len(finite):
                min_v = finite.min()
                max_v = finite.max()

                suspicious = finite[
                    (finite < VOLTAGE_MIN) |
                    (finite > VOLTAGE_MAX)
                ]

                print("\nVOLTAGE RESULT")
                print(f"Minimum voltage : {min_v:.6g} pu")
                print(f"Maximum voltage : {max_v:.6g} pu")
                print(f"Suspicious buses: {len(suspicious)}")

                if len(suspicious):
                    print(suspicious.sort_values().head(10))
            else:
                min_v = np.nan
                max_v = np.nan
        else:
            min_v = np.nan
            max_v = np.nan

        # --------------------------------------------------------
        # LINE FLOWS
        # --------------------------------------------------------

        max_line_flow = np.nan

        if len(n.lines) > 0:
            try:
                p0 = n.lines_t.p0.loc[SNAPSHOT].astype(float)
                p1 = n.lines_t.p1.loc[SNAPSHOT].astype(float)

                finite0 = p0[np.isfinite(p0)]
                finite1 = p1[np.isfinite(p1)]

                if len(finite0):
                    max_p0 = finite0.abs().max()
                else:
                    max_p0 = np.nan

                if len(finite1):
                    max_p1 = finite1.abs().max()
                else:
                    max_p1 = np.nan

                max_line_flow = np.nanmax(
                    [max_p0, max_p1]
                )

                print("\nLINE FLOW RESULT")
                print(f"Maximum |P0| : {max_p0:.6g} MW")
                print(f"Maximum |P1| : {max_p1:.6g} MW")

            except Exception as exc:
                print(
                    "Line-flow extraction failed:",
                    type(exc).__name__,
                    exc
                )

        if converged:
            print("\nRESULT: PASS — AC PF converged.")
        else:
            print("\nRESULT: FAIL — AC PF did not converge.")

        return {
            "label": label,
            "converged": converged,
            "reason": "PF",
            "min_v": min_v,
            "max_v": max_v,
            "max_line_flow": max_line_flow,
        }

    except Exception as exc:

        print("\nRESULT: EXCEPTION")
        print(type(exc).__name__, exc)

        return {
            "label": label,
            "converged": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "min_v": np.nan,
            "max_v": np.nan,
            "max_line_flow": np.nan,
        }


def prepare_base():
    """
    Load a completely fresh network for every test.
    """

    n = pypsa.Network(NETWORK_PATH)

    if SNAPSHOT not in n.snapshots:
        raise RuntimeError(
            f"{SNAPSHOT} not present in network."
        )

    # ------------------------------------------------------------
    # Remove reactive-power complications
    # ------------------------------------------------------------

    set_q_zero(n)

    # ------------------------------------------------------------
    # Balance active power
    # ------------------------------------------------------------

    generation, load, difference, slack = balance_active_power(n)

    print("\nOPERATING POINT")
    print("-" * 70)
    print(f"Generation before balance : {generation:.6f} MW")
    print(f"Load                       : {load:.6f} MW")
    print(f"Original difference        : {difference:.6f} MW")
    print(f"Balancing generator        : {slack}")
    print(
        f"Balanced generation       : "
        f"{n.generators_t.p_set.loc[SNAPSHOT].sum():.6f} MW"
    )

    return n


# ================================================================
# TEST DEFINITIONS
# ================================================================

def test_1_original():
    n = prepare_base()

    return run_pf(
        n,
        "TEST 1 — BALANCED ORIGINAL NETWORK"
    )


def test_2_no_line_charging():
    n = prepare_base()

    print("\nRemoving AC line charging: b = 0")

    if len(n.lines):
        n.lines["b"] = 0.0

    return run_pf(
        n,
        "TEST 2 — BALANCED + LINE CHARGING REMOVED"
    )


def test_3_no_transformers():
    n = prepare_base()

    print("\nRemoving all transformers")

    n.mremove(
        "Transformer",
        list(n.transformers.index)
    )

    return run_pf(
        n,
        "TEST 3 — BALANCED + NO TRANSFORMERS"
    )


def test_4_no_lines():
    n = prepare_base()

    print("\nRemoving all AC lines")

    n.mremove(
        "Line",
        list(n.lines.index)
    )

    return run_pf(
        n,
        "TEST 4 — BALANCED + NO LINES"
    )


def test_5_no_lines_no_transformers():
    n = prepare_base()

    print("\nRemoving all lines and transformers")

    if len(n.lines):
        n.mremove(
            "Line",
            list(n.lines.index)
        )

    if len(n.transformers):
        n.mremove(
            "Transformer",
            list(n.transformers.index)
        )

    return run_pf(
        n,
        "TEST 5 — BALANCED + NO LINES + NO TRANSFORMERS"
    )


def test_6_no_line_charging_no_transformers():
    n = prepare_base()

    print("\nSetting line charging to zero")

    if len(n.lines):
        n.lines["b"] = 0.0

    print("Removing all transformers")

    if len(n.transformers):
        n.mremove(
            "Transformer",
            list(n.transformers.index)
        )

    return run_pf(
        n,
        "TEST 6 — b=0 + NO TRANSFORMERS"
    )


def test_7_main_220kv_only():
    n = prepare_base()

    print("\nKeeping only 220 kV buses")

    keep_buses = list(
        n.buses.index[
            np.isclose(
                n.buses.v_nom.astype(float),
                220.0
            )
        ]
    )

    remove_buses = [
        b for b in n.buses.index
        if b not in keep_buses
    ]

    print(f"220 kV buses kept   : {len(keep_buses)}")
    print(f"Non-220 kV removed  : {len(remove_buses)}")

    if remove_buses:
        n.mremove("Bus", remove_buses)

    return run_pf(
        n,
        "TEST 7 — BALANCED + 220 kV ONLY"
    )


def test_8_220kv_no_charging():
    n = prepare_base()

    print("\nKeeping only 220 kV buses")

    keep_buses = list(
        n.buses.index[
            np.isclose(
                n.buses.v_nom.astype(float),
                220.0
            )
        ]
    )

    remove_buses = [
        b for b in n.buses.index
        if b not in keep_buses
    ]

    if remove_buses:
        n.mremove("Bus", remove_buses)

    print("Setting b = 0")

    if len(n.lines):
        n.lines["b"] = 0.0

    return run_pf(
        n,
        "TEST 8 — 220 kV ONLY + b=0"
    )


def test_9_no_links():
    n = prepare_base()

    print("\nRemoving links")

    if len(n.links):
        n.mremove(
            "Link",
            list(n.links.index)
        )

    return run_pf(
        n,
        "TEST 9 — BALANCED + NO LINKS"
    )


def test_10_no_transformers_no_links():
    n = prepare_base()

    print("\nRemoving transformers")

    if len(n.transformers):
        n.mremove(
            "Transformer",
            list(n.transformers.index)
        )

    print("Removing links")

    if len(n.links):
        n.mremove(
            "Link",
            list(n.links.index)
        )

    return run_pf(
        n,
        "TEST 10 — NO TRANSFORMERS + NO LINKS"
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    header("S2 AC COMPONENT ISOLATION DIAGNOSTIC")

    print(f"\nNetwork : {NETWORK_PATH}")
    print(f"Snapshot: {SNAPSHOT}")

    base = pypsa.Network(NETWORK_PATH)

    print("\nNETWORK")
    print("-" * 70)
    network_summary(base)

    print("\nBUS VOLTAGES")
    print("-" * 70)
    print(
        base.buses.v_nom.value_counts().sort_index()
    )

    print("\nLINE PARAMETERS")
    print("-" * 70)

    if len(base.lines):

        for col in ["r", "x", "b", "s_nom"]:

            if col in base.lines.columns:

                values = pd.to_numeric(
                    base.lines[col],
                    errors="coerce"
                ).dropna()

                if len(values):

                    print(
                        f"{col:8s} "
                        f"min={values.min():.6g} "
                        f"max={values.max():.6g} "
                        f"mean={values.mean():.6g}"
                    )

    print("\nTRANSFORMERS")
    print("-" * 70)

    if len(base.transformers):

        cols = [
            "bus0",
            "bus1",
            "r",
            "x",
            "s_nom",
            "tap_ratio",
            "phase_shift"
        ]

        available = [
            c for c in cols
            if c in base.transformers.columns
        ]

        print(
            base.transformers[available].to_string()
        )

    # ============================================================
    # RUN TESTS
    # ============================================================

    results = []

    tests = [
        test_1_original,
        test_2_no_line_charging,
        test_3_no_transformers,
        test_4_no_lines,
        test_5_no_lines_no_transformers,
        test_6_no_line_charging_no_transformers,
        test_7_main_220kv_only,
        test_8_220kv_no_charging,
        test_9_no_links,
        test_10_no_transformers_no_links,
    ]

    for test in tests:

        try:
            result = test()
            results.append(result)

        except Exception as exc:

            print("\nTEST CRASHED")
            print(type(exc).__name__, exc)

            results.append({
                "label": test.__name__,
                "converged": False,
                "reason": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "min_v": np.nan,
                "max_v": np.nan,
                "max_line_flow": np.nan,
            })

    # ============================================================
    # SUMMARY
    # ============================================================

    header("DIAGNOSTIC SUMMARY")

    summary = pd.DataFrame(results)

    if len(summary):

        display_summary = summary.copy()

        display_summary["converged"] = (
            display_summary["converged"]
            .map({True: "PASS", False: "FAIL"})
        )

        print(
            display_summary.to_string(
                index=False
            )
        )

    print("\n" + "-" * 70)

    passed = [
        r for r in results
        if r.get("converged", False)
    ]

    if passed:

        print("IMPORTANT:")
        print(
            "At least one isolated configuration converged."
        )
        print(
            "This means the AC failure is associated with "
            "one or more removed component classes."
        )

        print("\nConverged configurations:")

        for r in passed:
            print(
                f"  PASS — {r['label']}"
            )

    else:

        print("IMPORTANT:")
        print(
            "No isolated configuration converged."
        )
        print(
            "The next step should be bus/branch-level "
            "electrical parameter diagnosis."
        )

    print("\nDo NOT modify reinforcement based on this test.")

    header("S2 AC COMPONENT ISOLATION COMPLETE")