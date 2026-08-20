import copy
import numpy as np
import pandas as pd
import pypsa


# ============================================================================
# CONFIG
# ============================================================================

NETWORK_PATH = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

TOL_V_MIN = 0.90
TOL_V_MAX = 1.10

SBASE_MVA = 100.0


# ============================================================================
# HELPERS
# ============================================================================

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def remove_components(n, component, names):
    """
    Compatible removal using drop(..., inplace=True).
    Avoids obsolete Network.mremove().
    """

    names = list(names)

    if not names:
        return

    table = getattr(n, component)

    existing = [x for x in names if x in table.index]

    if not existing:
        return

    table.drop(index=existing, inplace=True)


def run_pf(n, label):
    """
    Run AC PF and return compact diagnostic information.
    """

    print()
    print("-" * 78)
    print(f"RUNNING: {label}")
    print("-" * 78)

    try:

        pf = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-8,
            use_seed=True,
        )

        print("\nPF RETURN")
        print(pf)

        converged = bool(
            pf["converged"].loc[SNAPSHOT].iloc[0]
        )

        print("\nCONVERGED:", converged)

    except Exception as e:

        print("\nPF EXCEPTION")
        print(type(e).__name__, str(e))

        return {
            "label": label,
            "converged": False,
            "exception": str(e),
            "min_v": np.nan,
            "max_v": np.nan,
            "max_line_flow": np.nan,
        }

    # ----------------------------------------------------------------------
    # VOLTAGE
    # ----------------------------------------------------------------------

    try:

        v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

        finite_v = v[np.isfinite(v)]

        if len(finite_v):

            min_v = float(finite_v.min())
            max_v = float(finite_v.max())

            suspicious = v[
                (~np.isfinite(v))
                | (v < TOL_V_MIN)
                | (v > TOL_V_MAX)
            ]

        else:

            min_v = np.nan
            max_v = np.nan
            suspicious = v

        print("\nVOLTAGE")
        print("Min:", min_v)
        print("Max:", max_v)
        print("Suspicious buses:", len(suspicious))

        if len(suspicious):
            print(suspicious.head(15))

    except Exception as e:

        print("\nVoltage diagnostic failed:", e)

        min_v = np.nan
        max_v = np.nan

    # ----------------------------------------------------------------------
    # LINE FLOWS
    # ----------------------------------------------------------------------

    try:

        p0 = n.lines_t.p0.loc[SNAPSHOT]
        p1 = n.lines_t.p1.loc[SNAPSHOT]

        max_p0 = float(np.nanmax(np.abs(p0)))
        max_p1 = float(np.nanmax(np.abs(p1)))

        max_line_flow = max(max_p0, max_p1)

        print("\nLINE FLOWS")
        print("Maximum |P0|:", max_p0, "MW")
        print("Maximum |P1|:", max_p1, "MW")

    except Exception:

        max_line_flow = np.nan

    return {
        "label": label,
        "converged": converged,
        "exception": "",
        "min_v": min_v,
        "max_v": max_v,
        "max_line_flow": max_line_flow,
    }


def balance_slack(n):

    """
    Balance active power using the existing non-wind generator.
    """

    if "eirgrid_non_wind_generation" not in n.generators.index:
        raise RuntimeError(
            "eirgrid_non_wind_generation not found"
        )

    gen = n.generators.index

    total_load = float(
        n.loads_t.p_set.loc[SNAPSHOT].sum()
    )

    other_generation = 0.0

    for g in gen:

        if g == "eirgrid_non_wind_generation":
            continue

        other_generation += float(
            n.generators_t.p_set.loc[SNAPSHOT, g]
        )

    required_slack = total_load - other_generation

    n.generators_t.p_set.loc[
        SNAPSHOT,
        "eirgrid_non_wind_generation"
    ] = required_slack

    return required_slack


# ============================================================================
# LOAD NETWORK
# ============================================================================

banner("S2 AC TOPOLOGY / ELECTRICAL DIAGNOSTIC")

print()
print("Network :", NETWORK_PATH)
print("Snapshot:", SNAPSHOT)

n0 = pypsa.Network(NETWORK_PATH)

print()
print("NETWORK")
print("-" * 78)

print("Buses       :", len(n0.buses))
print("Lines       :", len(n0.lines))
print("Transformers:", len(n0.transformers))
print("Generators  :", len(n0.generators))
print("Loads       :", len(n0.loads))
print("Links       :", len(n0.links))


# ============================================================================
# 1. BUS AUDIT
# ============================================================================

banner("1. BUS ELECTRICAL AUDIT")

print("\nVOLTAGE LEVELS")
print(n0.buses.v_nom.value_counts())

invalid_v = n0.buses[
    (~np.isfinite(n0.buses.v_nom))
    | (n0.buses.v_nom <= 0)
]

print("\nInvalid v_nom buses:", len(invalid_v))

if len(invalid_v):
    print(invalid_v[["v_nom"]])


# ============================================================================
# 2. TOPOLOGY
# ============================================================================

banner("2. TOPOLOGY AUDIT")

# determine topology without running PF

n_top = copy.deepcopy(n0)

try:

    n_top.determine_network_topology()

    print("\nSUBNETWORKS")
    print(n_top.sub_networks)

except Exception as e:

    print("Topology determination failed:", e)


# --------------------------------------------------------------------------
# Connected components using NetworkX
# --------------------------------------------------------------------------

try:

    import networkx as nx

    G = nx.Graph()

    for bus in n0.buses.index:
        G.add_node(bus)

    # Lines
    for name, row in n0.lines.iterrows():

        if row.bus0 in G and row.bus1 in G:

            G.add_edge(
                row.bus0,
                row.bus1,
                component="line",
                name=name,
            )

    # Transformers
    for name, row in n0.transformers.iterrows():

        if row.bus0 in G and row.bus1 in G:

            G.add_edge(
                row.bus0,
                row.bus1,
                component="transformer",
                name=name,
            )

    components = list(nx.connected_components(G))

    components = sorted(
        components,
        key=len,
        reverse=True,
    )

    print("\nCONNECTED COMPONENTS")
    print("Number:", len(components))

    for i, comp in enumerate(components):

        print(
            f"Component {i}: {len(comp)} buses"
        )

        if len(comp) <= 15:
            print(sorted(comp))

except Exception as e:

    print("NetworkX topology check failed:", e)


# ============================================================================
# 3. BUS INJECTION AUDIT
# ============================================================================

banner("3. BUS INJECTION AUDIT")

bus_p = pd.Series(
    0.0,
    index=n0.buses.index
)

# Loads are negative injections

for load in n0.loads.index:

    bus = n0.loads.at[load, "bus"]

    if bus in bus_p.index:

        bus_p.loc[bus] -= float(
            n0.loads_t.p_set.loc[SNAPSHOT, load]
        )


# Generators

for gen in n0.generators.index:

    bus = n0.generators.at[gen, "bus"]

    if bus in bus_p.index:

        bus_p.loc[bus] += float(
            n0.generators_t.p_set.loc[SNAPSHOT, gen]
        )


bus_injection = pd.DataFrame({
    "P_net_MW": bus_p
})

bus_injection["abs_P_MW"] = (
    bus_injection["P_net_MW"].abs()
)

print(
    "\nLargest absolute bus injections:"
)

print(
    bus_injection
    .sort_values("abs_P_MW", ascending=False)
    .head(20)
)


# ============================================================================
# 4. LINE PER-UNIT AUDIT
# ============================================================================

banner("4. LINE PER-UNIT IMPEDANCE AUDIT")

print(
    "\nUsing Sbase =",
    SBASE_MVA,
    "MVA"
)

line_audit = []

for name, row in n0.lines.iterrows():

    bus0 = row.bus0
    bus1 = row.bus1

    v0 = safe_float(
        n0.buses.at[bus0, "v_nom"]
    )

    v1 = safe_float(
        n0.buses.at[bus1, "v_nom"]
    )

    # Use nominal voltage of bus0.
    # Lines should normally connect same-voltage buses.

    zbase = (v0 ** 2) / SBASE_MVA

    r_pu = row.r / zbase
    x_pu = row.x / zbase

    line_audit.append({
        "name": name,
        "bus0": bus0,
        "bus1": bus1,
        "v0_kV": v0,
        "v1_kV": v1,
        "r_ohm": row.r,
        "x_ohm": row.x,
        "r_pu": r_pu,
        "x_pu": x_pu,
        "s_nom": row.s_nom,
    })

line_audit = pd.DataFrame(line_audit)

print("\nHIGHEST X PU")

print(
    line_audit
    .sort_values("x_pu", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nLOWEST X PU")

print(
    line_audit
    .sort_values("x_pu")
    .head(15)
    .to_string(index=False)
)

print("\nLINE PU STATISTICS")

print(
    line_audit[
        ["r_pu", "x_pu"]
    ].describe()
)


# ============================================================================
# 5. CROSS-VOLTAGE LINE AUDIT
# ============================================================================

banner("5. CROSS-VOLTAGE LINE AUDIT")

cross_voltage = []

for name, row in n0.lines.iterrows():

    v0 = float(
        n0.buses.at[row.bus0, "v_nom"]
    )

    v1 = float(
        n0.buses.at[row.bus1, "v_nom"]
    )

    if abs(v0 - v1) > 1e-9:

        cross_voltage.append({
            "line": name,
            "bus0": row.bus0,
            "bus1": row.bus1,
            "v0": v0,
            "v1": v1,
        })

print(
    "Cross-voltage AC lines:",
    len(cross_voltage)
)

if cross_voltage:

    print(
        pd.DataFrame(cross_voltage)
        .to_string(index=False)
    )


# ============================================================================
# 6. TRANSFORMER AUDIT
# ============================================================================

banner("6. TRANSFORMER ELECTRICAL AUDIT")

print(
    n0.transformers[
        [
            "bus0",
            "bus1",
            "r",
            "x",
            "s_nom",
            "tap_ratio",
            "phase_shift",
        ]
    ].to_string()
)


# ============================================================================
# 7. BASELINE BALANCED TEST
# ============================================================================

banner("7. BASELINE BALANCED TEST")

n_base = copy.deepcopy(n0)

required_slack = balance_slack(n_base)

print(
    "\nRequired slack generation:",
    required_slack,
    "MW"
)

print(
    "Total load:",
    n_base.loads_t.p_set.loc[SNAPSHOT].sum(),
    "MW"
)

print(
    "Total generation after balance:",
    n_base.generators_t.p_set.loc[SNAPSHOT].sum(),
    "MW"
)

# Set Q loads to zero for clean diagnostic

if hasattr(n_base.loads_t, "q_set"):

    n_base.loads_t.q_set.loc[
        SNAPSHOT,
        :
    ] = 0.0

# Generator Q zero

if hasattr(n_base.generators_t, "q_set"):

    n_base.generators_t.q_set.loc[
        SNAPSHOT,
        :
    ] = 0.0


results = []

results.append(
    run_pf(
        n_base,
        "TEST 1 — BALANCED ORIGINAL"
    )
)


# ============================================================================
# 8. DC POWER FLOW
# ============================================================================

banner("8. DC POWER-FLOW CROSS-CHECK")

n_dc = copy.deepcopy(n0)

balance_slack(n_dc)

try:

    dc = n_dc.pf(
        snapshots=[SNAPSHOT],
        use_seed=True,
    )

    print("\nAC PF already attempted above.")

except Exception as e:

    print(
        "Standard PF failed:",
        type(e).__name__,
        str(e)
    )


# PyPSA has a separate lpf method

try:

    lpf_result = n_dc.lpf(
        snapshots=[SNAPSHOT]
    )

    print("\nLINEAR POWER FLOW RESULT")
    print(lpf_result)

    if hasattr(n_dc.lines_t, "p0"):

        p0 = n_dc.lines_t.p0.loc[SNAPSHOT]

        print(
            "\nMaximum absolute linear line flow:",
            np.nanmax(np.abs(p0)),
            "MW"
        )

except Exception as e:

    print(
        "\nLinear PF failed:",
        type(e).__name__,
        str(e)
    )


# ============================================================================
# 9. REMOVE TRANSFORMERS CORRECTLY
# ============================================================================

banner("9. TRANSFORMER ISOLATION")

n_no_tr = copy.deepcopy(n0)

balance_slack(n_no_tr)

remove_components(
    n_no_tr,
    "transformers",
    n_no_tr.transformers.index,
)

print(
    "\nTransformers remaining:",
    len(n_no_tr.transformers)
)

results.append(
    run_pf(
        n_no_tr,
        "TEST 2 — NO TRANSFORMERS"
    )
)


# ============================================================================
# 10. REMOVE LINKS
# ============================================================================

banner("10. LINK ISOLATION")

n_no_links = copy.deepcopy(n0)

balance_slack(n_no_links)

remove_components(
    n_no_links,
    "links",
    n_no_links.links.index,
)

print(
    "\nLinks remaining:",
    len(n_no_links.links)
)

results.append(
    run_pf(
        n_no_links,
        "TEST 3 — NO LINKS"
    )
)


# ============================================================================
# 11. REMOVE LINE CHARGING
# ============================================================================

banner("11. ZERO LINE CHARGING")

n_no_b = copy.deepcopy(n0)

balance_slack(n_no_b)

n_no_b.lines.loc[:, "b"] = 0.0

results.append(
    run_pf(
        n_no_b,
        "TEST 4 — BALANCED + b=0"
    )
)


# ============================================================================
# 12. REMOVE TRANSFORMERS + LINE CHARGING
# ============================================================================

banner("12. ZERO CHARGING + NO TRANSFORMERS")

n_simple = copy.deepcopy(n0)

balance_slack(n_simple)

n_simple.lines.loc[:, "b"] = 0.0

remove_components(
    n_simple,
    "transformers",
    n_simple.transformers.index,
)

results.append(
    run_pf(
        n_simple,
        "TEST 5 — b=0 + NO TRANSFORMERS"
    )
)


# ============================================================================
# 13. 220 KV ONLY
# ============================================================================

banner("13. 220 KV MAIN GRID ISOLATION")

n_220 = copy.deepcopy(n0)

balance_slack(n_220)

keep_buses = n_220.buses.index[
    np.isclose(
        n_220.buses.v_nom,
        220.0
    )
]

remove_buses = [
    b for b in n_220.buses.index
    if b not in keep_buses
]

print(
    "220 kV buses kept:",
    len(keep_buses)
)

print(
    "Non-220 kV buses:",
    len(remove_buses)
)

# Remove dependent components FIRST

for component in [
    "generators",
    "loads",
]:

    table = getattr(n_220, component)

    bad = [
        x for x in table.index
        if table.at[x, "bus"] in remove_buses
    ]

    remove_components(
        n_220,
        component,
        bad
    )


# Lines

bad_lines = []

for name, row in n_220.lines.iterrows():

    if (
        row.bus0 in remove_buses
        or row.bus1 in remove_buses
    ):

        bad_lines.append(name)

remove_components(
    n_220,
    "lines",
    bad_lines
)


# Transformers

bad_tr = []

for name, row in n_220.transformers.iterrows():

    if (
        row.bus0 in remove_buses
        or row.bus1 in remove_buses
    ):

        bad_tr.append(name)

remove_components(
    n_220,
    "transformers",
    bad_tr
)


# Links

bad_links = []

for name, row in n_220.links.iterrows():

    if (
        row.bus0 in remove_buses
        or row.bus1 in remove_buses
    ):

        bad_links.append(name)

remove_components(
    n_220,
    "links",
    bad_links
)


# Finally buses

remove_components(
    n_220,
    "buses",
    remove_buses
)

print(
    "\nFinal 220 kV network:"
)

print(
    "Buses:",
    len(n_220.buses)
)

print(
    "Lines:",
    len(n_220.lines)
)

print(
    "Transformers:",
    len(n_220.transformers)
)

print(
    "Generators:",
    len(n_220.generators)
)

print(
    "Loads:",
    len(n_220.loads)
)

results.append(
    run_pf(
        n_220,
        "TEST 6 — 220 kV ONLY"
    )
)


# ============================================================================
# 14. SUMMARY
# ============================================================================

banner("FINAL DIAGNOSTIC SUMMARY")

summary = pd.DataFrame(results)

print(
    summary.to_string(index=False)
)


# ============================================================================
# 15. INTERPRETATION
# ============================================================================

banner("INTERPRETATION")

if any(summary["converged"]):

    print(
        """
At least one isolated configuration converged.

This means the failure is localized to one or more
specific network components/topological features.

Next step:
identify exactly which configuration converged and
perform branch-level isolation around the failed component.
"""
    )

else:

    print(
        """
NO TEST CONVERGED.

The basic r/x/b/s_nom parameter ranges are not obviously
invalid, so the next suspects are:

1. bus connectivity/topology
2. component electrical scaling
3. generator/load placement
4. transformer parameter interpretation
5. one or more pathological branch equations
6. initialization / voltage-angle starting point

DO NOT MODIFY REINFORCEMENT YET.
"""
    )


print()
print("=" * 78)
print("S2 AC TOPOLOGY / ELECTRICAL DIAGNOSTIC COMPLETE")
print("=" * 78)