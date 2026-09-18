import pandas as pd
import pypsa
from pathlib import Path

INPUT = Path("data/processed/eirgrid_interconnected_scenarios.nc")
OUTPUT = Path("data/processed/eirgrid_interconnected_powerflow.csv")

print("=" * 70)
print("   IRELAND GRID - INTERCONNECTED EIRGRID SCENARIO POWER FLOW")
print("=" * 70)

print()
print("Loading:")
print(INPUT)

n = pypsa.Network(INPUT)

print("OK: Network loaded.")

print()
print("-" * 70)
print("NETWORK")
print("-" * 70)

print(f"Buses        : {len(n.buses)}")
print(f"Lines        : {len(n.lines)}")
print(f"Transformers : {len(n.transformers)}")
print(f"Generators   : {len(n.generators)}")
print(f"Loads        : {len(n.loads)}")
print(f"Links        : {len(n.links)}")

results = []

print()
print("-" * 70)
print("INTERCONNECTED SCENARIO POWER FLOW")
print("-" * 70)

for snapshot in n.snapshots:

    print()
    print("=" * 70)
    print(f"SCENARIO: {snapshot}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Run linear power flow
    # ------------------------------------------------------------

    n.pf(
        snapshots=[snapshot],
        use_seed=True
    )

    # ------------------------------------------------------------
    # LINE LOADING
    # ------------------------------------------------------------

    line_flow = n.lines_t.p0.loc[snapshot].abs()
    line_limit = n.lines.s_nom

    line_loading = (
        line_flow / line_limit * 100
    )

    max_line_loading = line_loading.max()

    overloaded_lines = (
        line_loading > 100
    ).sum()

    # ------------------------------------------------------------
    # TRANSFORMER LOADING
    # ------------------------------------------------------------

    if len(n.transformers) > 0:

        transformer_flow = (
            n.transformers_t.p0.loc[snapshot].abs()
        )

        transformer_limit = n.transformers.s_nom

        transformer_loading = (
            transformer_flow / transformer_limit * 100
        )

        max_transformer_loading = transformer_loading.max()

        overloaded_transformers = (
            transformer_loading > 100
        ).sum()

    else:

        max_transformer_loading = 0
        overloaded_transformers = 0

    # ------------------------------------------------------------
    # INTERCONNECTOR FLOWS
    # ------------------------------------------------------------

    if len(n.links) > 0:

        link_flow = n.links_t.p0.loc[snapshot]

    else:

        link_flow = pd.Series(dtype=float)

    # ------------------------------------------------------------
    # PRINT TRANSMISSION RESULTS
    # ------------------------------------------------------------

    print()
    print("TRANSMISSION")

    print(
        f"Maximum line loading : "
        f"{max_line_loading:.2f}%"
    )

    print(
        f"Overloaded lines     : "
        f"{overloaded_lines}"
    )

    if overloaded_lines > 0:

        print()
        print("Overloaded lines:")

        overloaded = line_loading[
            line_loading > 100
        ].sort_values(ascending=False)

        for name, loading in overloaded.items():

            flow = line_flow[name]
            limit = line_limit[name]

            line = n.lines.loc[name]

            print()
            print(f"  {name}")
            print(f"    Flow    : {flow:.2f} MW")
            print(f"    Limit   : {limit:.2f} MW")
            print(f"    Loading : {loading:.2f}%")
            print(f"    From    : {line.bus0}")
            print(f"    To      : {line.bus1}")

    # ------------------------------------------------------------
    # TRANSFORMERS
    # ------------------------------------------------------------

    print()
    print("TRANSFORMERS")

    print(
        f"Maximum transformer loading : "
        f"{max_transformer_loading:.2f}%"
    )

    print(
        f"Overloaded transformers     : "
        f"{overloaded_transformers}"
    )

    # ------------------------------------------------------------
    # INTERCONNECTORS
    # ------------------------------------------------------------

    print()
    print("INTERCONNECTORS")

    for link_name, flow in link_flow.items():

        link = n.links.loc[link_name]

        print()
        print(f"  {link_name}")
        print(f"    Flow : {flow:.2f} MW")
        print(f"    From : {link.bus0}")
        print(f"    To   : {link.bus1}")

    # ------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------

    if overloaded_lines > 0:

        status = "THERMAL VIOLATIONS"

    elif overloaded_transformers > 0:

        status = "TRANSFORMER VIOLATIONS"

    else:

        status = "NO THERMAL VIOLATIONS"

    print()
    print(f"STATUS: {status}")

    # ------------------------------------------------------------
    # SAVE RESULT
    # ------------------------------------------------------------

    results.append({
        "Scenario": snapshot,
        "Max_Line_Loading_Percent": max_line_loading,
        "Overloaded_Lines": overloaded_lines,
        "Max_Transformer_Loading_Percent": max_transformer_loading,
        "Overloaded_Transformers": overloaded_transformers,
        "Status": status
    })


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print("        INTERCONNECTED POWER FLOW SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print()
print("Saved:")
print(OUTPUT)

print()
print("=" * 70)
print("INTERCONNECTED POWER-FLOW COMPLETE")
print("=" * 70)

print()
print("NEXT:")
print("Identify recurring transmission bottlenecks")
print("before performing network optimisation.")