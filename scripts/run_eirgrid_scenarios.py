import pypsa
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/eirgrid_scenarios.nc")

print("=" * 70)
print("       IRELAND GRID - EIRGRID SCENARIO POWER FLOW")
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

print()
print("-" * 70)
print("SCENARIO POWER FLOW")
print("-" * 70)

results = []

for scenario in n.snapshots:

    print()
    print("=" * 70)
    print(f"SCENARIO: {scenario}")
    print("=" * 70)

    demand = n.loads_t.p_set.loc[scenario].sum()
    generation = n.generators_t.p_set.loc[scenario].sum()

    print()
    print(f"Demand     : {demand:,.2f} MW")
    print(f"Generation : {generation:,.2f} MW")
    print(f"Balance    : {generation - demand:,.2f} MW")

    # Run linear power flow
    n.lpf()

    # --------------------------------------------------------
    # LINE LOADING
    # --------------------------------------------------------

    line_flow = n.lines_t.p0.loc[scenario].abs()
    line_limit = n.lines.s_nom

    line_loading = (
        line_flow / line_limit * 100
    )

    max_line = line_loading.max()
    overloaded_lines = line_loading[line_loading > 100]

    # --------------------------------------------------------
    # TRANSFORMER LOADING
    # --------------------------------------------------------

    if len(n.transformers) > 0:

        transformer_flow = (
            n.transformers_t.p0.loc[scenario].abs()
        )

        transformer_limit = n.transformers.s_nom

        transformer_loading = (
            transformer_flow
            / transformer_limit
            * 100
        )

        max_transformer = transformer_loading.max()

        overloaded_transformers = (
            transformer_loading[
                transformer_loading > 100
            ]
        )

    else:

        max_transformer = 0.0
        overloaded_transformers = pd.Series(dtype=float)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("TRANSMISSION")

    print(
        f"Maximum line loading : "
        f"{max_line:.2f}%"
    )

    print(
        f"Overloaded lines     : "
        f"{len(overloaded_lines)}"
    )

    if len(overloaded_lines) > 0:

        print()
        print("Overloaded lines:")

        for line_name in overloaded_lines.index:

            flow = line_flow[line_name]
            limit = line_limit[line_name]
            loading = line_loading[line_name]

            print()
            print(f"  {line_name}")
            print(f"    Flow    : {flow:.2f} MW")
            print(f"    Limit   : {limit:.2f} MW")
            print(f"    Loading : {loading:.2f}%")
            print(
                f"    From    : "
                f"{n.lines.at[line_name, 'bus0']}"
            )
            print(
                f"    To      : "
                f"{n.lines.at[line_name, 'bus1']}"
            )

    print()
    print("TRANSFORMERS")

    print(
        f"Maximum transformer loading : "
        f"{max_transformer:.2f}%"
    )

    print(
        f"Overloaded transformers     : "
        f"{len(overloaded_transformers)}"
    )

    status = (
        "THERMAL VIOLATIONS"
        if len(overloaded_lines) > 0
        or len(overloaded_transformers) > 0
        else "NO THERMAL VIOLATIONS"
    )

    print()
    print(f"STATUS: {status}")

    results.append({
        "Scenario": scenario,
        "Demand_MW": demand,
        "Generation_MW": generation,
        "Balance_MW": generation - demand,
        "Max_Line_Loading_Percent": max_line,
        "Overloaded_Lines": len(overloaded_lines),
        "Max_Transformer_Loading_Percent": max_transformer,
        "Overloaded_Transformers": len(
            overloaded_transformers
        ),
        "Status": status,
    })

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

output = Path(
    "data/processed/eirgrid_powerflow_results.csv"
)

results_df.to_csv(output, index=False)

print()
print("=" * 70)
print("                 EIRGRID POWER FLOW SUMMARY")
print("=" * 70)

print()

print(
    results_df.to_string(index=False)
)

print()
print("Saved:")
print(output)

print()
print("=" * 70)
print("POWER-FLOW VALIDATION COMPLETE")
print("=" * 70)