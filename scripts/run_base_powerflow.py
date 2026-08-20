import os
import sys
import pypsa
import pandas as pd

NETWORK_FILE = "data/processed/ireland_base_scenario.nc"


def main():

    print("=" * 70)
    print("       IRELAND GRID - BASE CASE POWER FLOW")
    print("=" * 70)

    if not os.path.exists(NETWORK_FILE):
        print("\nERROR: Network file not found:")
        print(NETWORK_FILE)
        sys.exit(1)

    print("\nLoading:")
    print(NETWORK_FILE)

    try:
        n = pypsa.Network(NETWORK_FILE)
    except Exception as e:
        print("\nERROR: Could not load network.")
        print(e)
        sys.exit(1)

    print("OK: Network loaded.")

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("NETWORK")
    print("-" * 70)

    print(f"Buses        : {len(n.buses)}")
    print(f"Lines        : {len(n.lines)}")
    print(f"Transformers : {len(n.transformers)}")
    print(f"Generators   : {len(n.generators)}")
    print(f"Loads        : {len(n.loads)}")

    # ---------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------

    n.set_snapshots(["base"])

    # ---------------------------------------------------------
    # Power balance
    # ---------------------------------------------------------

    total_load = n.loads_t.p_set.loc["base"].sum()
    total_generation = n.generators_t.p_set.loc["base"].sum()

    print("\n" + "-" * 70)
    print("POWER BALANCE")
    print("-" * 70)

    print(f"Total demand     : {total_load:.2f} MW")
    print(f"Total generation : {total_generation:.2f} MW")
    print(
        f"Balance          : "
        f"{total_generation - total_load:.2f} MW"
    )

    # ---------------------------------------------------------
    # Run linear power flow
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("LINEAR POWER FLOW")
    print("-" * 70)

    try:
        n.lpf()

        print("OK: Linear power flow completed.")

    except Exception as e:

        print("\nFAILED: Linear power flow.")

        print("Reason:")
        print(e)

        sys.exit(1)

    # ---------------------------------------------------------
    # Line loading
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRANSMISSION LINE LOADING")
    print("-" * 70)

    if len(n.lines) > 0:

        line_flow = n.lines_t.p0.loc["base"].abs()

        line_loading = (
            line_flow
            / n.lines["s_nom"]
            * 100
        )

        line_results = pd.DataFrame({
            "flow_mw": line_flow,
            "thermal_limit_mw": n.lines["s_nom"],
            "loading_percent": line_loading
        })

        line_results = line_results.sort_values(
            "loading_percent",
            ascending=False
        )

        print(
            line_results.head(15).to_string()
        )

        max_line = line_results.iloc[0]

        print(
            f"\nMaximum line loading: "
            f"{max_line['loading_percent']:.2f}%"
        )

        overloaded_lines = line_results[
            line_results["loading_percent"] > 100
        ]

        print(
            f"Lines above 100%: "
            f"{len(overloaded_lines)}"
        )

    # ---------------------------------------------------------
    # Transformer loading
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRANSFORMER LOADING")
    print("-" * 70)

    if len(n.transformers) > 0:

        trafo_flow = (
            n.transformers_t.p0.loc["base"].abs()
        )

        trafo_loading = (
            trafo_flow
            / n.transformers["s_nom"]
            * 100
        )

        trafo_results = pd.DataFrame({
            "flow_mw": trafo_flow,
            "thermal_limit_mw": n.transformers["s_nom"],
            "loading_percent": trafo_loading
        })

        trafo_results = trafo_results.sort_values(
            "loading_percent",
            ascending=False
        )

        print(
            trafo_results.to_string()
        )

        max_trafo = trafo_results.iloc[0]

        print(
            f"\nMaximum transformer loading: "
            f"{max_trafo['loading_percent']:.2f}%"
        )

        overloaded_trafos = trafo_results[
            trafo_results["loading_percent"] > 100
        ]

        print(
            f"Transformers above 100%: "
            f"{len(overloaded_trafos)}"
        )

    # ---------------------------------------------------------
    # Voltage information
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("BUS VOLTAGES")
    print("-" * 70)

    if hasattr(n, "buses_t") and "v_ang" in n.buses_t:

        angles = n.buses_t.v_ang.loc["base"]

        print(
            f"Minimum voltage angle : "
            f"{angles.min():.6f}"
        )

        print(
            f"Maximum voltage angle : "
            f"{angles.max():.6f}"
        )

    # ---------------------------------------------------------
    # Final classification
    # ---------------------------------------------------------

    max_line_loading = (
        line_results["loading_percent"].max()
        if len(n.lines) > 0
        else 0
    )

    max_trafo_loading = (
        trafo_results["loading_percent"].max()
        if len(n.transformers) > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("                 BASE CASE RESULT")
    print("=" * 70)

    print(
        f"\nMaximum line loading        : "
        f"{max_line_loading:.2f}%"
    )

    print(
        f"Maximum transformer loading : "
        f"{max_trafo_loading:.2f}%"
    )

    if max_line_loading <= 100 and max_trafo_loading <= 100:

        print("\nSTATUS: BASE CASE WITHIN THERMAL LIMITS")

    else:

        print("\nSTATUS: BASE CASE HAS THERMAL VIOLATIONS")

    print("\nIMPORTANT:")
    print(
        "This is a synthetic base-case operating scenario."
    )
    print(
        "It is being used to test the electrical behaviour "
        "of the model."
    )
    print(
        "It is NOT yet a representation of an actual "
        "EirGrid operating condition."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()