import pypsa
import os
import sys


NETWORK_FILE = "data/processed/ireland_network.nc"


def main():
    print("=" * 70)
    print("        IRELAND GRID NETWORK VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Check network file
    # ------------------------------------------------------------
    if not os.path.exists(NETWORK_FILE):
        print(f"\nERROR: Network file not found:")
        print(f"  {NETWORK_FILE}")
        sys.exit(1)

    print(f"\nLoading network:")
    print(f"  {NETWORK_FILE}")

    try:
        n = pypsa.Network(NETWORK_FILE)
    except Exception as e:
        print("\nERROR: Could not load PyPSA network.")
        print(e)
        sys.exit(1)

    print("OK: Network loaded successfully.")

    # ------------------------------------------------------------
    # 2. Basic network size
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. NETWORK SIZE")
    print("-" * 70)

    print(f"Buses         : {len(n.buses)}")
    print(f"Lines         : {len(n.lines)}")
    print(f"Transformers  : {len(n.transformers)}")
    print(f"Generators    : {len(n.generators)}")
    print(f"Loads         : {len(n.loads)}")
    print(f"Storage units : {len(n.storage_units)}")

    # ------------------------------------------------------------
    # 3. Check buses
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. BUS VALIDATION")
    print("-" * 70)

    missing_bus_names = n.buses.index[n.buses.index.isna()]

    if len(missing_bus_names) == 0:
        print("OK: No missing bus names.")
    else:
        print(f"WARNING: {len(missing_bus_names)} buses have missing names.")

    if "v_nom" in n.buses.columns:
        invalid_voltage = n.buses[
            n.buses["v_nom"].isna() | (n.buses["v_nom"] <= 0)
        ]

        if len(invalid_voltage) == 0:
            print("OK: All buses have valid nominal voltages.")
        else:
            print(
                f"WARNING: {len(invalid_voltage)} buses have "
                "invalid nominal voltages."
            )

        print("\nVoltage levels:")
        print(n.buses["v_nom"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------
    # 4. Check lines
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. LINE VALIDATION")
    print("-" * 70)

    if len(n.lines) == 0:
        print("WARNING: No transmission lines found.")
    else:
        print(f"OK: {len(n.lines)} transmission lines found.")

        invalid_lines = n.lines[
            n.lines["bus0"].isna()
            | n.lines["bus1"].isna()
            | n.lines["r"].isna()
            | n.lines["x"].isna()
            | n.lines["s_nom"].isna()
            | (n.lines["s_nom"] <= 0)
        ]

        if len(invalid_lines) == 0:
            print("OK: All lines have valid buses, impedance and thermal limits.")
        else:
            print(
                f"WARNING: {len(invalid_lines)} lines contain "
                "invalid/missing electrical data."
            )

        missing_bus_connections = []

        for name, row in n.lines.iterrows():
            if row["bus0"] not in n.buses.index:
                missing_bus_connections.append((name, row["bus0"]))

            if row["bus1"] not in n.buses.index:
                missing_bus_connections.append((name, row["bus1"]))

        if not missing_bus_connections:
            print("OK: All line bus connections exist.")
        else:
            print(
                f"WARNING: {len(missing_bus_connections)} line endpoints "
                "refer to missing buses."
            )

    # ------------------------------------------------------------
    # 5. Check transformers
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. TRANSFORMER VALIDATION")
    print("-" * 70)

    if len(n.transformers) == 0:
        print("WARNING: No transformers found.")
    else:
        print(f"OK: {len(n.transformers)} transformers found.")

        invalid_transformers = n.transformers[
            n.transformers["bus0"].isna()
            | n.transformers["bus1"].isna()
            | n.transformers["r"].isna()
            | n.transformers["x"].isna()
            | n.transformers["s_nom"].isna()
            | (n.transformers["s_nom"] <= 0)
        ]

        if len(invalid_transformers) == 0:
            print(
                "OK: All transformers have valid buses, "
                "impedance and ratings."
            )
        else:
            print(
                f"WARNING: {len(invalid_transformers)} transformers "
                "contain invalid/missing data."
            )

        missing_trafo_connections = []

        for name, row in n.transformers.iterrows():
            if row["bus0"] not in n.buses.index:
                missing_trafo_connections.append((name, row["bus0"]))

            if row["bus1"] not in n.buses.index:
                missing_trafo_connections.append((name, row["bus1"]))

        if not missing_trafo_connections:
            print("OK: All transformer bus connections exist.")
        else:
            print(
                f"WARNING: {len(missing_trafo_connections)} transformer "
                "endpoints refer to missing buses."
            )

    # ------------------------------------------------------------
    # 6. Check electrical parameter ranges
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. ELECTRICAL PARAMETER CHECK")
    print("-" * 70)

    if len(n.lines) > 0:
        negative_r = (n.lines["r"] < 0).sum()
        negative_x = (n.lines["x"] < 0).sum()

        print(f"Lines with negative resistance : {negative_r}")
        print(f"Lines with negative reactance   : {negative_x}")

        if negative_r == 0 and negative_x == 0:
            print("OK: No negative line resistance/reactance detected.")

    if len(n.transformers) > 0:
        negative_r = (n.transformers["r"] < 0).sum()
        negative_x = (n.transformers["x"] < 0).sum()

        print(f"Transformers with negative resistance : {negative_r}")
        print(f"Transformers with negative reactance   : {negative_x}")

        if negative_r == 0 and negative_x == 0:
            print("OK: No negative transformer resistance/reactance detected.")

    # ------------------------------------------------------------
    # 7. Check thermal ratings
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. THERMAL LIMIT CHECK")
    print("-" * 70)

    if len(n.lines) > 0:
        print(
            f"Line thermal limits: "
            f"{n.lines.s_nom.min():.3f} MW to "
            f"{n.lines.s_nom.max():.3f} MW"
        )

    if len(n.transformers) > 0:
        print(
            f"Transformer ratings: "
            f"{n.transformers.s_nom.min():.3f} MW to "
            f"{n.transformers.s_nom.max():.3f} MW"
        )

    # ------------------------------------------------------------
    # 8. Network connectivity
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("7. NETWORK CONNECTIVITY")
    print("-" * 70)

    try:
        components = list(n.graph().connected_components())

        print(f"Connected components: {len(components)}")

        if len(components) == 1:
            print("OK: Entire network is electrically connected.")
        else:
            print("WARNING: Network contains multiple disconnected components.")

            component_sizes = sorted(
                [len(c) for c in components],
                reverse=True
            )

            print("Component sizes:")
            print(component_sizes)

    except Exception as e:
        print("WARNING: Could not perform graph connectivity check.")
        print(e)

    # ------------------------------------------------------------
    # 9. Power-flow validation
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("8. POWER-FLOW VALIDATION")
    print("-" * 70)

    try:
        n.set_snapshots(["now"])

        print("Running linear power flow...")

        result = n.lpf()

        print("OK: Linear power flow completed.")

        # Line loading
        if len(n.lines) > 0 and "s_nom" in n.lines.columns:
            line_loading = (
                n.lines_t.p0.loc["now"].abs()
                / n.lines["s_nom"]
                * 100
            )

            max_line_loading = line_loading.max()

            print(
                f"Maximum line loading: "
                f"{max_line_loading:.2f}%"
            )

            if max_line_loading <= 100:
                print("OK: No line exceeds its thermal limit.")
            else:
                print("WARNING: At least one line exceeds its thermal limit.")

        # Transformer loading
        if len(n.transformers) > 0:
            trafo_loading = (
                n.transformers_t.p0.loc["now"].abs()
                / n.transformers["s_nom"]
                * 100
            )

            max_trafo_loading = trafo_loading.max()

            print(
                f"Maximum transformer loading: "
                f"{max_trafo_loading:.2f}%"
            )

            if max_trafo_loading <= 100:
                print("OK: No transformer exceeds its thermal limit.")
            else:
                print(
                    "WARNING: At least one transformer exceeds "
                    "its thermal limit."
                )

    except Exception as e:
        print("WARNING: Linear power flow could not be completed.")
        print(f"Reason: {e}")

    # ------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                 VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Network file : {NETWORK_FILE}")
    print(f"Buses        : {len(n.buses)}")
    print(f"Lines        : {len(n.lines)}")
    print(f"Transformers : {len(n.transformers)}")

    print("\nValidation completed.")
    print(
        "\nIMPORTANT:"
        "\nThis script validates the structure and basic electrical"
        "\nconsistency of the current PyPSA network."
        "\nIt does NOT prove that the network represents the real Irish"
        "\ngrid accurately. Real-world validation requires authoritative"
        "\ngrid data and comparison against known system behaviour."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()