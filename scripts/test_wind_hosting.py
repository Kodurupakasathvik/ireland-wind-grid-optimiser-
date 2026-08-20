import pypsa
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent.parent

NETWORK_FILE = BASE / "data" / "processed" / "ireland_network.nc"
CANDIDATES_FILE = BASE / "data" / "processed" / "wind_connection_candidates.csv"
OUTPUT_FILE = BASE / "data" / "processed" / "wind_hosting_capacity.csv"

# Known 220 kV sink bus from the validated 500 MW power-flow test.
PREFERRED_SINK_BUS = "way/1003262502-220"

# Test wind capacity in steps.
STEP_MW = 100

# Maximum wind capacity tested at each bus.
MAX_TEST_MW = 2000

# Thermal loading limit.
# 100% means nominal capacity is the limiting threshold.
THERMAL_LIMIT_PERCENT = 100.0


# ============================================================
# HELPER: SELECT A SINK BUS
# ============================================================

def select_sink_bus(network, candidate_bus):
    """
    Select a fixed load/sink bus.

    Prefer the bus used in the previously validated 500 MW
    generator-to-load power-flow test.

    If the candidate is the same bus, choose another 220 kV bus.
    """

    # First choice: previously validated sink.
    if (
        PREFERRED_SINK_BUS in network.buses.index
        and PREFERRED_SINK_BUS != candidate_bus
    ):
        return PREFERRED_SINK_BUS

    # Otherwise choose a different 220 kV bus.
    voltage_buses = network.buses[
        network.buses["v_nom"] == 220.0
    ].index.tolist()

    for bus in voltage_buses:
        if bus != candidate_bus:
            return bus

    # Final fallback.
    for bus in network.buses.index:
        if bus != candidate_bus:
            return bus

    return None


# ============================================================
# HELPER: CALCULATE MAX LINE LOADING
# ============================================================

def calculate_line_loading(network):
    """
    Return:
        maximum line loading percentage
        limiting line name
    """

    if len(network.lines) == 0:
        return 0.0, None

    if len(network.lines_t.p0) == 0:
        return 0.0, None

    snapshot = network.snapshots[0]

    p0 = network.lines_t.p0.loc[snapshot].abs()

    capacities = network.lines["s_nom"].replace(0, float("nan"))

    loading = (p0 / capacities) * 100.0
    loading = loading.replace([float("inf"), -float("inf")], float("nan"))
    loading = loading.dropna()

    if loading.empty:
        return 0.0, None

    max_value = float(loading.max())
    max_element = loading.idxmax()

    return max_value, max_element


# ============================================================
# HELPER: CALCULATE MAX TRANSFORMER LOADING
# ============================================================

def calculate_transformer_loading(network):
    """
    Return:
        maximum transformer loading percentage
        limiting transformer name
    """

    if len(network.transformers) == 0:
        return 0.0, None

    if len(network.transformers_t.p0) == 0:
        return 0.0, None

    snapshot = network.snapshots[0]

    p0 = network.transformers_t.p0.loc[snapshot].abs()

    capacities = network.transformers["s_nom"].replace(
        0, float("nan")
    )

    loading = (p0 / capacities) * 100.0
    loading = loading.replace(
        [float("inf"), -float("inf")],
        float("nan")
    )
    loading = loading.dropna()

    if loading.empty:
        return 0.0, None

    max_value = float(loading.max())
    max_element = loading.idxmax()

    return max_value, max_element


# ============================================================
# HELPER: RUN ONE WIND TEST
# ============================================================

def run_wind_test(candidate_bus, sink_bus, wind_mw):
    """
    Build a fresh copy of the network.

    Wind generator:
        candidate_bus -> +wind_mw

    Sink load:
        sink_bus -> -wind_mw

    A slack generator is placed at the sink bus so PyPSA has
    a valid reference/slack bus for the linear power flow.

    The wind generator itself is PQ-controlled so that its
    output is explicitly fixed to wind_mw.
    """

    network = pypsa.Network(NETWORK_FILE)

    snapshot = "now"

    network.set_snapshots([snapshot])

    # --------------------------------------------------------
    # Remove any pre-existing generators/loads from the test
    # network if present.
    #
    # The network currently appears to contain no operational
    # generation/load model, but this makes the test explicit
    # and reproducible.
    # --------------------------------------------------------

    if len(network.generators) > 0:
        network.remove("Generator", network.generators.index)

    if len(network.loads) > 0:
        network.remove("Load", network.loads.index)

    # --------------------------------------------------------
    # WIND GENERATOR
    # --------------------------------------------------------

    wind_name = "TEST_WIND"

    network.add(
        "Generator",
        wind_name,
        bus=candidate_bus,
        carrier="wind",
        p_nom=float(wind_mw),
        p_set=float(wind_mw),
        control="PQ",
    )

    # --------------------------------------------------------
    # FIXED SINK LOAD
    # --------------------------------------------------------

    sink_name = "TEST_SINK"

    network.add(
        "Load",
        sink_name,
        bus=sink_bus,
        p_set=float(wind_mw),
    )

    # --------------------------------------------------------
    # SLACK GENERATOR
    #
    # It is placed at the sink bus and starts at zero.
    # The purpose is to give the AC network a valid reference
    # bus for the linear load-flow calculation.
    # --------------------------------------------------------

    slack_name = "TEST_SLACK"

    network.add(
        "Generator",
        slack_name,
        bus=sink_bus,
        carrier="slack",
        control="Slack",
        p_nom=10000.0,
        p_set=0.0,
    )

    # --------------------------------------------------------
    # RUN LINEAR POWER FLOW
    # --------------------------------------------------------

    try:
        network.lpf()

    except Exception as exc:
        return {
            "success": False,
            "line_loading": None,
            "line_element": None,
            "transformer_loading": None,
            "transformer_element": None,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # CALCULATE LOADINGS
    # --------------------------------------------------------

    try:
        max_line_loading, limiting_line = calculate_line_loading(
            network
        )

        max_transformer_loading, limiting_transformer = (
            calculate_transformer_loading(network)
        )

    except Exception as exc:
        return {
            "success": False,
            "line_loading": None,
            "line_element": None,
            "transformer_loading": None,
            "transformer_element": None,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # VERIFY THAT THE WIND ACTUALLY INJECTED POWER
    # --------------------------------------------------------

    try:
        actual_wind_output = float(
            network.generators_t.p.loc[snapshot, wind_name]
        )

    except Exception:
        actual_wind_output = None

    # --------------------------------------------------------
    # VERIFY THAT THE SINK ACTUALLY CONSUMED POWER
    # --------------------------------------------------------

    try:
        actual_sink_load = float(
            network.loads_t.p.loc[snapshot, sink_name]
        )

    except Exception:
        actual_sink_load = None

    # --------------------------------------------------------
    # IMPORTANT VALIDATION
    #
    # If the wind output is effectively zero, the test is not
    # meaningful.
    # --------------------------------------------------------

    if actual_wind_output is None:
        return {
            "success": False,
            "line_loading": max_line_loading,
            "line_element": limiting_line,
            "transformer_loading": max_transformer_loading,
            "transformer_element": limiting_transformer,
            "error": "Wind generator output could not be read.",
        }

    if abs(actual_wind_output - wind_mw) > 1e-3:
        return {
            "success": False,
            "line_loading": max_line_loading,
            "line_element": limiting_line,
            "transformer_loading": max_transformer_loading,
            "transformer_element": limiting_transformer,
            "error": (
                f"Wind output mismatch: requested {wind_mw} MW, "
                f"actual {actual_wind_output:.3f} MW."
            ),
        }

    # --------------------------------------------------------
    # RETURN VALID RESULT
    # --------------------------------------------------------

    return {
        "success": True,
        "line_loading": max_line_loading,
        "line_element": limiting_line,
        "transformer_loading": max_transformer_loading,
        "transformer_element": limiting_transformer,
        "wind_output": actual_wind_output,
        "sink_load": actual_sink_load,
        "error": None,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("=== WIND HOSTING CAPACITY TEST ===")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD CANDIDATES
    # --------------------------------------------------------

    if not NETWORK_FILE.exists():
        print(f"ERROR: Network file not found:")
        print(NETWORK_FILE)
        return

    if not CANDIDATES_FILE.exists():
        print(f"ERROR: Candidate file not found:")
        print(CANDIDATES_FILE)
        return

    candidates = pd.read_csv(CANDIDATES_FILE)

    if "bus" not in candidates.columns:
        print("ERROR: Candidate CSV does not contain a 'bus' column.")
        return

    # Remove duplicates while preserving order.
    candidate_buses = (
        candidates["bus"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # LOAD NETWORK ONCE FOR VALIDATION
    # --------------------------------------------------------

    base_network = pypsa.Network(NETWORK_FILE)

    print("Network:")
    print(f"  Buses:        {len(base_network.buses)}")
    print(f"  Lines:        {len(base_network.lines)}")
    print(f"  Transformers: {len(base_network.transformers)}")
    print()

    # --------------------------------------------------------
    # VALIDATE CANDIDATES
    # --------------------------------------------------------

    candidate_buses = [
        bus
        for bus in candidate_buses
        if bus in base_network.buses.index
    ]

    print(f"Valid candidate buses: {len(candidate_buses)}")
    print()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    total = len(candidate_buses)

    # --------------------------------------------------------
    # TEST EACH CANDIDATE
    # --------------------------------------------------------

    for rank, candidate_bus in enumerate(candidate_buses, start=1):

        print()
        print("-" * 70)
        print(f"[{rank}/{total}] Testing: {candidate_bus}")
        print("-" * 70)

        # Select fixed sink.
        sink_bus = select_sink_bus(
            base_network,
            candidate_bus
        )

        if sink_bus is None:
            print("  ERROR: Could not find a valid sink bus.")

            results.append({
                "bus": candidate_bus,
                "hosting_capacity_mw": 0,
                "first_failed_mw": 100,
                "max_line_loading_percent": 0.0,
                "max_transformer_loading_percent": 0.0,
                "limiting_element": "NO_SINK",
                "limiting_type": "ERROR",
            })

            continue

        print(f"  Wind bus: {candidate_bus}")
        print(f"  Sink bus: {sink_bus}")
        print()

        hosting_capacity = 0
        first_failed_mw = None

        maximum_line_loading_seen = 0.0
        maximum_transformer_loading_seen = 0.0

        limiting_element = None
        limiting_type = None

        # ----------------------------------------------------
        # TEST 100, 200, 300 ... MW
        # ----------------------------------------------------

        for wind_mw in range(
            STEP_MW,
            MAX_TEST_MW + STEP_MW,
            STEP_MW
        ):

            print(
                f"  Testing {wind_mw} MW... ",
                end="",
                flush=True
            )

            result = run_wind_test(
                candidate_bus,
                sink_bus,
                wind_mw
            )

            if not result["success"]:

                print(
                    f"FAILED: {result['error']}"
                )

                first_failed_mw = wind_mw

                # If the test failed because the electrical
                # model could not solve, record it explicitly.
                limiting_element = (
                    result.get("line_element")
                    or result.get("transformer_element")
                    or "POWER_FLOW"
                )

                limiting_type = "POWER_FLOW_FAILURE"

                break

            line_loading = float(
                result["line_loading"]
            )

            transformer_loading = float(
                result["transformer_loading"]
            )

            maximum_line_loading_seen = max(
                maximum_line_loading_seen,
                line_loading
            )

            maximum_transformer_loading_seen = max(
                maximum_transformer_loading_seen,
                transformer_loading
            )

            # ------------------------------------------------
            # DETERMINE LIMITING ELEMENT
            # ------------------------------------------------

            if (
                line_loading >= transformer_loading
                and line_loading >= THERMAL_LIMIT_PERCENT
            ):

                first_failed_mw = wind_mw
                limiting_element = result["line_element"]
                limiting_type = "LINE"

                print(
                    f"LIMIT "
                    f"(line {line_loading:.2f}%, "
                    f"trafo {transformer_loading:.2f}%)"
                )

                break

            if (
                transformer_loading > line_loading
                and transformer_loading >= THERMAL_LIMIT_PERCENT
            ):

                first_failed_mw = wind_mw
                limiting_element = result["transformer_element"]
                limiting_type = "TRANSFORMER"

                print(
                    f"LIMIT "
                    f"(line {line_loading:.2f}%, "
                    f"trafo {transformer_loading:.2f}%)"
                )

                break

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            hosting_capacity = wind_mw

            print(
                f"OK "
                f"(line {line_loading:.2f}%, "
                f"trafo {transformer_loading:.2f}%)"
            )

        # ----------------------------------------------------
        # RESULT FOR THIS BUS
        # ----------------------------------------------------

        if first_failed_mw is None:
            limiting_element = None
            limiting_type = None

        print()
        print(f"  RESULT: {hosting_capacity} MW")

        results.append({
            "bus": candidate_bus,
            "hosting_capacity_mw": hosting_capacity,
            "first_failed_mw": first_failed_mw,
            "max_line_loading_percent": round(
                maximum_line_loading_seen,
                4
            ),
            "max_transformer_loading_percent": round(
                maximum_transformer_loading_seen,
                4
            ),
            "limiting_element": limiting_element,
            "limiting_type": limiting_type,
        })

    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print()
        print("ERROR: No valid results.")
        return

    # --------------------------------------------------------
    # RANK BY HOSTING CAPACITY
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by=[
            "hosting_capacity_mw",
            "max_line_loading_percent",
            "max_transformer_loading_percent",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("=== WIND HOSTING CAPACITY RESULTS ===")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print()
    print()
    print("Saved:")
    print(OUTPUT_FILE)

    # ========================================================
    # SUMMARY
    # ========================================================

    best = results_df.iloc[0]

    print()
    print("=" * 70)
    print("=== SUMMARY ===")
    print("=" * 70)

    print(
        f"Highest hosting capacity: "
        f"{best['hosting_capacity_mw']} MW"
    )

    print(
        f"Best candidate: "
        f"{best['bus']}"
    )

    if pd.isna(best["limiting_type"]):
        print(
            "Limiting type: "
            "None within tested range"
        )
    else:
        print(
            f"Limiting type: "
            f"{best['limiting_type']}"
        )

    if pd.notna(best["first_failed_mw"]):
        print(
            f"First failed level: "
            f"{best['first_failed_mw']} MW"
        )
    else:
        print(
            f"No failure up to "
            f"{MAX_TEST_MW} MW"
        )

    print()
    print(
        "IMPORTANT: Hosting capacity is the highest tested "
        "wind injection that remains below the thermal limit."
    )
    print(
        "A result equal to the maximum test level means the "
        "actual capacity may be higher and was not searched."
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()