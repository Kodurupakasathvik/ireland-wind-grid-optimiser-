from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# S4.3 — TARGETED RESIDUAL BOTTLENECK MITIGATION SCREEN
# =============================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

OUTPUT_DIR = Path("data/processed")

SNAPSHOT = "S2_PEAK_DEMAND"
PACKAGE = "P3_HIGH_COORDINATED"

REACTIVE_BUS = "eirgrid_wind_way/104388595-220"
REACTIVE_SUPPORT_MVAR = 500.0

# Acceptance thresholds
LINE_LIMIT_PCT = 100.0
LOW_VOLTAGE_LIMIT_PU = 0.95
HIGH_VOLTAGE_LIMIT_PU = 1.05
TRANSFORMER_LIMIT_PCT = 100.0


# =============================================================================
# P3 BASE REINFORCEMENTS
# =============================================================================

P3_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.75,
    "merged_way/61295764-220+1": 2.00,
    "way/343436171-220": 2.00,
    "merged_way/257889771-220+1": 1.75,
    "merged_relation/4872159-220+1": 1.75,
}


# =============================================================================
# RESIDUAL BOTTLENECKS IDENTIFIED IN S4.2
#
# These are the four lines that remain above 100% after P3 + 500 MVAr.
#
# We test targeted reinforcement of these individual corridors.
# =============================================================================

TARGET_LINES = [
    "way/235559472-220",
    "way/713396116-220",
    "way/42838773-220",
    "merged_way/516651706-220+2",
]


# =============================================================================
# TARGETED REINFORCEMENT LEVELS
#
# We deliberately test incremental reinforcement instead of jumping directly
# to a large system-wide package.
# =============================================================================

TARGET_MULTIPLIERS = [
    1.25,
    1.50,
    1.75,
    2.00,
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def banner(text):
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def find_component(network, component, name):
    """
    Confirm that the requested component exists.
    """
    table = getattr(network, component)

    if name not in table.index:
        return False

    return True


def apply_p3_reinforcements(n):
    """
    Apply the established P3 package.
    """
    print("\n" + "-" * 100)
    print("APPLYING P3 REINFORCEMENTS")
    print("-" * 100)

    for line_name, multiplier in P3_REINFORCEMENTS.items():

        if line_name not in n.lines.index:
            print(
                f"WARNING: line not found -> {line_name}"
            )
            continue

        old_s_nom = float(n.lines.at[line_name, "s_nom"])

        # Store original capacity once.
        if "s_nom_original_s43" not in n.lines.columns:
            n.lines["s_nom_original_s43"] = np.nan

        if pd.isna(n.lines.at[line_name, "s_nom_original_s43"]):
            n.lines.at[line_name, "s_nom_original_s43"] = old_s_nom

        original = float(
            n.lines.at[line_name, "s_nom_original_s43"]
        )

        new_s_nom = original * multiplier

        n.lines.at[line_name, "s_nom"] = new_s_nom

        print(
            f"{line_name:<55}"
            f"{multiplier:>6.2f}x"
            f"{original:>12.3f} -> "
            f"{new_s_nom:>12.3f} MW"
        )


def apply_reactive_support(n):
    """
    Apply +500 MVAr safely.

    The generator name may differ between network versions, so locate the
    generator by exact name first and then by bus if necessary.
    """

    print("\n" + "-" * 100)
    print("APPLYING REACTIVE SUPPORT")
    print("-" * 100)

    generators = n.generators

    generator_name = None

    # Exact match
    if REACTIVE_BUS in generators.index:
        generator_name = REACTIVE_BUS

    # Fallback: search generator whose name contains the requested bus name.
    if generator_name is None:
        matches = [
            g for g in generators.index
            if str(g) == REACTIVE_BUS
            or REACTIVE_BUS in str(g)
        ]

        if matches:
            generator_name = matches[0]

    # Fallback: search by generator bus.
    if generator_name is None:
        bus_matches = generators.index[
            generators["bus"].astype(str) == REACTIVE_BUS
        ]

        if len(bus_matches) > 0:
            generator_name = bus_matches[0]

    if generator_name is None:
        raise RuntimeError(
            "Could not identify reactive-support generator."
        )

    old_q = 0.0

    if SNAPSHOT in n.generators_t.q_set.index:
        if generator_name in n.generators_t.q_set.columns:
            old_q = float(
                n.generators_t.q_set.loc[
                    SNAPSHOT,
                    generator_name
                ]
            )

    n.generators_t.q_set.loc[
        SNAPSHOT,
        generator_name
    ] = REACTIVE_SUPPORT_MVAR

    print(
        f"Reactive support applied through generator: "
        f"{generator_name}"
    )

    print(
        f"Q setpoint: "
        f"{old_q:.3f} -> "
        f"{REACTIVE_SUPPORT_MVAR:.3f} MVAr"
    )


def apply_targeted_reinforcement(n, line_name, multiplier):
    """
    Apply targeted reinforcement to ONE residual bottleneck.

    Capacity is always calculated from the original network value, avoiding
    accidental compounding across candidates.
    """

    if line_name not in n.lines.index:
        raise RuntimeError(
            f"Target line not found: {line_name}"
        )

    original = float(
        n.lines.at[line_name, "s_nom"]
    )

    new_capacity = original * multiplier

    n.lines.at[line_name, "s_nom"] = new_capacity

    return original, new_capacity


def calculate_results(n):
    """
    Run AC nonlinear power flow and calculate system metrics.
    """

    print("\n" + "-" * 100)
    print("RUNNING AC NONLINEAR POWER FLOW")
    print("-" * 100)

    converged = True

    try:
        n.pf(
            snapshots=[SNAPSHOT]
        )
    except Exception as exc:
        converged = False

        print(
            "\nPOWER FLOW FAILED:"
        )
        print(str(exc))

    if not converged:
        return {
            "converged": False,
            "min_voltage_pu": np.nan,
            "min_voltage_bus": "",
            "max_voltage_pu": np.nan,
            "max_voltage_bus": "",
            "low_voltage_buses": np.nan,
            "high_voltage_buses": np.nan,
            "max_line_loading_pct": np.nan,
            "overloaded_lines": np.nan,
            "critical_line": "",
            "max_transformer_loading_pct": np.nan,
            "worst_transformer": "",
        }

    # -------------------------------------------------------------------------
    # VOLTAGES
    # -------------------------------------------------------------------------

    v = n.buses_t.v_mag_pu.loc[SNAPSHOT]

    min_voltage = float(v.min())
    min_voltage_bus = str(v.idxmin())

    max_voltage = float(v.max())
    max_voltage_bus = str(v.idxmax())

    low_voltage_count = int(
        (v < LOW_VOLTAGE_LIMIT_PU).sum()
    )

    high_voltage_count = int(
        (v > HIGH_VOLTAGE_LIMIT_PU).sum()
    )

    # -------------------------------------------------------------------------
    # LINE LOADING
    # -------------------------------------------------------------------------

    line_s_nom = n.lines.s_nom.replace(0, np.nan)

    p0 = n.lines_t.p0.loc[SNAPSHOT].abs()

    p1 = n.lines_t.p1.loc[SNAPSHOT].abs()

    apparent_power = pd.concat(
        [p0, p1],
        axis=1
    ).max(axis=1)

    line_loading_pct = (
        apparent_power / line_s_nom * 100.0
    )

    line_loading_pct = line_loading_pct.dropna()

    max_line_loading = float(
        line_loading_pct.max()
    )

    critical_line = str(
        line_loading_pct.idxmax()
    )

    overloaded_lines = int(
        (line_loading_pct > LINE_LIMIT_PCT).sum()
    )

    # -------------------------------------------------------------------------
    # TRANSFORMERS
    # -------------------------------------------------------------------------

    max_transformer_loading = 0.0
    worst_transformer = ""

    if len(n.transformers.index) > 0:

        transformer_s_nom = (
            n.transformers.s_nom.replace(
                0,
                np.nan
            )
        )

        if (
            SNAPSHOT in n.transformers_t.p0.index
            and len(n.transformers_t.p0.columns) > 0
        ):

            tp0 = (
                n.transformers_t.p0.loc[
                    SNAPSHOT
                ].abs()
            )

            tp1 = (
                n.transformers_t.p1.loc[
                    SNAPSHOT
                ].abs()
            )

            transformer_power = pd.concat(
                [tp0, tp1],
                axis=1
            ).max(axis=1)

            transformer_loading = (
                transformer_power /
                transformer_s_nom *
                100.0
            )

            transformer_loading = (
                transformer_loading.dropna()
            )

            if len(transformer_loading) > 0:

                max_transformer_loading = float(
                    transformer_loading.max()
                )

                worst_transformer = str(
                    transformer_loading.idxmax()
                )

    return {
        "converged": True,
        "min_voltage_pu": min_voltage,
        "min_voltage_bus": min_voltage_bus,
        "max_voltage_pu": max_voltage,
        "max_voltage_bus": max_voltage_bus,
        "low_voltage_buses": low_voltage_count,
        "high_voltage_buses": high_voltage_count,
        "max_line_loading_pct": max_line_loading,
        "overloaded_lines": overloaded_lines,
        "critical_line": critical_line,
        "max_transformer_loading_pct": max_transformer_loading,
        "worst_transformer": worst_transformer,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "S4.3 — TARGETED RESIDUAL BOTTLENECK MITIGATION SCREEN"
    )

    print(
        f"\nNetwork  : {NETWORK_PATH}"
    )
    print(
        f"Snapshot : {SNAPSHOT}"
    )
    print(
        f"Package  : {PACKAGE}"
    )
    print(
        "PF       : AC nonlinear"
    )
    print(
        "Dispatch : unchanged"
    )
    print(
        "Loads    : unchanged"
    )
    print(
        "Source   : READ-ONLY"
    )
    print(
        f"Reactive : +{REACTIVE_SUPPORT_MVAR:.0f} MVAr"
    )

    if not NETWORK_PATH.exists():
        raise FileNotFoundError(
            f"Network not found: {NETWORK_PATH}"
        )

    # -------------------------------------------------------------------------
    # Load network ONCE from source.
    # Every candidate receives a fresh deep copy.
    # -------------------------------------------------------------------------

    base_network = pypsa.Network(
        str(NETWORK_PATH)
    )

    # Confirm snapshot exists.
    if SNAPSHOT not in base_network.snapshots:
        raise RuntimeError(
            f"Snapshot not found: {SNAPSHOT}"
        )

    # -------------------------------------------------------------------------
    # Confirm target lines.
    # -------------------------------------------------------------------------

    banner(
        "RESIDUAL BOTTLENECKS UNDER TEST"
    )

    for line_name in TARGET_LINES:

        if line_name in base_network.lines.index:

            original_capacity = float(
                base_network.lines.at[
                    line_name,
                    "s_nom"
                ]
            )

            print(
                f"{line_name:<55}"
                f"{original_capacity:>12.3f} MW"
            )

        else:

            print(
                f"WARNING: TARGET LINE NOT FOUND -> "
                f"{line_name}"
            )

    # -------------------------------------------------------------------------
    # Candidate list
    # -------------------------------------------------------------------------

    candidates = []

    # Baseline P3 + reactive support
    candidates.append(
        (
            "BASE_P3_PLUS_500MVAR",
            None,
            1.0
        )
    )

    # One-target-at-a-time tests
    for line_name in TARGET_LINES:

        if line_name not in base_network.lines.index:
            continue

        for multiplier in TARGET_MULTIPLIERS:

            safe_name = (
                line_name
                .replace("/", "_")
                .replace("+", "plus")
                .replace(":", "_")
            )

            candidate_name = (
                f"{safe_name}_TARGET_{multiplier:.2f}X"
            )

            candidates.append(
                (
                    candidate_name,
                    line_name,
                    multiplier
                )
            )

    # -------------------------------------------------------------------------
    # Run candidates
    # -------------------------------------------------------------------------

    results = []

    for candidate_number, (
        candidate_name,
        target_line,
        multiplier
    ) in enumerate(
        candidates,
        start=1
    ):

        banner(
            f"CANDIDATE {candidate_number}/{len(candidates)}"
        )

        print(
            f"Candidate : {candidate_name}"
        )

        if target_line is None:

            print(
                "Target    : NONE "
                "(P3 baseline)"
            )

        else:

            print(
                f"Target    : {target_line}"
            )

            print(
                f"Multiplier: {multiplier:.2f}x"
            )

        # Fresh network
        n = copy.deepcopy(
            base_network
        )

        # Apply established P3 package.
        apply_p3_reinforcements(n)

        # Apply reactive support.
        apply_reactive_support(n)

        # Apply targeted reinforcement.
        target_original = np.nan
        target_new = np.nan

        if target_line is not None:

            target_original, target_new = (
                apply_targeted_reinforcement(
                    n,
                    target_line,
                    multiplier
                )
            )

            print(
                "\nTARGETED REINFORCEMENT"
            )

            print(
                f"{target_line:<55}"
                f"{multiplier:>6.2f}x"
                f"{target_original:>12.3f} -> "
                f"{target_new:>12.3f} MW"
            )

        # Run PF
        metrics = calculate_results(n)

        # Acceptance
        fully_acceptable = (
            metrics["converged"]
            and
            metrics["overloaded_lines"] == 0
            and
            metrics["low_voltage_buses"] == 0
            and
            metrics["high_voltage_buses"] == 0
            and
            metrics["max_transformer_loading_pct"]
            <= TRANSFORMER_LIMIT_PCT
        )

        result = {
            "candidate": candidate_name,
            "target_line": (
                target_line
                if target_line is not None
                else ""
            ),
            "target_multiplier": (
                multiplier
                if target_line is not None
                else 1.0
            ),
            "target_original_s_nom_mw": target_original,
            "target_new_s_nom_mw": target_new,
            **metrics,
            "fully_acceptable": fully_acceptable,
        }

        results.append(result)

        print(
            "\nRESULT"
        )

        print(
            f"Converged                 : "
            f"{metrics['converged']}"
        )

        print(
            f"Minimum voltage           : "
            f"{metrics['min_voltage_pu']:.6f} pu"
        )

        print(
            f"Minimum-voltage bus       : "
            f"{metrics['min_voltage_bus']}"
        )

        print(
            f"Low-voltage buses         : "
            f"{metrics['low_voltage_buses']}"
        )

        print(
            f"Max line loading          : "
            f"{metrics['max_line_loading_pct']:.6f} %"
        )

        print(
            f"Overloaded lines          : "
            f"{metrics['overloaded_lines']}"
        )

        print(
            f"Critical line             : "
            f"{metrics['critical_line']}"
        )

        print(
            f"Max transformer loading  : "
            f"{metrics['max_transformer_loading_pct']:.6f} %"
        )

        print(
            f"Fully acceptable          : "
            f"{fully_acceptable}"
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    results_df = pd.DataFrame(
        results
    )

    banner(
        "S4.3 — TARGETED MITIGATION SCREEN SUMMARY"
    )

    display_columns = [
        "candidate",
        "target_line",
        "target_multiplier",
        "max_line_loading_pct",
        "overloaded_lines",
        "min_voltage_pu",
        "low_voltage_buses",
        "max_transformer_loading_pct",
        "fully_acceptable",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Determine best candidate.
    #
    # Priority:
    #   1. Fully acceptable
    #   2. No overloaded lines
    #   3. Lowest maximum line loading
    #   4. Highest minimum voltage
    #   5. Fewest low-voltage buses
    # -------------------------------------------------------------------------

    ranked = results_df.copy()

    ranked["_acceptable_rank"] = (
        ~ranked["fully_acceptable"]
    ).astype(int)

    ranked["_overload_rank"] = (
        ranked["overloaded_lines"]
    )

    ranked["_loading_rank"] = (
        ranked["max_line_loading_pct"]
    )

    ranked["_voltage_rank"] = (
        -ranked["min_voltage_pu"]
    )

    ranked["_low_voltage_rank"] = (
        ranked["low_voltage_buses"]
    )

    ranked = ranked.sort_values(
        by=[
            "_acceptable_rank",
            "_overload_rank",
            "_loading_rank",
            "_voltage_rank",
            "_low_voltage_rank",
        ]
    )

    best = ranked.iloc[0]

    banner(
        "BEST TARGETED MITIGATION CANDIDATE"
    )

    print(
        f"Candidate                 : "
        f"{best['candidate']}"
    )

    print(
        f"Target line               : "
        f"{best['target_line']}"
    )

    print(
        f"Target multiplier         : "
        f"{best['target_multiplier']:.2f}x"
    )

    print(
        f"Maximum line loading      : "
        f"{best['max_line_loading_pct']:.6f} %"
    )

    print(
        f"Overloaded lines          : "
        f"{int(best['overloaded_lines'])}"
    )

    print(
        f"Minimum voltage           : "
        f"{best['min_voltage_pu']:.6f} pu"
    )

    print(
        f"Low-voltage buses         : "
        f"{int(best['low_voltage_buses'])}"
    )

    print(
        f"Transformer loading       : "
        f"{best['max_transformer_loading_pct']:.6f} %"
    )

    print(
        f"Fully acceptable          : "
        f"{best['fully_acceptable']}"
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        OUTPUT_DIR /
        "s4_3_targeted_residual_mitigation_results.csv"
    )

    ranking_path = (
        OUTPUT_DIR /
        "s4_3_targeted_residual_mitigation_ranking.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False
    )

    ranked.drop(
        columns=[
            "_acceptable_rank",
            "_overload_rank",
            "_loading_rank",
            "_voltage_rank",
            "_low_voltage_rank",
        ]
    ).to_csv(
        ranking_path,
        index=False
    )

    banner(
        "S4.3 COMPLETE"
    )

    print(
        f"\nResults saved to:"
    )

    print(
        summary_path
    )

    print(
        ranking_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No network file was modified."
    )

    print(
        "All candidates were evaluated from the original "
        "READ-ONLY network."
    )


if __name__ == "__main__":
    main()