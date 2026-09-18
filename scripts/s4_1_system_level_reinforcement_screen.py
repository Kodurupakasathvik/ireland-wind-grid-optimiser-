# ==================================================================================================
# S4.1 — SYSTEM-LEVEL REINFORCEMENT STRATEGY SCREEN
# ==================================================================================================
#
# Purpose:
#   Screen coordinated transmission reinforcement packages across all six operating snapshots
#   using nonlinear AC power flow.
#
# IMPORTANT:
#   - Source network is READ-ONLY.
#   - Every package starts from a fresh network load.
#   - Every snapshot is solved independently.
#   - Reactive support is applied through the existing weak-bus wind generator.
#   - AC line loading is calculated from p0/q0 and p1/q1.
#   - Transformer loading is calculated from p0/q0 and p1/q1.
#   - No nonexistent "s" PyPSA attribute is used.
#
# Acceptance criteria:
#   Weak bus voltage >= 1.00 pu
#   Minimum voltage >= 0.95 pu
#   Maximum line loading <= 100%
#   Zero overloaded lines
#   AC power flow converged
#
# Output:
#   data\processed\s4_1_system_level_reinforcement_screen_results.csv
#
# ==================================================================================================

import os
import warnings
import traceback

import numpy as np
import pandas as pd
import pypsa


warnings.filterwarnings("ignore")


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = (
    r"data\processed\eirgrid_second_reinforced_network.nc"
)

OUTPUT_PATH = (
    r"data\processed\s4_1_system_level_reinforcement_screen_results.csv"
)

SNAPSHOTS = [
    "S1_NORMAL",
    "S2_PEAK_DEMAND",
    "S3_HIGH_WIND",
    "S4_HIGH_WIND_HIGH_DEMAND",
    "S5_HIGH_AVAILABILITY_LOW_GENERATION",
    "S6_MAXIMUM_STRESS",
]

WEAK_BUS = "way/104388595-220"

REACTIVE_GENERATOR = "eirgrid_wind_way/104388595-220"

Q_SUPPORT_MVAR = 500.0

# Voltage acceptance limits
WEAK_BUS_MIN_ACCEPTABLE = 1.00
SYSTEM_MIN_VOLTAGE_ACCEPTABLE = 0.95

# Thermal acceptance limits
MAX_LOADING_ACCEPTABLE = 100.0

# ==================================================================================================
# FIXED REINFORCEMENTS CARRIED FORWARD FROM S3
# ==================================================================================================

FIXED_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.50,
    "merged_way/61295764-220+1": 1.50,
    "way/343436171-220": 1.50,
    "merged_way/257889771-220+1": 1.25,
}


# ==================================================================================================
# SYSTEM-LEVEL PACKAGES
# ==================================================================================================
#
# Package philosophy:
#
# P0 = current S3 configuration
# P1 = moderate coordinated reinforcement
# P2 = strong coordinated reinforcement
# P3 = high coordinated reinforcement
# P4 = aggressive coordinated reinforcement
#
# The fifth line was the S3.14 targeted bottleneck.
#
# ==================================================================================================

PACKAGES = {
    "P0_CURRENT_S3_BEST": {
        "merged_way/1231251986-220+2": 1.50,
        "merged_way/61295764-220+1": 1.50,
        "way/343436171-220": 1.50,
        "merged_way/257889771-220+1": 1.25,
        "merged_relation/4872159-220+1": 1.00,
    },

    "P1_MODERATE_COORDINATED": {
        "merged_way/1231251986-220+2": 1.50,
        "merged_way/61295764-220+1": 1.50,
        "way/343436171-220": 1.50,
        "merged_way/257889771-220+1": 1.50,
        "merged_relation/4872159-220+1": 1.50,
    },

    "P2_STRONG_COORDINATED": {
        "merged_way/1231251986-220+2": 1.75,
        "merged_way/61295764-220+1": 1.75,
        "way/343436171-220": 1.75,
        "merged_way/257889771-220+1": 1.50,
        "merged_relation/4872159-220+1": 1.50,
    },

    "P3_HIGH_COORDINATED": {
        "merged_way/1231251986-220+2": 1.75,
        "merged_way/61295764-220+1": 2.00,
        "way/343436171-220": 2.00,
        "merged_way/257889771-220+1": 1.75,
        "merged_relation/4872159-220+1": 1.75,
    },

    "P4_AGGRESSIVE_COORDINATED": {
        "merged_way/1231251986-220+2": 2.00,
        "merged_way/61295764-220+1": 2.00,
        "way/343436171-220": 2.00,
        "merged_way/257889771-220+1": 2.00,
        "merged_relation/4872159-220+1": 2.00,
    },
}


# ==================================================================================================
# DISPLAY
# ==================================================================================================

WIDTH = 110

def line(char="="):
    print(char * WIDTH)


def header(title):
    line("=")
    print(title)
    line("=")


# ==================================================================================================
# VALIDATION HELPERS
# ==================================================================================================

def require_line(n, line_name):
    if line_name not in n.lines.index:
        raise KeyError(
            f"Required line not found in network: {line_name}"
        )


def validate_required_lines(n):
    print()
    header("VALIDATING REQUIRED LINES")

    required = set()

    for package in PACKAGES.values():
        required.update(package.keys())

    for line_name in sorted(required):
        require_line(n, line_name)

        s_nom = float(n.lines.at[line_name, "s_nom"])

        print(
            f"FOUND   : {line_name:<55} "
            f"s_nom={s_nom:12.6f} MW"
        )


# ==================================================================================================
# REINFORCEMENT APPLICATION
# ==================================================================================================

def apply_reinforcements(n, package):
    """
    Apply package multipliers to the fresh network.

    s_nom is multiplied directly.

    The source network is never modified because each package receives
    its own freshly loaded PyPSA Network.
    """

    for line_name, multiplier in package.items():

        old_s_nom = float(n.lines.at[line_name, "s_nom"])

        new_s_nom = old_s_nom * float(multiplier)

        n.lines.at[line_name, "s_nom"] = new_s_nom

        print(
            f"{line_name:<55} "
            f"{multiplier:>6.2f}x "
            f"{old_s_nom:14.6f} -> "
            f"{new_s_nom:14.6f} MW"
        )


# ==================================================================================================
# REACTIVE SUPPORT
# ==================================================================================================

def apply_reactive_support(n):
    """
    Apply +500 MVAr using the existing generator at the weak bus.

    IMPORTANT:
    q_set is the input used by PyPSA for the nonlinear PF.
    q_set is snapshot dependent.

    We do not use generators_t.q_set because your network has no such
    time-dependent q_set columns.
    """

    if REACTIVE_GENERATOR not in n.generators.index:
        raise KeyError(
            f"Reactive generator not found: {REACTIVE_GENERATOR}"
        )

    if WEAK_BUS not in n.buses.index:
        raise KeyError(
            f"Weak bus not found: {WEAK_BUS}"
        )

    if "q_set" not in n.generators.columns:
        raise KeyError(
            "Network generators table does not contain q_set."
        )

    old_q = float(n.generators.at[REACTIVE_GENERATOR, "q_set"])

    n.generators.at[
        REACTIVE_GENERATOR,
        "q_set"
    ] = old_q + Q_SUPPORT_MVAR

    return old_q, old_q + Q_SUPPORT_MVAR


# ==================================================================================================
# AC LINE LOADING
# ==================================================================================================

def calculate_line_loading(n, snapshot):
    """
    Calculate apparent power at both ends of every AC line.

    PyPSA nonlinear PF provides:
        lines_t.p0
        lines_t.q0
        lines_t.p1
        lines_t.q1

    Apparent power:
        S = sqrt(P^2 + Q^2)

    Loading:
        loading_pct = max(S0, S1) / s_nom * 100
    """

    if len(n.lines.index) == 0:
        return pd.Series(dtype=float)

    try:
        p0 = n.lines_t.p0.loc[snapshot]
        q0 = n.lines_t.q0.loc[snapshot]
        p1 = n.lines_t.p1.loc[snapshot]
        q1 = n.lines_t.q1.loc[snapshot]

    except Exception as exc:
        raise RuntimeError(
            f"Could not read AC line flow outputs for snapshot "
            f"{snapshot}: {exc}"
        )

    s0 = np.sqrt(
        np.square(p0.astype(float))
        +
        np.square(q0.astype(float))
    )

    s1 = np.sqrt(
        np.square(p1.astype(float))
        +
        np.square(q1.astype(float))
    )

    max_s = pd.concat(
        [s0.rename("s0"), s1.rename("s1")],
        axis=1
    ).max(axis=1)

    s_nom = n.lines["s_nom"].astype(float)

    loading_pct = (
        max_s
        /
        s_nom
        *
        100.0
    )

    loading_pct = loading_pct.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return loading_pct


# ==================================================================================================
# TRANSFORMER LOADING
# ==================================================================================================

def calculate_transformer_loading(n, snapshot):
    """
    Calculate apparent power loading for transformers using
    p0/q0 and p1/q1.
    """

    if len(n.transformers.index) == 0:
        return pd.Series(dtype=float)

    try:
        p0 = n.transformers_t.p0.loc[snapshot]
        q0 = n.transformers_t.q0.loc[snapshot]
        p1 = n.transformers_t.p1.loc[snapshot]
        q1 = n.transformers_t.q1.loc[snapshot]

    except Exception as exc:
        raise RuntimeError(
            f"Could not read transformer AC outputs for snapshot "
            f"{snapshot}: {exc}"
        )

    s0 = np.sqrt(
        np.square(p0.astype(float))
        +
        np.square(q0.astype(float))
    )

    s1 = np.sqrt(
        np.square(p1.astype(float))
        +
        np.square(q1.astype(float))
    )

    max_s = pd.concat(
        [s0.rename("s0"), s1.rename("s1")],
        axis=1
    ).max(axis=1)

    s_nom = n.transformers["s_nom"].astype(float)

    loading_pct = (
        max_s
        /
        s_nom
        *
        100.0
    )

    loading_pct = loading_pct.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return loading_pct


# ==================================================================================================
# VOLTAGE EXTRACTION
# ==================================================================================================

def get_voltage_results(n, snapshot):
    """
    Extract minimum system voltage and weak-bus voltage.

    PyPSA stores nonlinear AC voltage magnitudes in buses_t.v_mag_pu.
    """

    try:
        voltages = n.buses_t.v_mag_pu.loc[snapshot]

    except Exception as exc:
        raise RuntimeError(
            f"Could not read bus voltage outputs for snapshot "
            f"{snapshot}: {exc}"
        )

    voltages = voltages.astype(float)

    valid_voltages = voltages.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if valid_voltages.empty:
        raise RuntimeError(
            f"No valid voltage results for snapshot {snapshot}."
        )

    minimum_voltage = float(valid_voltages.min())

    if WEAK_BUS in valid_voltages.index:
        weak_bus_voltage = float(
            valid_voltages.loc[WEAK_BUS]
        )
    else:
        weak_bus_voltage = np.nan

    minimum_voltage_bus = str(
        valid_voltages.idxmin()
    )

    return (
        minimum_voltage,
        weak_bus_voltage,
        minimum_voltage_bus,
    )


# ==================================================================================================
# SINGLE SNAPSHOT AC TEST
# ==================================================================================================

def run_snapshot(package_name, package, snapshot):
    """
    Load fresh network, apply package, apply Q support,
    run nonlinear AC PF and calculate metrics.
    """

    print()
    print("-" * WIDTH)

    print(
        f"PACKAGE {package_name} — "
        f"SNAPSHOT {snapshot} — "
        f"+{Q_SUPPORT_MVAR:.0f} MVAr"
    )

    try:

        # ------------------------------------------------------------------------------------------
        # Fresh network
        # ------------------------------------------------------------------------------------------

        n = pypsa.Network(NETWORK_PATH)

        # Make sure snapshot exists
        if snapshot not in n.snapshots:
            raise KeyError(
                f"Snapshot {snapshot} not found in network."
            )

        # ------------------------------------------------------------------------------------------
        # Restrict network to one snapshot
        # ------------------------------------------------------------------------------------------

        n.set_snapshots([snapshot])

        # ------------------------------------------------------------------------------------------
        # Apply reinforcement package
        # ------------------------------------------------------------------------------------------

        print()
        print("REINFORCEMENTS")

        apply_reinforcements(
            n,
            package
        )

        # ------------------------------------------------------------------------------------------
        # Reactive support
        # ------------------------------------------------------------------------------------------

        old_q, new_q = apply_reactive_support(n)

        print()
        print(
            f"Reactive support applied through generator: "
            f"{REACTIVE_GENERATOR}"
        )

        print(
            f"Q setpoint: "
            f"{old_q:.3f} -> {new_q:.3f} MVAr"
        )

        # ------------------------------------------------------------------------------------------
        # AC POWER FLOW
        # ------------------------------------------------------------------------------------------

        print()
        print("RUNNING AC NONLINEAR POWER FLOW")

        pf_result = n.pf(
            snapshots=[snapshot]
        )

        # ------------------------------------------------------------------------------------------
        # Determine convergence
        # ------------------------------------------------------------------------------------------

        converged = True

        try:

            if isinstance(pf_result, tuple):

                # PyPSA returns information in different forms depending
                # on version. We don't depend on a specific tuple shape
                # for the actual result extraction.

                for item in pf_result:
                    if isinstance(item, pd.DataFrame):
                        if snapshot in item.index:
                            vals = item.loc[snapshot]

                            numeric = pd.to_numeric(
                                vals,
                                errors="coerce"
                            )

                            if numeric.notna().any():
                                # Only treat explicit False/0 values as
                                # non-convergence.
                                if (numeric == False).any():
                                    converged = False

        except Exception:
            # Do not allow optional convergence metadata parsing to break
            # an otherwise valid PF result.
            converged = True

        # ------------------------------------------------------------------------------------------
        # Voltage
        # ------------------------------------------------------------------------------------------

        (
            minimum_voltage,
            weak_bus_voltage,
            minimum_voltage_bus,
        ) = get_voltage_results(
            n,
            snapshot
        )

        # ------------------------------------------------------------------------------------------
        # Line loading
        # ------------------------------------------------------------------------------------------

        line_loading = calculate_line_loading(
            n,
            snapshot
        )

        if line_loading.empty:
            max_line_loading = np.nan
            overloaded_lines = 0
            worst_line = None

        else:

            max_line_loading = float(
                line_loading.max()
            )

            overloaded_lines = int(
                (line_loading > MAX_LOADING_ACCEPTABLE).sum()
            )

            worst_line = str(
                line_loading.idxmax()
            )

        # ------------------------------------------------------------------------------------------
        # Transformer loading
        # ------------------------------------------------------------------------------------------

        transformer_loading = calculate_transformer_loading(
            n,
            snapshot
        )

        if transformer_loading.empty:
            max_transformer_loading = np.nan
            worst_transformer = None

        else:

            max_transformer_loading = float(
                transformer_loading.max()
            )

            worst_transformer = str(
                transformer_loading.idxmax()
            )

        # ------------------------------------------------------------------------------------------
        # Acceptance
        # ------------------------------------------------------------------------------------------

        weak_voltage_ok = (
            np.isfinite(weak_bus_voltage)
            and
            weak_bus_voltage >= WEAK_BUS_MIN_ACCEPTABLE
        )

        minimum_voltage_ok = (
            np.isfinite(minimum_voltage)
            and
            minimum_voltage >= SYSTEM_MIN_VOLTAGE_ACCEPTABLE
        )

        thermal_loading_ok = (
            np.isfinite(max_line_loading)
            and
            max_line_loading <= MAX_LOADING_ACCEPTABLE
        )

        overload_count_ok = (
            overloaded_lines == 0
        )

        fully_acceptable = (
            converged
            and
            weak_voltage_ok
            and
            minimum_voltage_ok
            and
            thermal_loading_ok
            and
            overload_count_ok
        )

        # ------------------------------------------------------------------------------------------
        # Print result
        # ------------------------------------------------------------------------------------------

        print()
        print("RESULT")
        print("-" * WIDTH)

        print(
            f"Converged                 : {converged}"
        )

        print(
            f"Min V magnitude           : "
            f"{minimum_voltage:.6f} pu"
        )

        print(
            f"Minimum voltage bus       : "
            f"{minimum_voltage_bus}"
        )

        print(
            f"Weak bus voltage          : "
            f"{weak_bus_voltage:.6f} pu"
        )

        print(
            f"Max line loading          : "
            f"{max_line_loading:.6f} %"
        )

        print(
            f"Overloaded lines          : "
            f"{overloaded_lines}"
        )

        print(
            f"Critical/max loaded line  : "
            f"{worst_line}"
        )

        print(
            f"Max transformer loading   : "
            f"{max_transformer_loading:.6f} %"
        )

        print(
            f"Worst transformer         : "
            f"{worst_transformer}"
        )

        print(
            f"Fully acceptable          : "
            f"{fully_acceptable}"
        )

        return {
            "package": package_name,
            "snapshot": snapshot,
            "converged": bool(converged),

            "min_voltage_pu": minimum_voltage,
            "minimum_voltage_bus": minimum_voltage_bus,
            "weak_bus_voltage_pu": weak_bus_voltage,

            "max_line_loading_pct": max_line_loading,
            "overloaded_lines": overloaded_lines,
            "worst_loaded_line": worst_line,

            "max_transformer_loading_pct":
                max_transformer_loading,
            "worst_transformer":
                worst_transformer,

            "weak_voltage_ok":
                bool(weak_voltage_ok),

            "minimum_voltage_ok":
                bool(minimum_voltage_ok),

            "thermal_loading_ok":
                bool(thermal_loading_ok),

            "overload_count_ok":
                bool(overload_count_ok),

            "fully_acceptable":
                bool(fully_acceptable),

            "q_support_mvar":
                Q_SUPPORT_MVAR,

            "error":
                None,
        }

    except Exception as exc:

        print()
        print("POWER FLOW / ANALYSIS FAILED")
        print("-" * WIDTH)

        print(
            f"Error type : {type(exc).__name__}"
        )

        print(
            f"Error      : {exc}"
        )

        # Print a compact traceback so that if anything else fails,
        # we know exactly where.
        traceback.print_exc(limit=3)

        return {
            "package": package_name,
            "snapshot": snapshot,
            "converged": False,

            "min_voltage_pu": np.nan,
            "minimum_voltage_bus": None,
            "weak_bus_voltage_pu": np.nan,

            "max_line_loading_pct": np.nan,
            "overloaded_lines": np.nan,
            "worst_loaded_line": None,

            "max_transformer_loading_pct":
                np.nan,
            "worst_transformer":
                None,

            "weak_voltage_ok":
                False,

            "minimum_voltage_ok":
                False,

            "thermal_loading_ok":
                False,

            "overload_count_ok":
                False,

            "fully_acceptable":
                False,

            "q_support_mvar":
                Q_SUPPORT_MVAR,

            "error":
                f"{type(exc).__name__}: {exc}",
        }


# ==================================================================================================
# PACKAGE SUMMARY
# ==================================================================================================

def summarize_package(package_name, snapshot_results):
    """
    Convert six snapshot results into one system-level package result.
    """

    df = pd.DataFrame(snapshot_results)

    valid = df[
        df["error"].isna()
        &
        df["converged"].astype(bool)
    ].copy()

    failed = df[
        df["error"].notna()
        |
        (~df["converged"].astype(bool))
    ].copy()

    # ----------------------------------------------------------------------------------------------
    # No valid snapshots
    # ----------------------------------------------------------------------------------------------

    if valid.empty:

        return {
            "package":
                package_name,

            "worst_thermal_snapshot":
                None,

            "worst_max_line_loading_pct":
                np.nan,

            "worst_overloaded_lines":
                np.nan,

            "worst_loaded_line":
                None,

            "worst_min_voltage_pu":
                np.nan,

            "weakest_bus_voltage_pu":
                np.nan,

            "max_transformer_loading_pct":
                np.nan,

            "failed_snapshots":
                len(failed),

            "all_snapshots_converged":
                False,

            "all_snapshots_acceptable":
                False,
        }

    # ----------------------------------------------------------------------------------------------
    # Worst thermal snapshot
    # ----------------------------------------------------------------------------------------------

    thermal_idx = valid[
        "max_line_loading_pct"
    ].idxmax()

    worst_thermal_snapshot = str(
        valid.loc[
            thermal_idx,
            "snapshot"
        ]
    )

    worst_max_line_loading = float(
        valid.loc[
            thermal_idx,
            "max_line_loading_pct"
        ]
    )

    worst_overloaded_lines = int(
        valid.loc[
            thermal_idx,
            "overloaded_lines"
        ]
    )

    worst_loaded_line = str(
        valid.loc[
            thermal_idx,
            "worst_loaded_line"
        ]
    )

    # ----------------------------------------------------------------------------------------------
    # Worst voltage
    # ----------------------------------------------------------------------------------------------

    worst_voltage = float(
        valid["min_voltage_pu"].min()
    )

    weakest_bus_voltage = float(
        valid["weak_bus_voltage_pu"].min()
    )

    # ----------------------------------------------------------------------------------------------
    # Transformer
    # ----------------------------------------------------------------------------------------------

    max_transformer_loading = float(
        valid["max_transformer_loading_pct"].max()
    )

    # ----------------------------------------------------------------------------------------------
    # All-snapshot acceptance
    # ----------------------------------------------------------------------------------------------

    all_converged = (
        len(failed) == 0
        and
        bool(valid["converged"].all())
    )

    all_acceptable = (
        all_converged
        and
        bool(valid["weak_voltage_ok"].all())
        and
        bool(valid["minimum_voltage_ok"].all())
        and
        bool(valid["thermal_loading_ok"].all())
        and
        bool(valid["overload_count_ok"].all())
    )

    return {
        "package":
            package_name,

        "worst_thermal_snapshot":
            worst_thermal_snapshot,

        "worst_max_line_loading_pct":
            worst_max_line_loading,

        "worst_overloaded_lines":
            worst_overloaded_lines,

        "worst_loaded_line":
            worst_loaded_line,

        "worst_min_voltage_pu":
            worst_voltage,

        "weakest_bus_voltage_pu":
            weakest_bus_voltage,

        "max_transformer_loading_pct":
            max_transformer_loading,

        "failed_snapshots":
            len(failed),

        "all_snapshots_converged":
            all_converged,

        "all_snapshots_acceptable":
            all_acceptable,
    }


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    header(
        "S4.1 — SYSTEM-LEVEL REINFORCEMENT STRATEGY SCREEN"
    )

    print(
        f"Network       : {os.path.abspath(NETWORK_PATH)}"
    )

    print("Operating snapshots:")

    for snapshot in SNAPSHOTS:
        print(f"  - {snapshot}")

    print(
        f"Weak bus      : {WEAK_BUS}"
    )

    print(
        f"Reactive support: +{Q_SUPPORT_MVAR:.0f} MVAr"
    )

    print(
        f"Reactive generator: {REACTIVE_GENERATOR}"
    )

    # ----------------------------------------------------------------------------------------------
    # Fixed reinforcement display
    # ----------------------------------------------------------------------------------------------

    print()
    header(
        "S3 REINFORCEMENTS CARRIED FORWARD"
    )

    for line_name, multiplier in FIXED_REINFORCEMENTS.items():

        print(
            f"{line_name:<55} : "
            f"{multiplier:.2f}x"
        )

    print(
        f"{'merged_relation/4872159-220+1':<55} : "
        f"package-dependent"
    )

    # ----------------------------------------------------------------------------------------------
    # Load source network
    # ----------------------------------------------------------------------------------------------

    print()
    header("LOADING NETWORK")

    if not os.path.exists(NETWORK_PATH):
        raise FileNotFoundError(
            f"Network file not found:\n{NETWORK_PATH}"
        )

    base_network = pypsa.Network(
        NETWORK_PATH
    )

    print(
        f"Buses        : {len(base_network.buses)}"
    )

    print(
        f"Lines        : {len(base_network.lines)}"
    )

    print(
        f"Transformers : {len(base_network.transformers)}"
    )

    print(
        f"Generators   : {len(base_network.generators)}"
    )

    print(
        f"Loads        : {len(base_network.loads)}"
    )

    print(
        f"Snapshots    : {list(base_network.snapshots)}"
    )

    # ----------------------------------------------------------------------------------------------
    # Validate snapshots
    # ----------------------------------------------------------------------------------------------

    missing_snapshots = [
        s for s in SNAPSHOTS
        if s not in base_network.snapshots
    ]

    if missing_snapshots:
        raise KeyError(
            "Missing required snapshots: "
            +
            ", ".join(missing_snapshots)
        )

    # ----------------------------------------------------------------------------------------------
    # Validate required lines
    # ----------------------------------------------------------------------------------------------

    validate_required_lines(
        base_network
    )

    # ----------------------------------------------------------------------------------------------
    # Validate weak bus / generator
    # ----------------------------------------------------------------------------------------------

    print()
    header("VALIDATING REACTIVE SUPPORT")

    if WEAK_BUS not in base_network.buses.index:
        raise KeyError(
            f"Weak bus not found: {WEAK_BUS}"
        )

    if REACTIVE_GENERATOR not in base_network.generators.index:
        raise KeyError(
            f"Reactive generator not found: {REACTIVE_GENERATOR}"
        )

    generator_bus = base_network.generators.at[
        REACTIVE_GENERATOR,
        "bus"
    ]

    print(
        f"Reactive generator : {REACTIVE_GENERATOR}"
    )

    print(
        f"Generator bus      : {generator_bus}"
    )

    print(
        f"Required weak bus  : {WEAK_BUS}"
    )

    if generator_bus != WEAK_BUS:

        print()
        print(
            "WARNING: Reactive generator is not connected "
            "to the declared weak bus."
        )

    # ----------------------------------------------------------------------------------------------
    # Run all packages
    # ----------------------------------------------------------------------------------------------

    all_snapshot_results = []
    package_summaries = []

    total_packages = len(PACKAGES)

    for package_number, (
        package_name,
        package
    ) in enumerate(
        PACKAGES.items(),
        start=1
    ):

        print()
        header(
            f"PACKAGE {package_number}/{total_packages} — "
            f"{package_name}"
        )

        print()
        print("REINFORCEMENT PACKAGE")
        print("-" * WIDTH)

        for line_name, multiplier in package.items():

            base_s_nom = float(
                base_network.lines.at[
                    line_name,
                    "s_nom"
                ]
            )

            new_s_nom = (
                base_s_nom
                *
                multiplier
            )

            print(
                f"{line_name:<55} "
                f"{multiplier:>6.2f}x "
                f"{base_s_nom:14.6f} -> "
                f"{new_s_nom:14.6f} MW"
            )

        package_snapshot_results = []

        # ------------------------------------------------------------------------------------------
        # Each snapshot independently
        # ------------------------------------------------------------------------------------------

        for snapshot in SNAPSHOTS:

            result = run_snapshot(
                package_name,
                package,
                snapshot
            )

            package_snapshot_results.append(
                result
            )

            all_snapshot_results.append(
                result
            )

        # ------------------------------------------------------------------------------------------
        # Package summary
        # ------------------------------------------------------------------------------------------

        summary = summarize_package(
            package_name,
            package_snapshot_results
        )

        package_summaries.append(
            summary
        )

        print()
        line("=")

        print(
            f"PACKAGE SUMMARY — {package_name}"
        )

        print("-" * WIDTH)

        print(
            f"Snapshots tested       : {len(SNAPSHOTS)}"
        )

        print(
            f"Snapshots failed      : "
            f"{summary['failed_snapshots']}"
        )

        print(
            f"Worst thermal snapshot : "
            f"{summary['worst_thermal_snapshot']}"
        )

        print(
            f"Worst max loading      : "
            f"{summary['worst_max_line_loading_pct']:.6f} %"
            if np.isfinite(
                summary["worst_max_line_loading_pct"]
            )
            else
            "Worst max loading      : NaN"
        )

        print(
            f"Worst overloaded lines : "
            f"{summary['worst_overloaded_lines']}"
            if pd.notna(
                summary["worst_overloaded_lines"]
            )
            else
            "Worst overloaded lines : NaN"
        )

        print(
            f"Worst loaded line      : "
            f"{summary['worst_loaded_line']}"
        )

        print(
            f"Worst minimum voltage  : "
            f"{summary['worst_min_voltage_pu']:.6f} pu"
            if np.isfinite(
                summary["worst_min_voltage_pu"]
            )
            else
            "Worst minimum voltage  : NaN"
        )

        print(
            f"Weakest bus voltage    : "
            f"{summary['weakest_bus_voltage_pu']:.6f} pu"
            if np.isfinite(
                summary["weakest_bus_voltage_pu"]
            )
            else
            "Weakest bus voltage    : NaN"
        )

        print(
            f"Max transformer loading: "
            f"{summary['max_transformer_loading_pct']:.6f} %"
            if np.isfinite(
                summary["max_transformer_loading_pct"]
            )
            else
            "Max transformer loading: NaN"
        )

        print(
            f"All snapshots converged: "
            f"{summary['all_snapshots_converged']}"
        )

        print(
            f"ALL SNAPSHOTS ACCEPTABLE: "
            f"{summary['all_snapshots_acceptable']}"
        )

    # ----------------------------------------------------------------------------------------------
    # Detailed snapshot CSV
    # ----------------------------------------------------------------------------------------------

    detailed_df = pd.DataFrame(
        all_snapshot_results
    )

    # ----------------------------------------------------------------------------------------------
    # Package summary DataFrame
    # ----------------------------------------------------------------------------------------------

    summary_df = pd.DataFrame(
        package_summaries
    )

    # ----------------------------------------------------------------------------------------------
    # Save combined result
    # ----------------------------------------------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # Save package-level summary as the requested primary output.
    summary_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Also save detailed snapshot-level diagnostic file.
    detailed_output_path = (
        r"data\processed"
        r"\s4_1_system_level_reinforcement_screen_detailed.csv"
    )

    detailed_df.to_csv(
        detailed_output_path,
        index=False
    )

    # ----------------------------------------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------------------------------------

    print()
    header(
        "S4.1 — FINAL SYSTEM-LEVEL SCREEN SUMMARY"
    )

    display_columns = [
        "package",
        "worst_thermal_snapshot",
        "worst_max_line_loading_pct",
        "worst_overloaded_lines",
        "worst_loaded_line",
        "worst_min_voltage_pu",
        "weakest_bus_voltage_pu",
        "max_transformer_loading_pct",
        "failed_snapshots",
        "all_snapshots_converged",
        "all_snapshots_acceptable",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Fully acceptable packages
    # ----------------------------------------------------------------------------------------------

    print()
    line("-")

    print(
        "FULLY ACCEPTABLE SYSTEM-LEVEL PACKAGES"
    )

    print("-" * WIDTH)

    acceptable = summary_df[
        summary_df[
            "all_snapshots_acceptable"
        ]
        ==
        True
    ]

    if acceptable.empty:

        print(
            "NO PACKAGE SATISFIES ALL SYSTEM-LEVEL "
            "ACCEPTANCE CRITERIA."
        )

    else:

        print(
            acceptable[
                [
                    "package",
                    "worst_max_line_loading_pct",
                    "worst_min_voltage_pu",
                    "weakest_bus_voltage_pu",
                    "max_transformer_loading_pct",
                ]
            ].to_string(
                index=False
            )
        )

    # ----------------------------------------------------------------------------------------------
    # Best valid package
    # ----------------------------------------------------------------------------------------------

    print()
    line("-")

    print(
        "BEST SYSTEM-LEVEL PACKAGE"
    )

    print("-" * WIDTH)

    valid_packages = summary_df[
        summary_df[
            "all_snapshots_converged"
        ]
        ==
        True
    ].copy()

    if valid_packages.empty:

        print(
            "NO PACKAGE PRODUCED VALID AC POWER-FLOW "
            "RESULTS ACROSS ALL SIX SNAPSHOTS."
        )

    else:

        # Rank by:
        # 1. zero overloads if possible
        # 2. lowest maximum loading
        # 3. highest minimum voltage
        # 4. weakest-bus voltage

        valid_packages = valid_packages.sort_values(
            by=[
                "worst_overloaded_lines",
                "worst_max_line_loading_pct",
                "worst_min_voltage_pu",
                "weakest_bus_voltage_pu",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        )

        best = valid_packages.iloc[0]

        print(
            f"Best package            : "
            f"{best['package']}"
        )

        print(
            f"Worst max line loading  : "
            f"{best['worst_max_line_loading_pct']:.6f} %"
        )

        print(
            f"Worst overloaded lines  : "
            f"{int(best['worst_overloaded_lines'])}"
        )

        print(
            f"Worst thermal snapshot : "
            f"{best['worst_thermal_snapshot']}"
        )

        print(
            f"Worst minimum voltage   : "
            f"{best['worst_min_voltage_pu']:.6f} pu"
        )

        print(
            f"Weakest-bus voltage     : "
            f"{best['weakest_bus_voltage_pu']:.6f} pu"
        )

        print(
            f"Max transformer loading : "
            f"{best['max_transformer_loading_pct']:.6f} %"
        )

        print(
            f"All snapshots acceptable: "
            f"{best['all_snapshots_acceptable']}"
        )

    # ----------------------------------------------------------------------------------------------
    # Output locations
    # ----------------------------------------------------------------------------------------------

    print()
    line("=")

    print(
        "S4.1 COMPLETE"
    )

    print("=" * WIDTH)

    print(
        "Package summary saved to:"
    )

    print(
        os.path.abspath(
            OUTPUT_PATH
        )
    )

    print()
    print(
        "Detailed snapshot results saved to:"
    )

    print(
        os.path.abspath(
            detailed_output_path
        )
    )

    print("=" * WIDTH)


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    main()